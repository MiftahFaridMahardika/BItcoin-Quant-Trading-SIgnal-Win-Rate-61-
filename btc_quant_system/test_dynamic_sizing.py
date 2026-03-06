#!/usr/bin/env python3
"""
Dynamic Position Sizing: Fixed vs Dynamic Comparison
======================================================
Compares two sizing strategies across 2017-2024:

  OLD: Fixed 2% risk per trade (constant Kelly)
  NEW: Dynamic — Kelly × Streak × Volatility × Quality

Shows:
  - CAGR, total return, max drawdown per strategy
  - Risk-adjusted return (Return / MaxDD)
  - Per-year breakdown
  - Component multiplier distribution
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from utils.logger import setup_all_loggers
from utils.helpers import format_duration
from engines.signal_engine import SignalEngine
from engines.risk_engine import RiskEngine
from engines.trend_follower import MarketBiasDetector

# ── Colours ──────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"

RISK_CONFIG = str(PROJECT_ROOT / "configs" / "risk_config.yaml")
BASE_RISK   = 0.02        # 2% base risk
LEVERAGE    = 1.0
MAX_FWD     = 168         # ~28 days @ 4H
CAPITAL     = 100_000.0

BULL_YEARS = {2019, 2020, 2023, 2024}
BTC_BH = {2017: 13.19, 2018: -0.72, 2019: 0.94, 2020: 3.02,
           2021: 0.59,  2022: -0.65,  2023: 1.57,  2024: 1.21}


# ──────────────────────────────────────────────────────────────
# SIMULATE TRADE
# ──────────────────────────────────────────────────────────────

def simulate_trade(
    entry_price: float,
    direction:   str,
    atr:         float,
    fwd_high:    np.ndarray,
    fwd_low:     np.ndarray,
    fwd_close:   np.ndarray,
    fwd_atr:     np.ndarray,
) -> dict:
    """Complete trade simulation with tiered trailing stop."""
    sign    = 1 if direction == "LONG" else -1
    sl_dist = atr * 1.5
    sl      = entry_price - sign * sl_dist
    tp1     = entry_price + sign * atr * 1.65
    tp2     = entry_price + sign * atr * 2.75
    tp3     = entry_price + sign * atr * 5.0

    cur_sl    = sl
    tp1_hit   = False
    tp2_hit   = False
    breakeven = False
    total_r   = 0.0

    for i in range(len(fwd_high)):
        h   = fwd_high[i]
        l   = fwd_low[i]
        c   = fwd_close[i]
        a   = float(fwd_atr[i]) if i < len(fwd_atr) and not np.isnan(fwd_atr[i]) and fwd_atr[i] > 0 else atr

        if direction == "LONG":
            if not tp1_hit and h >= tp1:
                total_r += (tp1 - entry_price) / sl_dist * 0.40
                tp1_hit  = True
                cur_sl   = max(cur_sl, entry_price)
                breakeven = True
                continue
            if tp1_hit and not tp2_hit and h >= tp2:
                total_r += (tp2 - entry_price) / sl_dist * 0.30
                tp2_hit  = True
                new_sl   = max(c - a, entry_price)
                if new_sl > cur_sl: cur_sl = new_sl
                continue
            if tp2_hit and h >= tp3:
                total_r += (tp3 - entry_price) / sl_dist * 0.30
                return {"exit": "TP3", "pnl_r": total_r, "bars": i}
            if l <= cur_sl:
                rem = 1.0 - 0.40 * tp1_hit - 0.30 * tp2_hit
                total_r += (cur_sl - entry_price) / sl_dist * rem
                return {"exit": "TRAIL_STOP" if breakeven else "STOP_LOSS",
                        "pnl_r": total_r, "bars": i}
            profit = c - entry_price
            if profit >= 3 * a:   new_sl = c - 0.7 * a
            elif profit >= 2 * a: new_sl = c - 1.0 * a
            elif profit >= a:
                new_sl = max(entry_price, c - 1.5 * a); breakeven = True
            elif profit >= 0.5 * a:
                new_sl = entry_price; breakeven = True
            else: continue
            if new_sl > cur_sl: cur_sl = new_sl
        else:
            if not tp1_hit and l <= tp1:
                total_r += (entry_price - tp1) / sl_dist * 0.40
                tp1_hit  = True
                cur_sl   = min(cur_sl, entry_price)
                breakeven = True
                continue
            if tp1_hit and not tp2_hit and l <= tp2:
                total_r += (entry_price - tp2) / sl_dist * 0.30
                tp2_hit  = True
                new_sl   = min(c + a, entry_price)
                if new_sl < cur_sl: cur_sl = new_sl
                continue
            if tp2_hit and l <= tp3:
                total_r += (entry_price - tp3) / sl_dist * 0.30
                return {"exit": "TP3", "pnl_r": total_r, "bars": i}
            if h >= cur_sl:
                rem = 1.0 - 0.40 * tp1_hit - 0.30 * tp2_hit
                total_r += (entry_price - cur_sl) / sl_dist * rem
                return {"exit": "TRAIL_STOP" if breakeven else "STOP_LOSS",
                        "pnl_r": total_r, "bars": i}
            profit = entry_price - c
            if profit >= 3 * a:   new_sl = c + 0.7 * a
            elif profit >= 2 * a: new_sl = c + 1.0 * a
            elif profit >= a:
                new_sl = min(entry_price, c + 1.5 * a); breakeven = True
            elif profit >= 0.5 * a:
                new_sl = entry_price; breakeven = True
            else: continue
            if new_sl < cur_sl: cur_sl = new_sl

    rem   = 1.0 - 0.40 * tp1_hit - 0.30 * tp2_hit
    pnl_r = (fwd_close[-1] - entry_price) / sl_dist * rem * sign
    return {"exit": "TIME_EXIT", "pnl_r": pnl_r, "bars": len(fwd_high)}


# ──────────────────────────────────────────────────────────────
# BACKTEST LOOP
# ──────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, use_dynamic: bool = False) -> dict:
    """
    One-trade-at-a-time walk-forward backtest.
    When a signal fires, open a trade and skip until it closes.
    Cool-off: skip 12 bars after a trade closes.
    """
    risk_engine   = RiskEngine(RISK_CONFIG)
    bias_detector = MarketBiasDetector() if use_dynamic else None

    close_arr = df["Close"].values
    high_arr  = df["High"].values
    low_arr   = df["Low"].values
    atr_arr   = df.get("atr_14", df["Close"] * 0.02).fillna(df["Close"] * 0.02).values
    n_total   = len(df)

    # 200-bar rolling median ATR as historical baseline
    hist_atr_ser = pd.Series(atr_arr).rolling(200, min_periods=50).median().values

    balance   = CAPITAL
    peak      = CAPITAL
    max_dd    = 0.0
    equity    = {df.index[0]: CAPITAL}
    results   = []
    history   = []

    COOLOFF   = 12          # bars to skip after a trade closes
    next_trade_bar = 0      # earliest bar index we can open next trade

    for ts, row in df.iterrows():
        idx = df.index.get_loc(ts)
        if idx < 200 or idx < next_trade_bar:
            continue

        sig_type = row.get("signal_type", "SKIP")
        if sig_type not in ("STRONG_LONG", "LONG", "SHORT", "STRONG_SHORT"):
            continue

        direction   = str(sig_type).replace("STRONG_", "")
        entry_price = float(row["Close"])
        atr         = float(atr_arr[idx])
        if np.isnan(atr) or atr <= 0:
            atr = entry_price * 0.02
        hist_atr = float(hist_atr_ser[idx])
        if np.isnan(hist_atr) or hist_atr <= 0:
            hist_atr = atr

        end_idx   = min(idx + MAX_FWD + 1, n_total)
        fwd_high  = high_arr[idx + 1: end_idx]
        fwd_low   = low_arr[idx + 1: end_idx]
        fwd_close = close_arr[idx + 1: end_idx]
        fwd_atr   = atr_arr[idx + 1: end_idx]

        if len(fwd_high) == 0:
            continue

        outcome = simulate_trade(
            entry_price, direction, atr,
            fwd_high, fwd_low, fwd_close, fwd_atr
        )
        pnl_r       = outcome["pnl_r"]
        bars_held   = outcome["bars"]

        # Advance cursor past this trade + cool-off
        next_trade_bar = idx + bars_held + COOLOFF + 1

        # ── Position sizing ────────────────────────────────
        sl_price = (entry_price - atr * 1.5
                    if direction == "LONG"
                    else entry_price + atr * 1.5)

        if use_dynamic:
            bias_state  = bias_detector.detect_bias(df.iloc[:idx + 1])
            market_bias = bias_state.bias

            qual_raw = float(row.get("signal_quality",
                                     row.get("composite_score", 70.0)) or 70.0)
            # normalise to 0-100
            qual_raw = max(0.0, min(qual_raw * 10 if qual_raw <= 10 else qual_raw, 100.0))

            sizing = risk_engine.calculate_optimal_position_size(
                entry_price    = entry_price,
                stop_loss      = sl_price,
                recent_trades  = history,
                market_bias    = market_bias,
                regime         = "NORMAL",
                current_atr    = atr,
                historical_atr = hist_atr,
                signal_quality = qual_raw,
                leverage       = LEVERAGE,
                base_risk_pct  = BASE_RISK,
            )
            risk_amount = sizing["risk_amount"]
        else:
            risk_amount = balance * BASE_RISK    # 2% of current balance

        pnl_dollar = pnl_r * risk_amount

        balance += pnl_dollar
        balance  = max(balance, 1.0)   # floor to prevent negative
        if balance > peak:
            peak = balance

        dd = (peak - balance) / peak * 100
        if dd > max_dd:
            max_dd = dd

        risk_engine.account_balance  = balance
        risk_engine.current_drawdown = dd

        trade_rec = {
            "year":      ts.year,
            "direction": direction,
            "pnl_r":     pnl_r,
            "pnl":       pnl_dollar,
            "risk_pct":  risk_amount / max(balance - pnl_dollar, 1) * 100,
            "exit":      outcome["exit"],
        }
        results.append(trade_rec)
        history.append({"pnl": pnl_dollar, "pnl_r": pnl_r})
        equity[ts] = balance

    total_return = (balance - CAPITAL) / CAPITAL * 100
    n_years = max(1, (df.index[-1] - df.index[0]).days / 365.25)
    cagr    = ((balance / CAPITAL) ** (1 / n_years) - 1) * 100
    rar     = total_return / max(max_dd, 0.01)

    return {
        "results":       results,
        "equity":        equity,
        "final_balance": balance,
        "total_return":  total_return,
        "cagr":          cagr,
        "max_drawdown":  max_dd,
        "rar":           rar,
    }


# ──────────────────────────────────────────────────────────────
# REPORT HELPERS
# ──────────────────────────────────────────────────────────────

def year_stats(results: list, year: int) -> dict:
    yr = [r for r in results if r.get("year") == year]
    if not yr:
        return {}
    wins   = [r for r in yr if r["pnl"] > 0]
    total_pnl = sum(r["pnl"] for r in yr)
    return {
        "n":          len(yr),
        "wr":         len(wins) / len(yr) * 100,
        "total_pnl":  total_pnl,
        "avg_risk_pct": np.mean([r["risk_pct"] for r in yr]) * 100,
        "sl_pct":     sum(1 for r in yr if r["exit"] == "STOP_LOSS") / len(yr) * 100,
    }


def print_overall(label: str, stats: dict, color: str) -> None:
    print(f"\n{color}{'─' * 56}{R}")
    print(f"{B}{color}  {label}{R}")
    print(f"{color}{'─' * 56}{R}")
    print(f"  Total return   : {color}{stats['total_return']:>+7.1f}%{R}")
    print(f"  CAGR           : {color}{stats['cagr']:>+7.1f}%/yr{R}")
    print(f"  Max drawdown   : {RED}{stats['max_drawdown']:>7.1f}%{R}")
    print(f"  Return / MaxDD : {G}{stats['rar']:>7.2f}×{R}")
    print(f"  Final balance  : ${stats['final_balance']:>12,.0f}")


def print_year_table(old: dict, new: dict) -> None:
    print(f"\n{C}{'═'*72}")
    print(f"  PER-YEAR BREAKDOWN")
    print(f"{'═'*72}{R}")
    print(f"  {'Year':<5} {'B&H':>6} │ "
          f"{'Old Ret':>8} {'Old WR':>7} {'Old DD rsk':>10} │ "
          f"{'New Ret':>8} {'New WR':>7} {'New DD rsk':>10}")
    print(f"  {'─'*70}")

    years = sorted(set(r["year"] for r in old["results"]))
    for yr in years:
        o      = year_stats(old["results"], yr)
        n      = year_stats(new["results"], yr)
        bh_pct = BTC_BH.get(yr, 0) * 100
        bull   = yr in BULL_YEARS
        col    = G if bull else RED if bh_pct < -20 else ""
        print(
            f"  {col}{yr}{R} {bh_pct:>+6.0f}% │ "
            f"{o.get('total_pnl', 0):>+8.0f} "
            f"{o.get('wr', 0):>6.1f}% "
            f"{o.get('avg_risk_pct', 0):>9.2f}%  │ "
            f"{G if n.get('total_pnl', 0) > o.get('total_pnl', 0) else ''}"
            f"{n.get('total_pnl', 0):>+8.0f}{R} "
            f"{n.get('wr', 0):>6.1f}% "
            f"{G}{n.get('avg_risk_pct', 0):>9.2f}%{R}"
        )


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print(f"""
{C}{'═'*60}
  DYNAMIC POSITION SIZING: FIXED vs DYNAMIC (2017-2024)
{'═'*60}{R}
""")
    setup_all_loggers(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")

    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if not feat_path.exists():
        print(f"{RED}Feature cache not found: {feat_path}{R}")
        return

    print(f"{B}[1/3]{R} Loading data and generating signals...")
    t0 = time.perf_counter()
    df = pd.read_parquet(feat_path)
    se = SignalEngine()
    df = se.generate_signals_batch(df, start_idx=200)
    df_range = df.loc["2017-01-01":"2024-12-31"]
    print(f"  ✓ {len(df_range):,} bars, {len(df_range[df_range['signal_type'].isin(['STRONG_LONG','LONG','SHORT','STRONG_SHORT'])]):,} signals")

    print(f"\n{B}[2/3]{R} Running backtests...")
    old_stats = run_backtest(df_range, use_dynamic=False)
    print(f"  ✓ OLD (fixed 2%) done")
    new_stats = run_backtest(df_range, use_dynamic=True)
    print(f"  ✓ NEW (dynamic)  done  [{format_duration(time.perf_counter() - t0)}]")

    print(f"\n{B}[3/3]{R} Results...")
    print_overall("OLD — Fixed 2% Risk Per Trade", old_stats, RED)
    print_overall("NEW — Dynamic Kelly × Streak × Vol × Quality", new_stats, G)
    print_year_table(old_stats, new_stats)

    # ── Final comparison table ──────────────────────────────
    d_ret = new_stats["total_return"] - old_stats["total_return"]
    d_dd  = new_stats["max_drawdown"] - old_stats["max_drawdown"]
    d_rar = new_stats["rar"] - old_stats["rar"]

    print(f"""
{C}{'═'*60}
  FINAL COMPARISON
{'═'*60}{R}
  {'Metric':<26} │ {'OLD':>10} │ {'NEW':>10} │ {'Delta':>8}
  {'─'*58}
  {'Total Return':<26} │ {old_stats['total_return']:>+9.1f}% │ {new_stats['total_return']:>+9.1f}% │ {G if d_ret>0 else RED}{d_ret:>+7.1f}pp{R}
  {'CAGR':<26} │ {old_stats['cagr']:>+9.1f}% │ {new_stats['cagr']:>+9.1f}% │
  {'Max Drawdown':<26} │ {old_stats['max_drawdown']:>9.1f}% │ {new_stats['max_drawdown']:>9.1f}% │ {G if d_dd<0 else RED}{d_dd:>+7.1f}pp{R}
  {'Return / MaxDD':<26} │ {old_stats['rar']:>10.2f} │ {new_stats['rar']:>10.2f} │ {G if d_rar>0 else RED}{d_rar:>+7.2f}×{R}
  {'─'*58}

  Risk control (DD < 20%): {G + '✓ PASS' + R if new_stats['max_drawdown'] < 20 else RED + '✗ FAIL' + R}
  Total time : {format_duration(time.perf_counter() - t0)}
{G}{'═'*60}{R}
""")


if __name__ == "__main__":
    main()
