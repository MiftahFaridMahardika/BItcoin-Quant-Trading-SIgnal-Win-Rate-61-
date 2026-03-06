#!/usr/bin/env python3
"""
BTC Quant System — Yearly Backtest (Fixed Trade Plan)
======================================================
Trade Plan
----------
  POSITION : Fixed $1,000 collateral × 15x leverage = $15,000 notional
  SL       : 1.333% from entry  → -$200 per trade  = -20% account
             LONG  : entry × 0.98667
             SHORT : entry × 1.01333
  TP       : Trailing TP (dynamic)
             Activation  : 0.71% profit from entry
             Trail dist  : 1.333% from peak/trough (same as SL)
             Exit logic  : once activated, trailing stop updates every bar.
                           Exit when trailing stop is hit OR on signal reversal.
  MAX RISK : 21.2% per trade (20% + 0.12% fees round-trip on notional)
  HOLDING  : 4 hours candle; decision & management every 4H bar (trade can span multiple bars)
  PERIOD   : 2017 – 2026
  REPORT   : per-year with LONG and SHORT breakdown

Usage
-----
  python yearly_backtest.py
  python yearly_backtest.py --start_year 2019 --end_year 2024
  python yearly_backtest.py --no_skip_filter   # allow LONG and SHORT only (no STRONG filter)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from engines.data_pipeline import DataPipeline
from engines.feature_engine import FeatureEngine
from engines.signal_engine import SignalEngine

# ── ANSI colours ─────────────────────────────────────────────
C   = "\033[36m"   # cyan
G   = "\033[32m"   # green
Y   = "\033[33m"   # yellow
RED = "\033[31m"   # red
B   = "\033[1m"    # bold
D   = "\033[2m"    # dim
R   = "\033[0m"    # reset

W   = 110
SEP = f"{C}{'═' * W}{R}"
SUB = f"  {'─' * (W - 2)}"


# ══════════════════════════════════════════════════════════════
# TRADE PLAN CONSTANTS
# ══════════════════════════════════════════════════════════════

COLLATERAL        = 1_000.0          # USD per trade (fixed)
LEVERAGE          = 15.0
NOTIONAL          = COLLATERAL * LEVERAGE   # 15,000 USD
SL_PCT            = 0.01333          # 1.333% SL distance from entry
TP_ACTIVATION_PCT = 0.00710          # 0.71% minimum profit before trailing
TRAIL_DIST_PCT    = SL_PCT           # trail distance = same as SL
TAKER_FEE_PCT     = 0.0004           # 0.04% taker fee per side
FEE_ROUND_TRIP    = 2 * TAKER_FEE_PCT * NOTIONAL   # ≈ $12 per trade (approx; exact fee uses exit notional)
MAX_RISK_PCT      = 0.212            # 21.2% of collateral max loss (incl fee)
WARMUP_BARS       = 200              # skip first 200 bars (indicator warmup)


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION (4H, multi-bar; exit on reversal)
# ══════════════════════════════════════════════════════════════

@dataclass
class OpenTrade:
    direction: str                 # LONG | SHORT
    entry_signal: str              # LONG | STRONG_LONG | SHORT | STRONG_SHORT
    entry_time: pd.Timestamp
    entry_idx: int                 # positional index within df_period
    entry_price: float
    quantity: float
    entry_fee_usd: float
    sl_price: float
    tp_activation_price: float
    peak_or_trough: float          # peak for LONG, trough for SHORT
    trailing_active: bool = False
    trailing_stop: Optional[float] = None


def _fees(entry_price: float, exit_price: float, quantity: float) -> Tuple[float, float]:
    """
    Compute entry/exit taker fees in USD.
    Fee base is notional value on each side.
    """
    entry_notional = entry_price * quantity
    exit_notional = exit_price * quantity
    return (TAKER_FEE_PCT * entry_notional, TAKER_FEE_PCT * exit_notional)


def _pnl_usd(direction: str, entry_price: float, exit_price: float, quantity: float) -> float:
    if direction == "LONG":
        return (exit_price - entry_price) * quantity
    return (entry_price - exit_price) * quantity


def _exit_price_with_slippage(direction: str, price: float, slippage_pct: float) -> float:
    # Market exit: LONG sells (slippage down), SHORT buys (slippage up)
    if direction == "LONG":
        return price * (1 - slippage_pct)
    return price * (1 + slippage_pct)


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(trades: list) -> dict:
    """Compute summary stats from a list of trade dicts."""
    if not trades:
        return {
            "n_trades":        0,
            "win_rate":        0.0,
            "pnl_usd":         0.0,
            "profit_factor":   0.0,
            "avg_win_usd":     0.0,
            "avg_loss_usd":    0.0,
            "max_drawdown_usd":0.0,
            "expectancy_usd":  0.0,
            "sl_pct":          0.0,
            "trail_pct":       0.0,
            "reversal_pct":    0.0,
            "eod_pct":         0.0,
            "sharpe":          float("nan"),
            "avg_fees_usd":    0.0,
            "avg_bars_held":   float("nan"),
        }

    pnls     = [t["pnl_usd"] for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]
    n        = len(pnls)
    win_rate = len(wins) / n if n else 0.0
    pnl_tot  = sum(pnls)

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win      = np.mean(wins) if wins else 0.0
    avg_loss     = np.mean(losses) if losses else 0.0
    expectancy   = np.mean(pnls)

    # Max drawdown on cumulative PnL equity curve
    equity  = np.cumsum(pnls)
    peak    = np.maximum.accumulate(equity)
    dd      = equity - peak
    max_dd  = float(dd.min())

    # Annualised Sharpe (4H bars, 6 bars/day, ~2190/year)
    if n > 1:
        pnl_arr = np.array(pnls)
        pnl_ret = pnl_arr / COLLATERAL          # return as fraction of collateral
        sharpe  = (np.mean(pnl_ret) / np.std(pnl_ret, ddof=1)) * np.sqrt(2190)
    else:
        sharpe  = float("nan")

    exit_counts = {}
    for t in trades:
        k = t["exit_reason"]
        exit_counts[k] = exit_counts.get(k, 0) + 1

    avg_fees = float(np.mean([t.get("fees_usd", 0.0) for t in trades])) if trades else 0.0
    bars_held = [t.get("bars_held") for t in trades if t.get("bars_held") is not None]
    avg_bars = float(np.mean(bars_held)) if bars_held else float("nan")

    return {
        "n_trades":         n,
        "win_rate":         win_rate,
        "pnl_usd":          pnl_tot,
        "profit_factor":    pf,
        "avg_win_usd":      avg_win,
        "avg_loss_usd":     avg_loss,
        "max_drawdown_usd": max_dd,
        "expectancy_usd":   expectancy,
        "sl_pct":           exit_counts.get("SL", 0) / n,
        "trail_pct":        exit_counts.get("TRAIL_STOP", 0) / n,
        "reversal_pct":     exit_counts.get("REVERSAL", 0) / n,
        "eod_pct":          exit_counts.get("EOD", 0) / n,
        "sharpe":           sharpe,
        "avg_fees_usd":     avg_fees,
        "avg_bars_held":    avg_bars,
    }


# ══════════════════════════════════════════════════════════════
# BACKTEST RUNNER
# ══════════════════════════════════════════════════════════════

def run_backtest(
    df: pd.DataFrame,
    sig_df: pd.DataFrame,
    start_year: int = 2017,
    end_year: int   = 2026,
    slippage_pct: float = 0.0005,
    valid_signals: set  = None,
) -> dict:
    """
    Multi-bar trade simulation on 4H candles.

    Anti-lookahead convention:
      - Signal is computed from bar close at time t.
      - Entry occurs at next bar open (t+1) if signal at t is actionable.
      - Position management uses OHLC within each bar.
      - Exit occurs via:
          1) Stop loss (fixed 1.333% from entry)
          2) Trailing stop (activated after +0.71% move; trails peak/trough by 1.333%)
          3) Signal reversal at bar close (opposite actionable signal)

    Returns nested dict {exit_year: {direction: [trade_dicts]}}.
    """
    if valid_signals is None:
        valid_signals = {"LONG", "STRONG_LONG", "SHORT", "STRONG_SHORT"}

    # Filter to year range
    ts_start = pd.Timestamp(f"{start_year}-01-01", tz="UTC")
    ts_end   = pd.Timestamp(f"{end_year}-12-31 23:59:59", tz="UTC")

    # Handle tz-naive index
    if df.index.tz is None:
        ts_start = ts_start.tz_localize(None)
        ts_end   = ts_end.tz_localize(None)

    mask = (df.index >= ts_start) & (df.index <= ts_end)
    df_period  = df[mask]
    sig_period = sig_df.reindex(df_period.index)

    results = defaultdict(lambda: defaultdict(list))   # results[year][direction]

    sig_arr = (
        sig_period["signal_type"].values
        if isinstance(sig_period, pd.DataFrame) and "signal_type" in sig_period.columns
        else np.array(["SKIP"] * len(df_period), dtype=object)
    )

    open_trade: Optional[OpenTrade] = None

    for i in range(len(df_period)):
        ts = df_period.index[i]
        row = df_period.iloc[i]

        open_  = float(row.get("Open", row.get("open", 0)))
        high_  = float(row.get("High", row.get("high", 0)))
        low_   = float(row.get("Low", row.get("low", 0)))
        close_ = float(row.get("Close", row.get("close", 0)))
        if open_ <= 0:
            continue

        sig_now  = sig_arr[i] if i < len(sig_arr) else "SKIP"
        sig_prev = sig_arr[i - 1] if i - 1 >= 0 and i - 1 < len(sig_arr) else "SKIP"

        # ── Entry at this bar OPEN based on previous bar signal ────────────
        if open_trade is None and i > 0 and sig_prev in valid_signals:
            direction = "LONG" if sig_prev in ("LONG", "STRONG_LONG") else "SHORT"

            if direction == "LONG":
                entry_price = open_ * (1 + slippage_pct)
                sl_price = entry_price * (1 - SL_PCT)
                tp_act = entry_price * (1 + TP_ACTIVATION_PCT)
            else:
                entry_price = open_ * (1 - slippage_pct)
                sl_price = entry_price * (1 + SL_PCT)
                tp_act = entry_price * (1 - TP_ACTIVATION_PCT)

            quantity = NOTIONAL / entry_price
            entry_fee, _ = _fees(entry_price, entry_price, quantity)

            open_trade = OpenTrade(
                direction=direction,
                entry_signal=str(sig_prev),
                entry_time=ts,
                entry_idx=i,
                entry_price=entry_price,
                quantity=quantity,
                entry_fee_usd=entry_fee,
                sl_price=sl_price,
                tp_activation_price=tp_act,
                peak_or_trough=entry_price,
                trailing_active=False,
                trailing_stop=None,
            )

        if open_trade is None:
            continue

        # ── Manage open position within this bar (OHLC only) ───────────────
        exit_reason = None
        exit_price = None

        if open_trade.direction == "LONG":
            # Assume path O -> H -> L -> C (conservative for trailing)
            if high_ > open_trade.peak_or_trough:
                open_trade.peak_or_trough = high_

            if (not open_trade.trailing_active) and (open_trade.peak_or_trough >= open_trade.tp_activation_price):
                open_trade.trailing_active = True

            if open_trade.trailing_active:
                open_trade.trailing_stop = open_trade.peak_or_trough * (1 - TRAIL_DIST_PCT)
                if low_ <= open_trade.trailing_stop:
                    exit_reason = "TRAIL_STOP"
                    exit_price = open_trade.trailing_stop

            if exit_reason is None and low_ <= open_trade.sl_price:
                exit_reason = "SL"
                exit_price = open_trade.sl_price

        else:  # SHORT
            # Assume path O -> L -> H -> C (conservative for trailing)
            if low_ < open_trade.peak_or_trough:
                open_trade.peak_or_trough = low_

            if (not open_trade.trailing_active) and (open_trade.peak_or_trough <= open_trade.tp_activation_price):
                open_trade.trailing_active = True

            if open_trade.trailing_active:
                open_trade.trailing_stop = open_trade.peak_or_trough * (1 + TRAIL_DIST_PCT)
                if high_ >= open_trade.trailing_stop:
                    exit_reason = "TRAIL_STOP"
                    exit_price = open_trade.trailing_stop

            if exit_reason is None and high_ >= open_trade.sl_price:
                exit_reason = "SL"
                exit_price = open_trade.sl_price

        # ── Signal reversal exit at CLOSE (if still open) ──────────────────
        if exit_reason is None and sig_now in valid_signals:
            desired_dir = "LONG" if sig_now in ("LONG", "STRONG_LONG") else "SHORT"
            if desired_dir != open_trade.direction:
                exit_reason = "REVERSAL"
                exit_price = _exit_price_with_slippage(open_trade.direction, close_, slippage_pct)

        # ── Force close at end-of-period ───────────────────────────────────
        if exit_reason is None and i == len(df_period) - 1:
            exit_reason = "EOD"
            exit_price = _exit_price_with_slippage(open_trade.direction, close_, slippage_pct)

        if exit_reason is None or exit_price is None:
            continue

        _, exit_fee = _fees(open_trade.entry_price, exit_price, open_trade.quantity)
        gross = _pnl_usd(open_trade.direction, open_trade.entry_price, exit_price, open_trade.quantity)
        pnl_usd = gross - open_trade.entry_fee_usd - exit_fee

        trade = {
            "timestamp": ts,  # exit timestamp (for sorting / equity)
            "entry_time": open_trade.entry_time,
            "exit_time": ts,
            "direction": open_trade.direction,
            "signal": open_trade.entry_signal,
            "entry_price": open_trade.entry_price,
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "quantity": float(open_trade.quantity),
            "entry_fee_usd": float(open_trade.entry_fee_usd),
            "exit_fee_usd": float(exit_fee),
            "fees_usd": float(open_trade.entry_fee_usd + exit_fee),
            "pnl_usd": float(pnl_usd),
            "pnl_pct_collateral": float(pnl_usd / COLLATERAL),
            "bars_held": int(i - open_trade.entry_idx + 1),
            "trailing_activated": bool(open_trade.trailing_active),
        }

        results[ts.year][open_trade.direction].append(trade)
        open_trade = None

    return results


# ══════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════

def _c(v: float, good_positive: bool = True) -> str:
    if not np.isfinite(v):
        return D
    if good_positive:
        return G if v > 0 else (RED if v < 0 else D)
    return G if v < 0 else (RED if v > 0 else D)


def print_yearly_report(
    results: dict,
    start_year: int,
    end_year: int,
) -> None:
    """Print per-year long/short breakdown + overall summary."""

    years = sorted(results.keys())
    all_long_trades  = []
    all_short_trades = []
    cumulative_pnl   = 0.0

    # ── Header ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"{C}{B}  BTC QUANT — YEARLY BACKTEST  {start_year}–{end_year}{R}")
    print(f"{D}  Position: ${COLLATERAL:,.0f} × {LEVERAGE:.0f}x = ${NOTIONAL:,.0f} notional | "
          f"SL: {SL_PCT*100:.3f}% | TP: trailing from {TP_ACTIVATION_PCT*100:.2f}% | "
          f"Fee RT: ${FEE_ROUND_TRIP:.2f}{R}")
    print(SEP)

    # ── Column header ────────────────────────────────────────
    print(f"\n  {B}{'YEAR':>4s}  {'DIR':>5s}  {'N':>5s}  {'WIN%':>6s}  "
          f"{'PnL ($)':>10s}  {'PF':>5s}  {'AvgW':>7s}  {'AvgL':>7s}  "
          f"{'MaxDD':>8s}  {'Expect':>7s}  {'SL%':>5s}  {'TR%':>5s}  {'REV%':>5s}  "
          f"{'Fees':>7s}  {'Hold':>5s}  {'Sharpe':>6s}{R}")
    print(SUB)

    year_summaries = []

    for year in range(start_year, end_year + 1):
        if year not in results:
            continue

        long_trades  = results[year].get("LONG",  [])
        short_trades = results[year].get("SHORT", [])

        all_long_trades.extend(long_trades)
        all_short_trades.extend(short_trades)

        for direction, trades in [("LONG", long_trades), ("SHORT", short_trades)]:
            if not trades:
                print(f"  {year:4d}  {direction:>5s}  {'—':>5s}")
                continue

            m = compute_metrics(trades)
            pnl_col = _c(m["pnl_usd"])
            pf_val  = min(m["profit_factor"], 9.99)  # cap display

            print(
                f"  {year:4d}  {direction:>5s}  "
                f"{m['n_trades']:>5d}  "
                f"{m['win_rate']*100:>6.1f}%  "
                f"{pnl_col}{m['pnl_usd']:>+10,.2f}{R}  "
                f"{pf_val:>5.2f}  "
                f"{m['avg_win_usd']:>+7.2f}  "
                f"{m['avg_loss_usd']:>+7.2f}  "
                f"{m['max_drawdown_usd']:>+8.2f}  "
                f"{m['expectancy_usd']:>+7.2f}  "
                f"{m['sl_pct']*100:>5.1f}%  "
                f"{m['trail_pct']*100:>5.1f}%  "
                f"{m['reversal_pct']*100:>5.1f}%  "
                f"{m['avg_fees_usd']:>7.2f}  "
                f"{m['avg_bars_held']:>5.1f}  "
                f"{m['sharpe']:>6.2f}"
            )

        # Year total
        all_year = long_trades + short_trades
        if all_year:
            ym = compute_metrics(all_year)
            year_pnl = ym["pnl_usd"]
            cumulative_pnl += year_pnl
            col = _c(year_pnl)
            print(
                f"  {D}{'':>4s}  {'TOTAL':>5s}  "
                f"{ym['n_trades']:>5d}  "
                f"{ym['win_rate']*100:>6.1f}%  "
                f"{col}{year_pnl:>+10,.2f}{R}"
            )
            year_summaries.append((year, ym))

        print(SUB)

    # ── Overall summary ──────────────────────────────────────
    all_trades = all_long_trades + all_short_trades
    if not all_trades:
        print(f"\n  {RED}No trades executed in period.{R}\n")
        return

    om = compute_metrics(all_trades)
    lm = compute_metrics(all_long_trades)
    sm = compute_metrics(all_short_trades)

    print(f"\n{SEP}")
    print(f"{C}{B}  OVERALL SUMMARY  ({start_year}–{end_year}){R}")
    print(SEP)

    # Best/worst year
    if year_summaries:
        best  = max(year_summaries, key=lambda x: x[1]["pnl_usd"])
        worst = min(year_summaries, key=lambda x: x[1]["pnl_usd"])

    col_tot = _c(om["pnl_usd"])
    print(f"""
  {'Metric':<28s}  {'All':>12s}  {'Long':>12s}  {'Short':>12s}
  {'─'*70}
  {'Trades':<28s}  {om['n_trades']:>12d}  {lm['n_trades']:>12d}  {sm['n_trades']:>12d}
  {'Win Rate':<28s}  {om['win_rate']*100:>11.1f}%  {lm['win_rate']*100:>11.1f}%  {sm['win_rate']*100:>11.1f}%
  {'Total PnL ($)':<28s}  {col_tot}{om['pnl_usd']:>+12,.2f}{R}  {_c(lm['pnl_usd'])}{lm['pnl_usd']:>+12,.2f}{R}  {_c(sm['pnl_usd'])}{sm['pnl_usd']:>+12,.2f}{R}
  {'Profit Factor':<28s}  {min(om['profit_factor'],9.99):>12.3f}  {min(lm['profit_factor'],9.99):>12.3f}  {min(sm['profit_factor'],9.99):>12.3f}
  {'Avg Win ($)':<28s}  {om['avg_win_usd']:>+12.2f}  {lm['avg_win_usd']:>+12.2f}  {sm['avg_win_usd']:>+12.2f}
  {'Avg Loss ($)':<28s}  {om['avg_loss_usd']:>+12.2f}  {lm['avg_loss_usd']:>+12.2f}  {sm['avg_loss_usd']:>+12.2f}
  {'Expectancy ($)':<28s}  {om['expectancy_usd']:>+12.2f}  {lm['expectancy_usd']:>+12.2f}  {sm['expectancy_usd']:>+12.2f}
  {'Max Drawdown ($)':<28s}  {om['max_drawdown_usd']:>+12.2f}  {lm['max_drawdown_usd']:>+12.2f}  {sm['max_drawdown_usd']:>+12.2f}
  {'Annualised Sharpe':<28s}  {om['sharpe']:>12.3f}  {lm['sharpe']:>12.3f}  {sm['sharpe']:>12.3f}
  {'Exit: SL':<28s}  {om['sl_pct']*100:>11.1f}%  {lm['sl_pct']*100:>11.1f}%  {sm['sl_pct']*100:>11.1f}%
  {'Exit: Trailing Stop':<28s}  {om['trail_pct']*100:>11.1f}%  {lm['trail_pct']*100:>11.1f}%  {sm['trail_pct']*100:>11.1f}%
  {'Exit: Reversal':<28s}  {om['reversal_pct']*100:>11.1f}%  {lm['reversal_pct']*100:>11.1f}%  {sm['reversal_pct']*100:>11.1f}%
  {'Avg Fees ($)':<28s}  {om['avg_fees_usd']:>12.2f}  {lm['avg_fees_usd']:>12.2f}  {sm['avg_fees_usd']:>12.2f}
  {'Avg Hold (bars)':<28s}  {om['avg_bars_held']:>12.1f}  {lm['avg_bars_held']:>12.1f}  {sm['avg_bars_held']:>12.1f}""")

    if year_summaries:
        bc = _c(best[1]["pnl_usd"])
        wc = _c(worst[1]["pnl_usd"])
        print(f"\n  {'Best Year':<28s}  {bc}{best[0]} (${best[1]['pnl_usd']:+,.2f}){R}")
        print(f"  {'Worst Year':<28s}  {wc}{worst[0]} (${worst[1]['pnl_usd']:+,.2f}){R}")

    # R:R summary
    wins_all  = [t["pnl_usd"] for t in all_trades if t["pnl_usd"] > 0]
    losses_all = [abs(t["pnl_usd"]) for t in all_trades if t["pnl_usd"] <= 0]
    avg_w = np.mean(wins_all)   if wins_all   else 0
    avg_l = np.mean(losses_all) if losses_all else 0
    rr    = avg_w / avg_l       if avg_l > 0  else float("nan")
    print(f"  {'Avg R:R (realised)':<28s}  {rr:>12.3f}")

    # Account simulation (starting $1,000 compounding)
    equity = COLLATERAL
    for t in sorted(all_trades, key=lambda x: x["timestamp"]):
        equity += t["pnl_usd"]
    final_col = _c(equity - COLLATERAL)
    print(f"\n  {'Start Account':<28s}  ${COLLATERAL:>10,.2f}")
    print(f"  {'Final Account (compounded)':<28s}  {final_col}${equity:>10,.2f}{R}")
    print(f"  {'Total Return':<28s}  {final_col}{(equity/COLLATERAL - 1)*100:>+10.1f}%{R}\n")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════

def export_artifacts(results: dict, start_year: int, end_year: int) -> dict:
    """
    Save trades, equity curve, and yearly summary to disk.

    Returns a dict with output paths.
    """
    out_results = PROJECT_ROOT / "backtests" / "results"
    out_reports = PROJECT_ROOT / "backtests" / "reports"
    out_results.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    # Flatten trades
    all_trades = []
    for y in sorted(results.keys()):
        for d in ("LONG", "SHORT"):
            all_trades.extend(results[y].get(d, []))

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)

    ts_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    trades_path = out_results / f"tradeplan_trades_{start_year}_{end_year}_{ts_tag}.csv"
    equity_path = out_results / f"tradeplan_equity_{start_year}_{end_year}_{ts_tag}.csv"
    summary_path = out_results / f"tradeplan_yearly_summary_{start_year}_{end_year}_{ts_tag}.json"
    md_path = out_reports / f"tradeplan_report_{start_year}_{end_year}_{ts_tag}.md"

    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)

        # Equity curve (starting $1,000; compounding by trade PnL)
        eq = COLLATERAL
        equity_rows = []
        for _, t in trades_df.iterrows():
            eq += float(t["pnl_usd"])
            equity_rows.append(
                {
                    "timestamp": t["exit_time"],
                    "equity_usd": eq,
                    "pnl_usd": float(t["pnl_usd"]),
                    "direction": t["direction"],
                    "exit_reason": t["exit_reason"],
                }
            )
        pd.DataFrame(equity_rows).to_csv(equity_path, index=False)

    # Yearly metrics summary
    yearly = {}
    for year in range(start_year, end_year + 1):
        long_trades = results.get(year, {}).get("LONG", [])
        short_trades = results.get(year, {}).get("SHORT", [])
        total_trades = long_trades + short_trades
        yearly[str(year)] = {
            "LONG": compute_metrics(long_trades),
            "SHORT": compute_metrics(short_trades),
            "TOTAL": compute_metrics(total_trades),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "start_year": start_year,
                "end_year": end_year,
                "trade_plan": {
                    "collateral_usd": COLLATERAL,
                    "leverage": LEVERAGE,
                    "notional_usd": NOTIONAL,
                    "sl_pct": SL_PCT,
                    "tp_activation_pct": TP_ACTIVATION_PCT,
                    "trail_dist_pct": TRAIL_DIST_PCT,
                    "taker_fee_pct": TAKER_FEE_PCT,
                    "slippage_pct": 0.0005,
                },
                "yearly": yearly,
            },
            f,
            indent=2,
        )

    # Markdown report (compact but complete)
    lines = []
    lines.append(f"## BTC Quant — Tradeplan Backtest ({start_year}–{end_year})\n")
    lines.append("### Trade plan\n")
    lines.append(f"- **Position**: ${COLLATERAL:,.0f} collateral × {LEVERAGE:.0f}x = ${NOTIONAL:,.0f} notional\n")
    lines.append(f"- **SL**: {SL_PCT*100:.3f}% from entry\n")
    lines.append(f"- **TP**: trailing stop after {TP_ACTIVATION_PCT*100:.2f}% move, trail distance {TRAIL_DIST_PCT*100:.3f}%\n")
    lines.append(f"- **Fees**: {TAKER_FEE_PCT*100:.2f}% taker/side (approx RT ${FEE_ROUND_TRIP:.2f})\n")
    lines.append("\n### Yearly summary (by exit year)\n")
    lines.append("| Year | Dir | Trades | Win% | PnL ($) | PF | AvgWin | AvgLoss | MaxDD | Expct | SL% | TR% | REV% | AvgFees | AvgHold |\n")
    lines.append("|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for year in range(start_year, end_year + 1):
        if year not in results:
            continue
        for direction, trades in [("LONG", results[year].get("LONG", [])), ("SHORT", results[year].get("SHORT", [])), ("TOTAL", results[year].get("LONG", []) + results[year].get("SHORT", []))]:
            m = compute_metrics(trades)
            lines.append(
                f"| {year} | {direction} | {m['n_trades']} | {m['win_rate']*100:.1f}% | {m['pnl_usd']:+,.2f} | "
                f"{(m['profit_factor'] if np.isfinite(m['profit_factor']) else float('nan')):.2f} | "
                f"{m['avg_win_usd']:+.2f} | {m['avg_loss_usd']:+.2f} | {m['max_drawdown_usd']:+.2f} | "
                f"{m['expectancy_usd']:+.2f} | {m['sl_pct']*100:.1f}% | {m['trail_pct']*100:.1f}% | {m['reversal_pct']*100:.1f}% | "
                f"{m['avg_fees_usd']:.2f} | {m['avg_bars_held']:.1f} |\n"
            )

    md_path.write_text("".join(lines), encoding="utf-8")

    return {
        "trades_csv": str(trades_path) if trades_df is not None else None,
        "equity_csv": str(equity_path),
        "yearly_json": str(summary_path),
        "report_md": str(md_path),
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BTC Yearly Backtest — Fixed Trade Plan")
    parser.add_argument("--start_year", type=int, default=2017)
    parser.add_argument("--end_year",   type=int, default=2026)
    parser.add_argument("--no_strong",  action="store_true",
                        help="Use LONG and SHORT signals only (exclude STRONG_LONG/STRONG_SHORT)")
    parser.add_argument("--strong_only", action="store_true",
                        help="Use STRONG_LONG and STRONG_SHORT signals only")
    args = parser.parse_args()

    if args.strong_only:
        valid_signals = {"STRONG_LONG", "STRONG_SHORT"}
    elif args.no_strong:
        valid_signals = {"LONG", "SHORT"}
    else:
        valid_signals = {"LONG", "STRONG_LONG", "SHORT", "STRONG_SHORT"}

    print(f"\n{SEP}")
    print(f"{C}{B}  BTC QUANT SYSTEM — YEARLY BACKTEST{R}")
    print(SEP)

    # ── 1. Load data ──────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"\n  {C}[1/3]{R} Loading 4H featured data …")

    try:
        # Try cached parquet first
        feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
        if feat_path.exists():
            df = pd.read_parquet(feat_path)
            print(f"       Loaded from cache: {feat_path.name}  "
                  f"({len(df):,} bars, {df.index[0].date()} → {df.index[-1].date()})")
        else:
            print("       Cache not found — building from raw data …")
            pipeline = DataPipeline()
            df_raw   = pipeline.get_data(timeframe="4h", use_cache=True)
            fe       = FeatureEngine()
            df       = fe.compute_all_features(df_raw)
    except Exception as e:
        print(f"  {RED}ERROR loading data: {e}{R}")
        return

    elapsed_load = time.perf_counter() - t0
    print(f"       Done in {elapsed_load:.1f}s")

    # ── 2. Generate signals ───────────────────────────────────
    print(f"\n  {C}[2/3]{R} Generating signals (batch, start_idx={WARMUP_BARS}) …")
    t1 = time.perf_counter()

    sig_engine = SignalEngine()
    sig_df = sig_engine.generate_signals_batch(df.copy(), start_idx=WARMUP_BARS)

    elapsed_sig = time.perf_counter() - t1
    total_signals = sig_df["signal_type"].value_counts()
    print(f"       Done in {elapsed_sig:.1f}s")
    print(f"       Signal distribution:")
    for sig_name, count in total_signals.items():
        pct = count / len(sig_df) * 100
        print(f"         {sig_name:<15s}: {count:>6,d}  ({pct:>5.1f}%)")

    print(f"       Active signals (using): {valid_signals}")

    # ── 3. Run backtest ───────────────────────────────────────
    print(f"\n  {C}[3/3]{R} Running backtest {args.start_year}–{args.end_year} …")
    t2 = time.perf_counter()

    results = run_backtest(
        df, sig_df,
        start_year=args.start_year,
        end_year=args.end_year,
        valid_signals=valid_signals,
    )

    elapsed_bt = time.perf_counter() - t2
    print(f"       Done in {elapsed_bt:.1f}s")

    # ── 4. Print report ───────────────────────────────────────
    print_yearly_report(results, args.start_year, args.end_year)

    # ── 5. Export artifacts ───────────────────────────────────
    paths = export_artifacts(results, args.start_year, args.end_year)
    print(f"\n{C}{B}  Saved outputs:{R}")
    for k, v in paths.items():
        print(f"    - {k}: {v}")


if __name__ == "__main__":
    main()
