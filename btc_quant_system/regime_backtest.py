#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  BTC QUANT SYSTEM — Market Regime Detection & Backtest Comparison
  Train: 2015-2022  |  Test: 2023-2024
═══════════════════════════════════════════════════════════════════════════════

Pipeline
--------
  1. Load 4H OHLCV data + features (uses cache)
  2. Train 4-state GaussianHMM on 2015-2022
  3. Predict regimes for full period
  4. Print regime statistics + transition matrix
  5. Baseline backtest      (2023-2024, no regime filter)
  6. Regime-aware backtest  (2023-2024, HMM filter + sizing)
  7. Compare metrics side-by-side
  8. Save YAML report + PNG charts → backtests/reports/regime/

Usage
-----
  python3 regime_backtest.py               # uses feature cache if present
  python3 regime_backtest.py --no_charts   # skip matplotlib output
  python3 regime_backtest.py --retrain     # force HMM retrain even if model cached
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from engines.data_pipeline    import DataPipeline
from engines.feature_engine   import FeatureEngine
from engines.signal_engine    import SignalEngine
from engines.regime_detector  import (
    RegimeDetector, REGIME_CONFIG,
    BULL, BEAR, SIDEWAYS, HIGH_VOL, ALL_REGIMES,
)

# ── ANSI ─────────────────────────────────────────────────────────────────────
C   = "\033[36m"   # cyan
G   = "\033[32m"   # green
Y   = "\033[33m"   # yellow
RED = "\033[31m"   # red
B   = "\033[1m"    # bold
D   = "\033[2m"    # dim
R   = "\033[0m"    # reset

W   = 74
SEP = f"{C}{'═' * W}{R}"
SUB = f"  {'─' * (W - 2)}"

OUTPUT_DIR  = PROJECT_ROOT / "backtests" / "reports" / "regime"
MODEL_PATH  = PROJECT_ROOT / "models" / "regime_hmm.pkl"
FEAT_CACHE  = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
RAW_CACHE   = PROJECT_ROOT / "data" / "processed" / "btcusd_4h.parquet"

TRAIN_START = "2015-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2024-12-31"

# Backtest parameters
BT_INITIAL_CAPITAL = 100_000.0
BT_RISK_PCT        = 0.01     # 1 % of equity per 1R trade
BT_SLIPPAGE        = 0.0005   # 0.05 %
BT_FEE             = 0.0004   # 0.04 % taker fee (per leg)
BT_SL_MULT         = 1.5      # ATR multiplier for stop-loss
BT_TP_MULT         = 5.0      # ATR multiplier for take-profit 3
WARMUP_BARS        = 200      # bars to skip for indicator warm-up

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="  %(message)s",
)
logger = logging.getLogger("regime_bt")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Market Regime Detection & Backtest Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no_charts", action="store_true",
                   help="Skip matplotlib chart generation")
    p.add_argument("--retrain",   action="store_true",
                   help="Force HMM retrain (ignore cached model)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Load 4H featured OHLCV from cache or compute from scratch."""
    if FEAT_CACHE.exists():
        print(f"  {D}Loading cached features: {FEAT_CACHE.name}{R}")
        df = pd.read_parquet(FEAT_CACHE)
        return df

    print(f"  {D}Feature cache not found — computing from scratch…{R}")

    cfg_path = str(PROJECT_ROOT / "configs" / "trading_config.yaml")
    pipeline = DataPipeline(config_path=cfg_path)

    if RAW_CACHE.exists():
        df_raw = pd.read_parquet(RAW_CACHE)
    else:
        df_raw = pipeline.get_data("4h", start_date="2015-01-01",
                                   end_date="2024-12-31", use_cache=True)

    feat_engine = FeatureEngine()
    df = feat_engine.compute_all_features(df_raw)
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEAT_CACHE)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLIFIED BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

def _close_col(df: pd.DataFrame) -> pd.Series:
    for n in ("close", "Close", "CLOSE"):
        if n in df.columns:
            return df[n]
    raise KeyError("No close column found")


def _atr_col(df: pd.DataFrame, default_mult: float = 0.02) -> pd.Series:
    for n in ("atr", "atr_14", "ATR"):
        if n in df.columns:
            return df[n]
    # Fallback: 2 % of close
    return _close_col(df) * default_mult


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    regime_series: Optional[pd.Series] = None,
    use_regime: bool = False,
    label: str = "Backtest",
) -> Dict:
    """
    Event-driven simplified backtester.

    Parameters
    ----------
    df             : full OHLCV + features DataFrame
    signals        : output of SignalEngine.generate_signals_batch()
                     (must have columns: signal_type, signal_score, signal_confidence)
    regime_series  : pd.Series[str] with regime label per bar (None → ignore)
    use_regime     : if True, apply regime-based signal filtering + sizing
    label          : name for logging

    Returns
    -------
    dict with metrics, equity_curve (pd.Series), trades (list of dicts)
    """
    capital  = BT_INITIAL_CAPITAL
    peak     = capital
    equity   = [capital]
    eq_dates = [df.index[0]]

    trades: List[Dict] = []
    in_trade = False
    trade: Optional[Dict] = None

    close_s = _close_col(df)
    atr_s   = _atr_col(df)
    high_s  = df.get("high", df.get("High", close_s))
    low_s   = df.get("low",  df.get("Low",  close_s))

    for i in range(WARMUP_BARS, len(df)):
        bar      = df.index[i]
        close_i  = float(close_s.iloc[i])
        high_i   = float(high_s.iloc[i])
        low_i    = float(low_s.iloc[i])
        atr_i    = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else close_i * 0.02

        # ── Monitor open trade ────────────────────────────────────────────────
        if in_trade and trade is not None:
            exited = False
            if trade["direction"] == "LONG":
                if low_i <= trade["sl"]:
                    _close_trade(trade, bar, trade["sl"], "STOP_LOSS", capital, trades)
                    capital += trade["pnl"]
                    exited = True
                elif high_i >= trade["tp"]:
                    _close_trade(trade, bar, trade["tp"], "TAKE_PROFIT", capital, trades)
                    capital += trade["pnl"]
                    exited = True
            else:  # SHORT
                if high_i >= trade["sl"]:
                    _close_trade(trade, bar, trade["sl"], "STOP_LOSS", capital, trades)
                    capital += trade["pnl"]
                    exited = True
                elif low_i <= trade["tp"]:
                    _close_trade(trade, bar, trade["tp"], "TAKE_PROFIT", capital, trades)
                    capital += trade["pnl"]
                    exited = True

            if exited:
                in_trade = False
                trade = None
                peak = max(peak, capital)

        # ── Look for new entry (only when flat) ───────────────────────────────
        if not in_trade:
            sig_row = signals.iloc[i] if i < len(signals) else None
            if sig_row is None:
                equity.append(capital)
                eq_dates.append(bar)
                continue

            raw_sig    = str(sig_row.get("signal_type", "SKIP"))
            raw_score  = int(sig_row.get("signal_score", 0))
            raw_conf   = float(sig_row.get("signal_confidence", 0))
            size_mult  = 1.0
            regime     = "SIDEWAYS"

            if use_regime and regime_series is not None and bar in regime_series.index:
                regime = str(regime_series.loc[bar])
                adj_sig, adj_conf, size_mult = RegimeDetector.adjust_signal(
                    raw_score, raw_sig, raw_conf, regime
                )
            else:
                adj_sig  = raw_sig
                adj_conf = raw_conf

            # Only enter on actionable signals
            if adj_sig in ("LONG", "STRONG_LONG", "SHORT", "STRONG_SHORT") \
               and adj_conf >= 0.60 and size_mult > 0:

                direction = "LONG" if adj_sig in ("LONG", "STRONG_LONG") else "SHORT"

                # Entry price with slippage
                if direction == "LONG":
                    entry = close_i * (1 + BT_SLIPPAGE)
                    sl    = entry - atr_i * BT_SL_MULT
                    tp    = entry + atr_i * BT_TP_MULT
                else:
                    entry = close_i * (1 - BT_SLIPPAGE)
                    sl    = entry + atr_i * BT_SL_MULT
                    tp    = entry - atr_i * BT_TP_MULT

                sl_dist = abs(entry - sl)
                if sl_dist <= 0:
                    equity.append(capital)
                    eq_dates.append(bar)
                    continue

                # Risk-based position size
                risk_usd  = capital * BT_RISK_PCT * size_mult
                qty       = risk_usd / sl_dist
                notional  = qty * entry

                # Entry fee
                entry_fee = notional * BT_FEE

                trade = {
                    "id":        len(trades) + 1,
                    "entry_time": bar,
                    "direction": direction,
                    "entry":     entry,
                    "sl":        sl,
                    "tp":        tp,
                    "qty":       qty,
                    "risk_usd":  risk_usd,
                    "entry_fee": entry_fee,
                    "regime":    regime,
                    "score":     raw_score,
                    "conf":      adj_conf,
                    "size_mult": size_mult,
                }
                capital -= entry_fee
                in_trade = True

        equity.append(capital)
        eq_dates.append(bar)

    # Close any open trade at end
    if in_trade and trade is not None:
        last_close = float(close_s.iloc[-1])
        _close_trade(trade, df.index[-1], last_close, "END_OF_DATA", capital, trades)
        capital += trade["pnl"]
        equity[-1] = capital

    equity_series = pd.Series(equity, index=eq_dates)
    return _compute_metrics(trades, equity_series, BT_INITIAL_CAPITAL, label)


def _close_trade(trade: Dict, exit_time, exit_price: float, reason: str,
                 current_capital: float, trades: List[Dict]) -> None:
    """Fill trade dict with exit details and append to trades list."""
    direction = trade["direction"]
    qty       = trade["qty"]
    entry     = trade["entry"]
    risk_usd  = trade["risk_usd"]

    gross_pnl = qty * (exit_price - entry) if direction == "LONG" \
                else qty * (entry - exit_price)
    exit_fee  = abs(qty * exit_price) * BT_FEE
    net_pnl   = gross_pnl - exit_fee - trade["entry_fee"]
    pnl_r     = net_pnl / risk_usd if risk_usd > 0 else 0.0

    trade.update({
        "exit_time":  exit_time,
        "exit_price": exit_price,
        "exit_reason": reason,
        "gross_pnl":  gross_pnl,
        "net_pnl":    net_pnl,
        "pnl_r":      pnl_r,
        "pnl":        net_pnl,
    })
    trades.append(dict(trade))


def _compute_metrics(
    trades: List[Dict], equity: pd.Series,
    initial_capital: float, label: str
) -> Dict:
    """Compute standard backtest metrics from trades + equity curve."""
    if not trades:
        return {
            "label": label, "total_trades": 0,
            "error": "No trades executed",
            "equity_curve": equity,
            "trades": [],
        }

    pnls  = [t["pnl"] for t in trades]
    pnl_r = [t["pnl_r"] for t in trades]
    wins  = [t for t in trades if t["pnl"] > 0]
    loss  = [t for t in trades if t["pnl"] <= 0]

    win_rate = len(wins) / len(trades)
    avg_win  = float(np.mean([t["pnl"] for t in wins]))  if wins else 0.0
    avg_loss = float(np.mean([abs(t["pnl"]) for t in loss])) if loss else 0.0
    avg_wr   = float(np.mean([t["pnl_r"] for t in wins]))  if wins else 0.0
    avg_lr   = float(np.mean([abs(t["pnl_r"]) for t in loss])) if loss else 0.0
    expectancy = win_rate * avg_wr - (1 - win_rate) * avg_lr

    total_wins  = sum(t["pnl"] for t in wins)
    total_loss  = abs(sum(t["pnl"] for t in loss))
    pf = total_wins / total_loss if total_loss > 0 else float("inf")

    final_equity = float(equity.iloc[-1])
    total_ret    = (final_equity - initial_capital) / initial_capital
    days = (equity.index[-1] - equity.index[0]).days if hasattr(equity.index[-1], "day") else len(equity)
    years = max(days / 365.25, 0.01)
    cagr = (final_equity / initial_capital) ** (1 / years) - 1

    ret_s    = equity.pct_change().dropna()
    sharpe   = float(ret_s.mean() / ret_s.std() * np.sqrt(2190)) if ret_s.std() > 0 else 0.0
    down_std = ret_s[ret_s < 0].std()
    sortino  = float(ret_s.mean() / down_std * np.sqrt(2190)) if down_std > 0 else float("inf")

    roll_max = equity.expanding().max()
    dd       = (equity - roll_max) / roll_max
    max_dd   = float(abs(dd.min()))
    calmar   = float(cagr / max_dd) if max_dd > 0 else float("inf")

    return {
        "label":             label,
        "initial_capital":   initial_capital,
        "final_equity":      round(final_equity, 2),
        "total_pnl":         round(final_equity - initial_capital, 2),
        "total_return_pct":  round(total_ret * 100, 2),
        "cagr_pct":          round(cagr * 100, 2),
        "total_trades":      len(trades),
        "winning_trades":    len(wins),
        "losing_trades":     len(loss),
        "win_rate_pct":      round(win_rate * 100, 2),
        "avg_win_usd":       round(avg_win, 2),
        "avg_loss_usd":      round(avg_loss, 2),
        "avg_win_r":         round(avg_wr, 3),
        "avg_loss_r":        round(avg_lr, 3),
        "expectancy_r":      round(expectancy, 3),
        "profit_factor":     round(pf, 3),
        "sharpe_ratio":      round(sharpe, 3),
        "sortino_ratio":     round(sortino, 3),
        "calmar_ratio":      round(calmar, 3),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "equity_curve":      equity,
        "trades":            trades,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _col_g(v: float, invert: bool = False) -> str:
    pos = v > 0
    if invert:
        pos = not pos
    return G if pos else (RED if v < 0 else D)


REGIME_COLOR = {BULL: G, BEAR: RED, SIDEWAYS: Y, HIGH_VOL: C}


def print_regime_stats(stats: pd.DataFrame, trans: pd.DataFrame):
    print(f"""
{SEP}
  {B}REGIME STATISTICS — 2015-2022 (Training Period){R}
{SUB}
  {'Regime':<12} {'Bars':>7} {'Time%':>7} {'AnnRet%':>9} {'AnnVol%':>9} {'Sharpe':>8} {'AvgRun':>8}
  {'──────':<12} {'────':>7} {'─────':>7} {'───────':>9} {'───────':>9} {'──────':>8} {'──────':>8}""")

    for regime, row in stats.iterrows():
        col = REGIME_COLOR.get(regime, D)
        ret_c = G if row["mean_ann_ret_pct"] > 0 else RED
        print(
            f"  {col}{regime:<12}{R} "
            f"{int(row['n_bars']):>7,} "
            f"{row['pct_time']:>6.1f}% "
            f"{ret_c}{row['mean_ann_ret_pct']:>8.1f}%{R} "
            f"{row['ann_vol_pct']:>8.1f}% "
            f"{_col_g(row['sharpe'])}{row['sharpe']:>8.3f}{R} "
            f"{row['avg_run_length']:>7.1f}"
        )

    print(f"\n  {B}Regime Transition Matrix{R}")
    _hdr = "From / To"
    print(f"  {_hdr:<12}", end="")
    for c in ALL_REGIMES:
        print(f"{REGIME_COLOR[c]}{c[:4]:>9}{R}", end="")
    print()
    print(f"  {'─'*12}", end="")
    print(f"{'─'*9}" * len(ALL_REGIMES))
    for regime in ALL_REGIMES:
        col = REGIME_COLOR.get(regime, D)
        print(f"  {col}{regime:<12}{R}", end="")
        for to in ALL_REGIMES:
            p = trans.loc[regime, to] if regime in trans.index and to in trans.columns else 0.0
            print(f"{p:>9.3f}", end="")
        print()


def print_regime_config():
    print(f"""
{SEP}
  {B}REGIME-AWARE TRADING ADJUSTMENTS{R}
{SUB}
  {'Regime':<12} {'Score Bias':>11} {'Long Min':>10} {'Short Min':>10} {'Size Mult':>10}  Action""")
    for regime, cfg in REGIME_CONFIG.items():
        col  = REGIME_COLOR.get(regime, D)
        bias = cfg["score_bias"]
        bias_s = f"{'+' if bias >= 0 else ''}{bias}"
        skip_s = f"{RED}SKIP ALL{R}" if cfg["skip"] else cfg["description"]
        print(
            f"  {col}{regime:<12}{R} "
            f"{Y}{bias_s:>11}{R} "
            f"{cfg['long_min_score']:>10} "
            f"{cfg['short_min_score']:>10} "
            f"{cfg['size_mult']:>10.2f}  {skip_s}"
        )


def print_backtest_result(res: Dict, label_override: str = None):
    label = label_override or res.get("label", "")
    if res.get("total_trades", 0) == 0:
        print(f"\n  {RED}[{label}] No trades executed!{R}")
        return

    ret_c  = G if res["total_return_pct"] >= 0 else RED
    sh_c   = G if res["sharpe_ratio"] >= 1.0   else (Y if res["sharpe_ratio"] >= 0 else RED)
    dd_c   = Y if res["max_drawdown_pct"] < 20  else RED
    pf_c   = G if res["profit_factor"] >= 1.5   else (Y if res["profit_factor"] >= 1.0 else RED)

    print(f"""
  {B}Return         {ret_c}{res['total_return_pct']:+.2f}%{R}  (CAGR {ret_c}{res['cagr_pct']:+.2f}%{R})
  Final Equity   ${res['final_equity']:>13,.2f}
  Total Trades   {res['total_trades']:>6}  (W: {res['winning_trades']}  L: {res['losing_trades']})
  Win Rate       {res['win_rate_pct']:>6.1f}%
  Avg Win / Loss {res['avg_win_r']:>6.2f}R / {res['avg_loss_r']:.2f}R
  Expectancy     {_col_g(res['expectancy_r'])}{res['expectancy_r']:>+.3f}R / trade{R}
  Profit Factor  {pf_c}{res['profit_factor']:.3f}{R}
  Sharpe         {sh_c}{res['sharpe_ratio']:.3f}{R}
  Sortino        {_col_g(res['sortino_ratio'])}{res['sortino_ratio']:.3f}{R}
  Max Drawdown   {dd_c}{res['max_drawdown_pct']:.2f}%{R}
  Calmar         {_col_g(res['calmar_ratio'])}{res['calmar_ratio']:.3f}{R}""")


def print_comparison(base: Dict, regime: Dict):
    print(f"""
{SEP}
  {B}PERFORMANCE COMPARISON — 2023-2024 (Out-of-Sample){R}
{SUB}
  {'Metric':<24} {'Baseline':>14} {'Regime-Aware':>14} {'Delta':>12}
  {'──────':<24} {'────────':>14} {'────────────':>14} {'─────':>12}""")

    metrics = [
        ("Total Return %",    "total_return_pct",  True,  "{:+.2f}%"),
        ("CAGR %",            "cagr_pct",          True,  "{:+.2f}%"),
        ("Total Trades",      "total_trades",      None,  "{:d}"),
        ("Win Rate %",        "win_rate_pct",      True,  "{:.1f}%"),
        ("Expectancy R",      "expectancy_r",      True,  "{:+.3f}R"),
        ("Profit Factor",     "profit_factor",     True,  "{:.3f}"),
        ("Sharpe Ratio",      "sharpe_ratio",      True,  "{:.3f}"),
        ("Sortino Ratio",     "sortino_ratio",     True,  "{:.3f}"),
        ("Max Drawdown %",    "max_drawdown_pct",  False, "{:.2f}%"),
        ("Calmar Ratio",      "calmar_ratio",      True,  "{:.3f}"),
    ]

    for name, key, higher_is_better, fmt in metrics:
        bv = base.get(key, 0)
        rv = regime.get(key, 0)
        delta = rv - bv if not isinstance(bv, str) else 0

        if higher_is_better is None:
            dc = D
        elif higher_is_better:
            dc = G if delta > 0 else (RED if delta < 0 else D)
        else:
            dc = G if delta < 0 else (RED if delta > 0 else D)

        bv_s = fmt.format(bv) if isinstance(bv, (int, float)) else str(bv)
        rv_s = fmt.format(rv) if isinstance(rv, (int, float)) else str(rv)
        d_s  = f"{'+' if delta >= 0 else ''}{delta:.2f}" if higher_is_better is not None else ""

        print(
            f"  {name:<24} {bv_s:>14} {rv_s:>14} {dc}{d_s:>12}{R}"
        )

    # Regime-based trade breakdown
    if regime.get("trades"):
        print(f"\n  {B}Regime Trade Distribution (Regime-Aware){R}")
        from collections import Counter
        regime_counts = Counter(t.get("regime", "?") for t in regime["trades"])
        total = len(regime["trades"])
        wins_by_r = {}
        for t in regime["trades"]:
            rg = t.get("regime", "?")
            wins_by_r.setdefault(rg, {"wins": 0, "total": 0})
            wins_by_r[rg]["total"] += 1
            if t["pnl"] > 0:
                wins_by_r[rg]["wins"] += 1

        print(f"  {'Regime':<12} {'Trades':>8} {'%':>6} {'Win Rate':>10}")
        print(f"  {'──────':<12} {'──────':>8} {'─':>6} {'────────':>10}")
        for rg in ALL_REGIMES:
            n = wins_by_r.get(rg, {}).get("total", 0)
            w = wins_by_r.get(rg, {}).get("wins",  0)
            if n == 0:
                continue
            col = REGIME_COLOR.get(rg, D)
            print(
                f"  {col}{rg:<12}{R} {n:>8} {n/total*100:>5.1f}% {w/n*100:>9.1f}%"
            )


# ══════════════════════════════════════════════════════════════════════════════
# CHART GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_charts(
    df_test: pd.DataFrame,
    regime_test: pd.Series,
    base_res: Dict,
    regime_res: Dict,
    proba_df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as mticker
    except ImportError:
        print(f"  {Y}[WARN] matplotlib not installed — skipping charts.{R}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#e6edf3",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "text.color":        "#e6edf3",
        "grid.color":        "#21262d",
        "grid.linewidth":    0.5,
        "axes.grid":         True,
        "font.family":       "monospace",
        "font.size":         9,
    })

    REGIME_HEX = {
        BULL:     "#3fb950",
        BEAR:     "#f85149",
        SIDEWAYS: "#d29922",
        HIGH_VOL: "#58a6ff",
    }

    # ── Figure 1: Regime timeline + equity comparison ─────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [1, 2, 2]},
        sharex=True,
    )
    fig.suptitle(
        "Market Regime Detection & Backtest Comparison  (2023-2024)",
        fontsize=11, color="#e6edf3", y=0.99,
    )

    # Panel 1: Regime bar
    ax_r = axes[0]
    for i, (ts, regime) in enumerate(regime_test.items()):
        ax_r.axvspan(
            pd.Timestamp(ts) - pd.Timedelta(hours=2),
            pd.Timestamp(ts) + pd.Timedelta(hours=2),
            color=REGIME_HEX.get(regime, "#8b949e"),
            alpha=0.8, linewidth=0,
        )
    ax_r.set_ylabel("Regime", fontsize=8)
    ax_r.set_yticks([])
    ax_r.grid(False)
    legend_patches = [
        mpatches.Patch(color=REGIME_HEX[r], label=r) for r in ALL_REGIMES
    ]
    ax_r.legend(handles=legend_patches, loc="upper right",
                fontsize=7, framealpha=0.4, ncol=4)

    # Panel 2: BTC Price
    ax_p = axes[1]
    close_s = df_test.get("close", df_test.get("Close"))
    if close_s is not None:
        ax_p.plot(close_s.index, close_s.values, color="#58a6ff",
                  linewidth=1.2, label="BTC Price")
    ax_p.set_ylabel("BTC Price (USDT)")
    ax_p.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_p.legend(fontsize=8, framealpha=0.3)

    # Panel 3: Equity curves
    ax_e = axes[2]
    if "equity_curve" in base_res:
        eq_base = base_res["equity_curve"]
        eq_base_test = eq_base[eq_base.index >= TEST_START]
        ax_e.plot(eq_base_test.index, eq_base_test.values,
                  color="#d29922", linewidth=1.5, linestyle="--",
                  label=f"Baseline ({base_res.get('total_return_pct', 0):+.1f}%)")
    if "equity_curve" in regime_res:
        eq_reg = regime_res["equity_curve"]
        eq_reg_test = eq_reg[eq_reg.index >= TEST_START]
        ax_e.plot(eq_reg_test.index, eq_reg_test.values,
                  color="#3fb950", linewidth=1.8,
                  label=f"Regime-Aware ({regime_res.get('total_return_pct', 0):+.1f}%)")
    ax_e.axhline(BT_INITIAL_CAPITAL, color="#8b949e", linewidth=0.8,
                 linestyle=":", alpha=0.6, label="Initial capital")
    ax_e.set_ylabel("Portfolio Equity ($)")
    ax_e.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_e.legend(fontsize=8, framealpha=0.3)

    plt.tight_layout()
    p = output_dir / "01_regime_equity_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── Figure 2: Regime posterior probabilities ───────────────────────────────
    proba_test = proba_df[proba_df.index >= TEST_START].dropna()
    if len(proba_test) > 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.suptitle("HMM Posterior Probabilities by Regime  (2023-2024)",
                     fontsize=11, color="#e6edf3")

        bottom = np.zeros(len(proba_test))
        for regime in ALL_REGIMES:
            if regime not in proba_test.columns:
                continue
            vals = proba_test[regime].values
            ax.fill_between(
                proba_test.index, bottom, bottom + vals,
                color=REGIME_HEX[regime], alpha=0.8, label=regime,
            )
            bottom += vals

        ax.set_ylim(0, 1)
        ax.set_ylabel("Posterior Probability")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(loc="upper right", fontsize=8, framealpha=0.4, ncol=4)
        plt.tight_layout()
        p = output_dir / "02_regime_probabilities.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # ── Figure 3: Metric comparison bar chart ─────────────────────────────────
    metrics_to_plot = {
        "Return %":    ("total_return_pct",  True),
        "Win Rate %":  ("win_rate_pct",      True),
        "Sharpe":      ("sharpe_ratio",      True),
        "Max DD %":    ("max_drawdown_pct",  False),
        "Profit Factor": ("profit_factor",   True),
        "Calmar":      ("calmar_ratio",      True),
    }

    fig, axes_m = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle("Baseline vs Regime-Aware — Metric Comparison",
                 fontsize=11, color="#e6edf3")

    for ax, (mname, (mkey, hib)) in zip(axes_m.flat, metrics_to_plot.items()):
        bv = base_res.get(mkey, 0)
        rv = regime_res.get(mkey, 0)
        colors_bars = []
        for v, better in [(bv, False), (rv, True)]:
            if hib:
                col = "#3fb950" if v >= 0 else "#f85149"
            else:
                col = "#f85149" if v > abs(bv) and better else "#3fb950"
            colors_bars.append(col)

        bars = ax.bar(["Baseline", "Regime-Aware"], [bv, rv],
                      color=["#d29922", "#3fb950"], width=0.5)
        ax.axhline(0, color="#8b949e", linewidth=0.8)
        ax.set_title(mname, fontsize=9)
        for bar, val in zip(bars, [bv, rv]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bv + rv) * 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p = output_dir / "03_metric_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = _parse_args()

    print(f"""
{C}╔{'═' * W}╗
║{'MARKET REGIME DETECTION — BTC QUANT SYSTEM':^{W}}║
║{'Hidden Markov Model (4-state Gaussian HMM)':^{W}}║
╚{'═' * W}╝{R}""")

    t_total = time.perf_counter()

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  {B}STEP 1  Load 4H Featured Data{R}")
    print(SUB)
    t0 = time.perf_counter()
    df = load_data()
    # Normalise index to tz-naive
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    print(f"  {G}Loaded {len(df):,} bars  "
          f"({df.index[0].date()} → {df.index[-1].date()})  "
          f"| {time.perf_counter()-t0:.1f}s{R}")

    # ── Step 2: Train / load HMM ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  {B}STEP 2  Train HMM (2015-2022){R}")
    print(SUB)
    t0 = time.perf_counter()

    if MODEL_PATH.exists() and not args.retrain:
        print(f"  {D}Loading cached HMM model: {MODEL_PATH.name}{R}")
        detector = RegimeDetector.load(MODEL_PATH)
        print(f"  {G}Model loaded  (train {detector.train_start} → {detector.train_end}){R}")
    else:
        print(f"  {D}Training 4-state GaussianHMM on 2015-2022 data…{R}", flush=True)
        detector = RegimeDetector(n_states=4, n_iter=200, n_init=5)
        detector.fit(df, TRAIN_START, TRAIN_END)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        detector.save(MODEL_PATH)
        print(f"  {G}Model trained + saved → {MODEL_PATH}  ({time.perf_counter()-t0:.1f}s){R}")

    print(f"  State mapping: {detector.state_to_label}")

    # ── Step 3: Predict regimes (full period) ─────────────────────────────────
    print(f"\n{SEP}")
    print(f"  {B}STEP 3  Predict Regimes (Full 2015-2024){R}")
    print(SUB)
    t0 = time.perf_counter()

    regime_full = detector.predict(df)
    proba_full  = detector.predict_proba(df)
    print(f"  {G}Predicted {len(regime_full):,} regime labels  ({time.perf_counter()-t0:.2f}s){R}")

    # Distribution
    dist = regime_full.value_counts()
    for rg in ALL_REGIMES:
        n   = dist.get(rg, 0)
        col = REGIME_COLOR.get(rg, D)
        print(f"    {col}{rg:<12}{R} {n:>6,} bars  ({n/len(regime_full)*100:.1f}%)")

    # Regime stats on training period
    df_train = df.loc[TRAIN_START:TRAIN_END]
    reg_train = regime_full.loc[TRAIN_START:TRAIN_END]
    stats = detector.regime_stats(df_train, reg_train)
    trans = detector.transition_matrix(reg_train)
    print_regime_stats(stats, trans)
    print_regime_config()

    # ── Step 4: Generate signals for test period ──────────────────────────────
    print(f"\n{SEP}")
    print(f"  {B}STEP 4  Generate Signals (2023-2024){R}")
    print(SUB)
    t0 = time.perf_counter()

    df_test      = df.loc[TEST_START:TEST_END].copy()
    regime_test  = regime_full.loc[TEST_START:TEST_END]
    proba_test   = proba_full.loc[TEST_START:TEST_END]

    sig_engine = SignalEngine()
    # Pre-generate signals on full df for indicator warmup
    df_with_sigs = sig_engine.generate_signals_batch(df.copy(), start_idx=WARMUP_BARS)
    signals_test = df_with_sigs.loc[TEST_START:TEST_END]

    print(f"  {G}Signals generated for {len(df_test):,} test bars  "
          f"({time.perf_counter()-t0:.1f}s){R}")

    sig_dist = df_with_sigs.loc[TEST_START:TEST_END, "signal_type"].value_counts()
    for st, n in sig_dist.items():
        print(f"    {st:<16} {n:>5,}  ({n/len(df_test)*100:.1f}%)")

    # ── Step 5: Baseline backtest ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  {B}STEP 5  Baseline Backtest (No Regime Filter){R}")
    print(SUB)
    t0 = time.perf_counter()

    base_res = run_backtest(
        df_test, signals_test, use_regime=False, label="Baseline"
    )
    print(f"  {G}Done ({time.perf_counter()-t0:.1f}s)  — "
          f"{base_res['total_trades']} trades{R}")
    print_backtest_result(base_res)

    # ── Step 6: Regime-aware backtest ─────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  {B}STEP 6  Regime-Aware Backtest{R}")
    print(SUB)
    t0 = time.perf_counter()

    regime_res = run_backtest(
        df_test, signals_test,
        regime_series=regime_test, use_regime=True,
        label="Regime-Aware",
    )
    print(f"  {G}Done ({time.perf_counter()-t0:.1f}s)  — "
          f"{regime_res['total_trades']} trades{R}")
    print_backtest_result(regime_res)

    # ── Step 7: Comparison ────────────────────────────────────────────────────
    print_comparison(base_res, regime_res)

    # ── Step 8: Save report + charts ──────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  {B}STEP 7  Save Report + Charts{R}")
    print(SUB)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Serialise regime stats for YAML
    def _safe(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    report = {
        "hmm_config": {
            "n_states":    detector.n_states,
            "n_iter":      detector.n_iter,
            "n_init":      detector.n_init,
            "train_start": TRAIN_START,
            "train_end":   TRAIN_END,
            "state_labels": detector.state_to_label,
        },
        "regime_distribution_full": {
            rg: int(dist.get(rg, 0)) for rg in ALL_REGIMES
        },
        "regime_stats": {
            rg: {k: _safe(v) for k, v in row.items()}
            for rg, row in stats.iterrows()
        },
        "baseline": {
            k: _safe(v) for k, v in base_res.items()
            if k not in ("equity_curve", "trades")
        },
        "regime_aware": {
            k: _safe(v) for k, v in regime_res.items()
            if k not in ("equity_curve", "trades")
        },
    }

    report_path = OUTPUT_DIR / "regime_report.yaml"
    with open(report_path, "w") as fh:
        yaml.dump(report, fh, default_flow_style=False, sort_keys=False)
    print(f"  {G}Report saved → {report_path}{R}")

    if not args.no_charts:
        saved = generate_charts(
            df_test, regime_test, base_res, regime_res, proba_full, OUTPUT_DIR
        )
        for p in saved:
            print(f"  {G}Chart  → {p}{R}")
    else:
        print(f"  {D}Charts skipped (--no_charts).{R}")

    total = time.perf_counter() - t_total
    print(f"\n{SEP}")
    print(f"  {B}{G}Regime backtest complete — {total:.1f}s total{R}\n")


if __name__ == "__main__":
    main()
