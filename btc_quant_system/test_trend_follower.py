#!/usr/bin/env python3
"""
Trend-Follower: OLD vs NEW Comparison
======================================
Compares standard neutral mode vs trend-bias-adjusted mode
on bull market years (2019, 2020, 2023, 2024) and bear years (2018, 2022).

Shows:
  - LONG/SHORT ratio per year for each mode
  - Win rate, avg PnL (R), SL exit %
  - Bias distribution (STRONG_BULL / BULL / NEUTRAL / BEAR)
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
from engines.signal_engine import SignalEngine, SignalQualityEngine
from engines.trend_follower import (
    MarketBiasDetector,
    get_dynamic_thresholds,
    TrendAwareSignalEngine,
    run_trend_follower_comparison,
)

C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"

BULL_YEARS = [2019, 2020, 2023, 2024]
BEAR_YEARS = [2018, 2022]


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:5.1f}%" if total else "  N/A"


def print_strategy_report(label: str, results: list, color: str) -> None:
    total = len(results)
    if total == 0:
        print(f"  No trades.\n")
        return

    longs  = [r for r in results if r.get("direction") == "LONG"]
    shorts = [r for r in results if r.get("direction") == "SHORT"]
    wins   = [r for r in results if r["pnl_r"] > 0]
    sl_ex  = [r for r in results if r["exit"] == "STOP_LOSS"]
    trail  = [r for r in results if r["exit"] == "TRAIL_STOP"]
    tp3    = [r for r in results if r["exit"] == "TP3"]
    filt   = [r for r in results if r["exit"] == "BIAS_FILTERED"]
    valid  = [r for r in results if r["exit"] != "BIAS_FILTERED"]

    win_rate   = len(wins) / max(len(valid), 1) * 100
    bad_exit   = (len(sl_ex) + len(trail)) / max(len(valid), 1) * 100
    avg_pnl    = float(np.mean([r["pnl_r"] for r in valid])) if valid else 0.0

    print(f"\n{color}{'─' * 60}{R}")
    print(f"{B}{color}  {label}{R}")
    print(f"{color}{'─' * 60}{R}")
    print(f"  Total signals    : {total:>6,}")
    print(f"  Bias-filtered    : {len(filt):>6,} ({_pct(len(filt), total)})  ← wrong-way shorts blocked")
    print(f"  Active trades    : {len(valid):>6,}")
    print(f"  LONG / SHORT     : {len(longs)} / {len(shorts)}  "
          f"({_pct(len(longs), len(valid) or 1)} vs {_pct(len(shorts), len(valid) or 1)})")
    print(f"  Win rate         : {color}{win_rate:>5.1f}%{R}")
    print(f"  Avg PnL (R)      : {G if avg_pnl >= 0 else RED}{avg_pnl:>+.3f}R{R}")
    print(f"  SL + Trail exits : {RED if bad_exit > 60 else G}{bad_exit:>5.1f}%{R}")
    print(f"  TP3 exits        : {tp3.__len__():>4}")


def print_year_comparison(old_results: list, new_results: list, year: int) -> None:
    old_y = [r for r in old_results if r.get("year") == year]
    new_y = [r for r in new_results if r.get("year") == year and r["exit"] != "BIAS_FILTERED"]

    if not old_y:
        return

    old_long  = sum(1 for r in old_y if r.get("direction") == "LONG")
    old_short = sum(1 for r in old_y if r.get("direction") == "SHORT")
    new_long  = sum(1 for r in new_y if r.get("direction") == "LONG")
    new_short = sum(1 for r in new_y if r.get("direction") == "SHORT")
    filt_y    = [r for r in new_results if r.get("year") == year and r["exit"] == "BIAS_FILTERED"]

    old_wr  = sum(1 for r in old_y if r["pnl_r"] > 0) / max(len(old_y), 1) * 100
    new_wr  = sum(1 for r in new_y if r["pnl_r"] > 0) / max(len(new_y), 1) * 100
    old_avg = float(np.mean([r["pnl_r"] for r in old_y])) if old_y else 0.0
    new_avg = float(np.mean([r["pnl_r"] for r in new_y])) if new_y else 0.0
    old_sl  = sum(1 for r in old_y if r["exit"] == "STOP_LOSS") / max(len(old_y), 1) * 100
    new_sl  = sum(1 for r in new_y if r["exit"] == "STOP_LOSS") / max(len(new_y), 1) * 100

    buy_hold_pcts = {
        2017: 1319, 2018: -72, 2019: 94, 2020: 302,
        2021: 59, 2022: -65, 2023: 157, 2024: 121,
    }
    bh = buy_hold_pcts.get(year, "?")
    is_bull = year in BULL_YEARS

    col = G if is_bull else RED
    label = "BULL" if is_bull else "BEAR"
    print(f"\n{col}{'═' * 65}")
    print(f"  {year} — {label} market  (BTC B&H ≈ {bh}%)")
    print(f"{'═' * 65}{R}")
    print(f"  {'Metric':<32s} │ {'OLD':>10s} │ {'NEW':>10s} │ {'Delta':>8s}")
    print(f"  {'─' * 62}")

    def row(metric, old_v, new_v, delta=""):
        print(f"  {metric:<32s} │ {old_v:>10s} │ {new_v:>10s} │ {delta:>8s}")

    row("Trades (active)",
        str(len(old_y)), str(len(new_y)),
        f"{len(new_y) - len(old_y):+d}")
    row("  LONG / SHORT",
        f"{old_long}/{old_short}", f"{new_long}/{new_short}", "")
    row(f"  Bias-filtered shorts (new)", "", str(len(filt_y)), "")
    row("Win rate (%)",
        f"{old_wr:.1f}%", f"{new_wr:.1f}%",
        f"{new_wr - old_wr:>+.1f}pp")
    row("Avg PnL (R)",
        f"{old_avg:>+.3f}R", f"{new_avg:>+.3f}R",
        f"{new_avg - old_avg:>+.3f}R")
    row("SL exits (%)",
        f"{old_sl:.1f}%", f"{new_sl:.1f}%",
        f"{new_sl - old_sl:>+.1f}pp")


def print_bias_distribution(bias_series: pd.Series, year: int) -> None:
    year_b = bias_series[bias_series.index.year == year]
    if len(year_b) == 0:
        return
    dist = year_b.value_counts()
    total = len(year_b)
    print(f"  Bias distribution {year}:")
    order = ["STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"]
    for b in order:
        n = dist.get(b, 0)
        if n == 0:
            continue
        col = G if "BULL" in b else (RED if "BEAR" in b else D)
        print(f"    {col}{b:<12s}{R}: {n:>5,} bars ({n/total*100:5.1f}%)")


def print_adaptive_threshold_examples() -> None:
    print(f"\n{C}{'═' * 65}")
    print(f"  ADAPTIVE THRESHOLD EXAMPLES")
    print(f"{'═' * 65}{R}")
    print(f"  {'Bias':<14s} │ {'L.thresh':>8} │ {'S.thresh':>8} │ {'PosLong':>7} │ {'PosShort':>8} │ {'TP3×ATR':>7}")
    print(f"  {'─' * 62}")
    neutral = {"long": 4, "short": -4}
    print(f"  {D}{'NEUTRAL(old)':<14s}{R} │ {neutral['long']:>8} │ {neutral['short']:>8} │ {'×1.0':>7} │ {'×1.0':>8} │ {'5.0':>7}")
    for bias in ["STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"]:
        t = get_dynamic_thresholds(bias)
        col = G if "BULL" in bias else (RED if "BEAR" in bias else D)
        print(
            f"  {col}{bias:<14s}{R} │ {t['long_threshold']:>8} │ {t['short_threshold']:>8} │ "
            + f"{'×' + str(round(t['position_mult_long'], 1)):>7} │ "
            + f"{'×' + str(round(t['position_mult_short'], 1)):>8} │ "
            + f"{t['tp3_atr_mult']:>7.1f}"
        )


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print(f"""
{C}{'═' * 65}
  TREND-FOLLOWER: OLD vs NEW (2017-2024)
{'═' * 65}{R}
""")
    setup_all_loggers(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")

    # ── Load data ──────────────────────────────────────────────────
    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if not feat_path.exists():
        print(f"{RED}Feature cache not found: {feat_path}{R}")
        return

    print(f"{B}[1/4]{R} Loading data and generating signals...")
    t0 = time.perf_counter()
    df = pd.read_parquet(feat_path)

    signal_engine = SignalEngine()
    df = signal_engine.generate_signals_batch(df, start_idx=200)
    print(f"  ✓ {len(df):,} bars loaded, signals computed ({format_duration(time.perf_counter()-t0)})")

    # ── Filter to 2017-2024 ────────────────────────────────────────
    all_years = BULL_YEARS + BEAR_YEARS + [2017, 2021]
    df_range  = df.loc["2017-01-01":"2024-12-31"]
    print(f"  ✓ {len(df_range):,} bars in 2017-2024 range")

    # ── Run comparison ─────────────────────────────────────────────
    print(f"\n{B}[2/4]{R} Running OLD vs NEW simulation...")
    t1 = time.perf_counter()
    results = run_trend_follower_comparison(
        df_range, signal_engine, years=None, bias_recalc_bars=24
    )
    old_results  = results["old"]
    new_results  = results["new"]
    bias_series  = results["bias_series"]
    sim_time = time.perf_counter() - t1
    print(f"  ✓ {len(old_results):,} signals simulated in {format_duration(sim_time)}")

    # ── Adaptive threshold examples ────────────────────────────────
    print(f"\n{B}[3/4]{R} Adaptive threshold table...")
    print_adaptive_threshold_examples()

    # ── Overall reports ────────────────────────────────────────────
    print(f"\n{B}[4/4]{R} Strategy reports & year-by-year comparison...")
    print_strategy_report("OLD (neutral thresholds, fixed 1.5/5.0 ATR)", old_results, RED)
    print_strategy_report("NEW (trend-bias, adaptive thresholds + wider TP3)", new_results, G)

    # ── Per-year drill-down ────────────────────────────────────────
    years_to_show = sorted(set(r.get("year", 0) for r in old_results))
    for year in years_to_show:
        print_year_comparison(old_results, new_results, year)
        print_bias_distribution(bias_series, year)

    # ── Final summary ──────────────────────────────────────────────
    old_valid = [r for r in old_results]
    new_valid = [r for r in new_results if r["exit"] != "BIAS_FILTERED"]
    old_wr  = sum(1 for r in old_valid if r["pnl_r"] > 0) / max(len(old_valid), 1) * 100
    new_wr  = sum(1 for r in new_valid if r["pnl_r"] > 0) / max(len(new_valid), 1) * 100
    old_avg = float(np.mean([r["pnl_r"] for r in old_valid])) if old_valid else 0
    new_avg = float(np.mean([r["pnl_r"] for r in new_valid])) if new_valid else 0

    # Bull-year-only
    old_bull = [r for r in old_valid if r.get("year") in BULL_YEARS]
    new_bull = [r for r in new_valid if r.get("year") in BULL_YEARS]
    old_bull_wr  = sum(1 for r in old_bull if r["pnl_r"] > 0) / max(len(old_bull), 1) * 100
    new_bull_wr  = sum(1 for r in new_bull if r["pnl_r"] > 0) / max(len(new_bull), 1) * 100
    old_bull_avg = float(np.mean([r["pnl_r"] for r in old_bull])) if old_bull else 0
    new_bull_avg = float(np.mean([r["pnl_r"] for r in new_bull])) if new_bull else 0

    old_bull_long  = sum(1 for r in old_bull if r.get("direction") == "LONG")
    old_bull_short = sum(1 for r in old_bull if r.get("direction") == "SHORT")
    new_bull_long  = sum(1 for r in new_bull if r.get("direction") == "LONG")
    new_bull_short = sum(1 for r in new_bull if r.get("direction") == "SHORT")

    filt_total = sum(1 for r in new_results if r["exit"] == "BIAS_FILTERED"
                     and r.get("year") in BULL_YEARS)

    print(f"""
{C}{'═' * 65}
  FINAL SUMMARY
{'═' * 65}{R}

  {B}ALL YEARS (2017-2024){R}
  ┌────────────────────────────────────────────────────┐
  │ Metric             │    OLD     │    NEW     │
  ├────────────────────────────────────────────────────┤
  │ Win rate           │ {old_wr:>6.1f}%   │ {new_wr:>6.1f}%   │
  │ Avg PnL (R)        │ {old_avg:>+6.3f}R  │ {new_avg:>+6.3f}R  │
  └────────────────────────────────────────────────────┘

  {B}BULL YEARS ONLY (2019/2020/2023/2024){R}
  ┌────────────────────────────────────────────────────┐
  │ Metric             │    OLD     │    NEW     │
  ├────────────────────────────────────────────────────┤
  │ Win rate           │ {old_bull_wr:>6.1f}%   │ {new_bull_wr:>6.1f}%   │
  │ Avg PnL (R)        │ {old_bull_avg:>+6.3f}R  │ {new_bull_avg:>+6.3f}R  │
  │ LONG / SHORT ratio │ {old_bull_long}/{old_bull_short:<7}   │ {new_bull_long}/{new_bull_short:<7}   │
  │ Shorts filtered    │       —    │ {filt_total:>6,}     │
  └────────────────────────────────────────────────────┘

  Total time : {format_duration(time.perf_counter() - t0)}
{G}{'═' * 65}{R}
""")


if __name__ == "__main__":
    main()
