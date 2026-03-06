#!/usr/bin/env python3
"""
Signal Quality Engine — Comparison Test
=========================================
Compares old (score >= 4) vs new (quality >= 70, score >= 10, MTF aligned)
signal filtering on 2023-2024 data.

Steps:
  1. Load featured 4H data
  2. Run OLD SignalEngine (batch)
  3. Run NEW SignalQualityEngine on top
  4. Print side-by-side comparison: old vs new signal counts
  5. Print period drill-downs: 2023, 2024
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from utils.logger import setup_all_loggers
from utils.helpers import format_number, format_duration
from engines.signal_engine import SignalEngine, SignalQualityEngine


C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"


def print_comparison_table(df_full: pd.DataFrame, df_period: pd.DataFrame, label: str) -> None:
    """Side-by-side old vs new signal counts for a given slice."""
    total = len(df_period)
    if total == 0:
        print(f"  No data for period {label}")
        return

    old_dist = df_period["signal_type"].value_counts()
    new_dist  = df_period["quality_decision"].value_counts()

    old_actionable = total - old_dist.get("SKIP", 0)
    take_count     = new_dist.get("TAKE", 0)
    wait_count     = new_dist.get("WAIT", 0)
    reduction      = (1 - take_count / old_actionable) * 100 if old_actionable > 0 else 0

    print(f"\n{C}{'═' * 65}")
    print(f"  PERIOD: {label}   ({total:,} bars)")
    print(f"{'═' * 65}{R}")
    print(f"\n  {'Metric':<30s} │ {'OLD':>8s} │ {'NEW':>8s}")
    print(f"  {'─' * 54}")

    rows = [
        ("STRONG_LONG signals",     old_dist.get("STRONG_LONG", 0),  ""),
        ("LONG signals",            old_dist.get("LONG", 0),          ""),
        ("SHORT signals",           old_dist.get("SHORT", 0),         ""),
        ("STRONG_SHORT signals",    old_dist.get("STRONG_SHORT", 0),  ""),
        ("── Total actionable ──",  old_actionable,                   take_count),
        ("  TAKE (quality ≥ 70)",   "",                               take_count),
        ("  WAIT (quality 50-69)",  "",                               wait_count),
    ]

    for metric, old_val, new_val in rows:
        old_s = f"{old_val:>8,}" if isinstance(old_val, int) else f"{'':>8s}"
        new_s = f"{new_val:>8,}" if isinstance(new_val, int) else f"{'':>8s}"
        print(f"  {metric:<30s} │ {old_s} │ {new_s}")

    print(f"  {'─' * 54}")
    print(f"  {'Signal reduction':<30s} │ {'':>8s} │ {RED}{reduction:>7.1f}%{R}")

    # Quality stats
    qs_take = df_period.loc[df_period["quality_decision"] == "TAKE", "quality_score"]
    if len(qs_take) > 0:
        print(f"\n  Quality (TAKE signals): "
              f"mean={qs_take.mean():.1f}  min={qs_take.min():.1f}  max={qs_take.max():.1f}")

    # Trend alignment for TAKE signals
    take_aligned = (
        (df_period["quality_decision"] == "TAKE") &
        (df_period["trend_alignment"] == "ALIGNED")
    ).sum()
    if take_count > 0:
        print(f"  MTF aligned in TAKE   : {take_aligned:,} / {take_count:,} "
              f"({take_aligned/take_count*100:.1f}%)")

    print()


def main():
    print(f"""
{C}{'═' * 65}
  SIGNAL QUALITY ENGINE — OLD vs NEW COMPARISON
{'═' * 65}{R}
""")

    setup_all_loggers(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")

    # ── Load data ──────────────────────────────────────────────────────────
    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if not feat_path.exists():
        print(f"{RED}Feature cache not found: {feat_path}{R}")
        print("Run test_features.py first.")
        return

    print(f"{B}[1/4]{R} Loading featured data...")
    t0 = time.perf_counter()
    df = pd.read_parquet(feat_path)
    print(f"  ✓ {len(df):,} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  ✓ {len(df.columns)} feature columns")

    # ── OLD system ─────────────────────────────────────────────────────────
    print(f"\n{B}[2/4]{R} Running OLD SignalEngine (score threshold = 4)...")
    engine = SignalEngine()
    warmup = 200
    df = engine.generate_signals_batch(df, start_idx=warmup)
    t1 = time.perf_counter()
    print(f"  ✓ Done in {format_duration(t1 - t0)}")

    old_dist = df["signal_type"].value_counts()
    old_total = len(df)
    old_actionable = old_total - old_dist.get("SKIP", 0)
    print(f"  Old actionable : {old_actionable:,} ({old_actionable/old_total*100:.1f}%)")

    # ── NEW quality engine ─────────────────────────────────────────────────
    print(f"\n{B}[3/4]{R} Running NEW SignalQualityEngine "
          f"(quality ≥ 70, score ≥ 10, MTF aligned)...")
    quality_engine = SignalQualityEngine()
    df = quality_engine.generate_quality_signals_batch(
        df,
        default_ml_confidence=0.70,   # conservative default without live ML model
    )
    t2 = time.perf_counter()
    print(f"  ✓ Done in {format_duration(t2 - t1)}")

    new_dist   = df["quality_decision"].value_counts()
    take_count = new_dist.get("TAKE", 0)
    wait_count = new_dist.get("WAIT", 0)
    print(f"  New TAKE       : {take_count:,} ({take_count/old_total*100:.1f}%)")
    print(f"  New WAIT       : {wait_count:,} ({wait_count/old_total*100:.1f}%)")

    # ── Reports ────────────────────────────────────────────────────────────
    print(f"\n{B}[4/4]{R} Generating comparison reports...")

    # Full dataset
    quality_engine.print_quality_report(df, period_label="ALL DATA")

    # 2023
    df_2023 = df.loc["2023-01-01":"2023-12-31"]
    if len(df_2023) > 0:
        print_comparison_table(df, df_2023, "2023")
        quality_engine.print_quality_report(df_2023, period_label="2023")

    # 2024
    df_2024 = df.loc["2024-01-01":"2024-12-31"]
    if len(df_2024) > 0:
        print_comparison_table(df, df_2024, "2024")
        quality_engine.print_quality_report(df_2024, period_label="2024")

    # ── Signal quality breakdown by type ────────────────────────────────────
    print(f"\n{C}{'═' * 65}")
    print(f"  QUALITY SCORE BREAKDOWN BY SIGNAL TYPE")
    print(f"{'═' * 65}{R}")

    df_23_24 = df.loc["2023-01-01":"2024-12-31"]
    for sig_type in ["STRONG_LONG", "LONG", "STRONG_SHORT", "SHORT"]:
        subset = df_23_24[df_23_24["signal_type"] == sig_type]
        if len(subset) == 0:
            continue
        take = (subset["quality_decision"] == "TAKE").sum()
        wait = (subset["quality_decision"] == "WAIT").sum()
        skip = (subset["quality_decision"] == "SKIP").sum()
        print(f"\n  {B}{sig_type}{R}  ({len(subset):,} signals 2023-2024)")
        print(f"    TAKE : {G}{take:>5,}{R}  ({take/len(subset)*100:5.1f}%)")
        print(f"    WAIT : {Y}{wait:>5,}{R}  ({wait/len(subset)*100:5.1f}%)")
        print(f"    SKIP : {D}{skip:>5,}{R}  ({skip/len(subset)*100:5.1f}%)")

    # ── Max achievable quality example ─────────────────────────────────────
    print(f"\n{C}{'═' * 65}")
    print(f"  MAX QUALITY SCORE EXAMPLES (top 5 TAKE signals in 2023-2024)")
    print(f"{'═' * 65}{R}")

    top5 = (
        df_23_24[df_23_24["quality_decision"] == "TAKE"]
        .nlargest(5, "quality_score")[
            ["Close", "signal_type", "signal_score", "quality_score",
             "trend_alignment", "vol_ratio", "signal_regime"]
        ]
    )
    if len(top5) > 0:
        print()
        print(f"  {'Date':<20s} {'Signal':<14s} {'Score':>6s} {'Quality':>8s} "
              f"{'Align':<10s} {'VolRatio':>9s}")
        print(f"  {'─' * 72}")
        for ts, row in top5.iterrows():
            print(
                f"  {str(ts)[:19]:<20s} "
                f"{row['signal_type']:<14s} "
                f"{int(row['signal_score']):>+6d} "
                f"{row['quality_score']:>8.1f} "
                f"{row['trend_alignment']:<10s} "
                f"{row.get('vol_ratio', 0):>9.2f}x"
            )

    # ── Summary ─────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t0
    reduction_pct = (1 - take_count / old_actionable) * 100 if old_actionable > 0 else 0

    print(f"""
{C}{'═' * 65}
  SUMMARY
{'═' * 65}{R}
  Dataset          : {len(df):,} bars total
  OLD system       : {G}{old_actionable:,}{R} actionable signals (score ≥ 4)
  NEW system TAKE  : {G}{take_count:,}{R} high-quality signals (quality ≥ 70)
  NEW system WAIT  : {Y}{wait_count:,}{R} borderline signals (quality 50-69)
  Signal reduction : {RED}{reduction_pct:.1f}%{R} fewer entries
  Filters applied  :
    ✓ Min score ≥ 10 (raised from 4)
    ✓ Multi-timeframe alignment (4H vs 1D/7D trend)
    ✓ Volume confirmation (vol_ratio > 1.2/1.5/2.0)
    ✓ ATR regime blocker (EXTREME blocked)
    ✓ Quality gate ≥ 70/100
  Default ML conf  : 0.70 (set ml_confidence_col= to use live model)
  Total time       : {format_duration(total_time)}
{G}{'═' * 65}{R}
""")


if __name__ == "__main__":
    main()
