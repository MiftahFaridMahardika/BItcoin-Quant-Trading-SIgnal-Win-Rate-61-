#!/usr/bin/env python3
"""
Signal Engine — End-to-End Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Load featured 4H data from cache
2. Generate batch signals
3. Print distribution, statistics, histograms
4. Show sample signals with full breakdown
5. Verify scoring logic with edge cases
6. Drill into 2023 period
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from utils.logger import setup_all_loggers
from utils.helpers import format_number, format_currency, format_duration
from engines.signal_engine import SignalEngine, SignalResult


# ── ANSI ─────────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"


def test_edge_cases(engine: SignalEngine):
    """
    Verify scoring classification with synthetic edge-case scores.
    """
    print(f"\n{C}{'═' * 65}")
    print(f"  EDGE CASE VERIFICATION")
    print(f"{'═' * 65}{R}")

    test_cases = [
        (+19, "STRONG_LONG"),
        (+10, "STRONG_LONG"),
        ( +8, "STRONG_LONG"),
        ( +7, "LONG"),
        ( +4, "LONG"),
        ( +3, "SKIP"),
        (  0, "SKIP"),
        ( -3, "SKIP"),
        ( -4, "SHORT"),
        ( -7, "SHORT"),
        ( -8, "STRONG_SHORT"),
        (-10, "STRONG_SHORT"),
        (-19, "STRONG_SHORT"),
    ]

    passed = 0
    failed = 0

    print(f"\n  {'Score':>6s}  │ {'Expected':<14s} │ {'Got':<14s} │ {'Conf':>6s} │ Status")
    print(f"  {'─' * 58}")

    for score, expected in test_cases:
        sig_type, conf = engine._classify_signal(score)
        ok = sig_type == expected
        status = f"{G}PASS{R}" if ok else f"{RED}FAIL{R}"
        print(
            f"  {score:+5d}  │ {expected:<14s} │ {sig_type:<14s} │ "
            f"{conf:5.2f}  │ {status}"
        )
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n  Results: {G}{passed} passed{R}, {RED}{failed} failed{R}")

    if failed == 0:
        print(f"  {G}✓ All classification thresholds verified{R}")
    else:
        print(f"  {RED}✗ Classification logic has errors!{R}")

    print(f"{C}{'═' * 65}{R}")
    return failed == 0


def test_weight_integrity(engine: SignalEngine):
    """Verify weights sum to MAX_SCORE and all layers covered."""
    print(f"\n{C}{'═' * 65}")
    print(f"  WEIGHT INTEGRITY CHECK")
    print(f"{'═' * 65}{R}")

    total = sum(engine.WEIGHTS.values())
    print(f"\n{B}  Weight Allocation:{R}")
    print(f"  {'─' * 50}")

    for layer, keys in engine.LAYER_MAP.items():
        layer_sum = sum(engine.WEIGHTS[k] for k in keys)
        components = ", ".join(f"{k}={engine.WEIGHTS[k]}" for k in keys)
        print(f"  {layer:<18s} │ {layer_sum:2d} │ {components}")

    print(f"  {'─' * 50}")
    print(f"  {B}TOTAL              │ {total:2d} │ MAX_SCORE={engine.MAX_SCORE}{R}")

    ok = total == engine.MAX_SCORE
    if ok:
        print(f"\n  {G}✓ Weights sum ({total}) == MAX_SCORE ({engine.MAX_SCORE}){R}")
    else:
        print(f"\n  {RED}✗ Weight mismatch! sum={total} ≠ MAX_SCORE={engine.MAX_SCORE}{R}")

    print(f"{C}{'═' * 65}{R}")
    return ok


def main():
    print(f"""
{C}{'═' * 65}
  SIGNAL ENGINE — END-TO-END TEST
{'═' * 65}{R}
""")

    loggers = setup_all_loggers(
        log_dir=str(PROJECT_ROOT / "logs"),
        level="INFO",
    )

    total_t0 = time.perf_counter()
    engine = SignalEngine()

    # ══════════════════════════════════════════════════════════
    # STEP 1: Load featured data
    # ══════════════════════════════════════════════════════════
    print(f"{B}{C}[STEP 1/6]{R} Loading featured 4H data...")
    print(f"{'─' * 65}")

    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if not feat_path.exists():
        print(f"{RED}Feature cache not found at {feat_path}{R}")
        print("Run test_features.py first.")
        return

    df = pd.read_parquet(feat_path)
    print(f"  ✓ Loaded {format_number(len(df), 0)} candles with {len(df.columns)} columns")
    print(f"  ✓ Range: {df.index[0]} → {df.index[-1]}")

    # ══════════════════════════════════════════════════════════
    # STEP 2: Weight & classification verification
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 2/6]{R} Verifying scoring logic...")
    print(f"{'─' * 65}")

    weight_ok = test_weight_integrity(engine)
    class_ok = test_edge_cases(engine)

    # ══════════════════════════════════════════════════════════
    # STEP 3: Generate batch signals (full dataset)
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 3/6]{R} Generating batch signals (full dataset)...")
    print(f"{'─' * 65}")

    # Determine warmup: skip rows with NaN in key features
    warmup = 200  # conservative warmup
    t0 = time.perf_counter()
    df = engine.generate_signals_batch(df, start_idx=warmup)
    batch_time = time.perf_counter() - t0

    print(f"  ✓ Batch signals generated in {format_duration(batch_time)}")
    print(f"  ✓ Warmup skipped: first {warmup} bars")

    # ══════════════════════════════════════════════════════════
    # STEP 4: Full dataset report
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 4/6]{R} Full dataset signal report...")
    print(f"{'─' * 65}")

    engine.print_signal_report(df)

    # ══════════════════════════════════════════════════════════
    # STEP 5: Drill into 2023
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 5/6]{R} Period analysis: 2023...")
    print(f"{'─' * 65}")

    df_2023 = df["2023-01-01":"2023-12-31"]
    print(f"  2023 range: {df_2023.index[0]} → {df_2023.index[-1]}")
    print(f"  Total bars: {len(df_2023):,}")

    engine.print_signal_report(df_2023)

    # ══════════════════════════════════════════════════════════
    # STEP 6: Sample signals with full breakdown
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 6/6]{R} Sample signals with full breakdown...")
    print(f"{'─' * 65}")

    # Show samples from each type
    for sig_type in ["STRONG_LONG", "LONG", "SHORT", "STRONG_SHORT"]:
        subset = df[df["signal_type"] == sig_type]
        if len(subset) > 0:
            engine.print_sample_signals(df, n=2, signal_type=sig_type)

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    total_time = time.perf_counter() - total_t0

    dist = df["signal_type"].value_counts()
    actionable = len(df) - dist.get("SKIP", 0)

    print(f"""
{C}{'═' * 65}
  FINAL SUMMARY
{'═' * 65}{R}
  Total bars         : {format_number(len(df), 0)}
  Warmup skipped     : {warmup}
  Actionable signals : {format_number(actionable, 0)} ({actionable/len(df)*100:.1f}%)
  STRONG_LONG        : {dist.get('STRONG_LONG', 0):,}
  LONG               : {dist.get('LONG', 0):,}
  SKIP               : {dist.get('SKIP', 0):,}
  SHORT              : {dist.get('SHORT', 0):,}
  STRONG_SHORT       : {dist.get('STRONG_SHORT', 0):,}
  Weight check       : {G + '✓ PASS' + R if weight_ok else RED + '✗ FAIL' + R}
  Classification     : {G + '✓ PASS' + R if class_ok else RED + '✗ FAIL' + R}
  Batch time         : {format_duration(batch_time)}
  Total time         : {format_duration(total_time)}
{G}{'═' * 65}
  ✓ Signal engine test complete — ready for risk management
{'═' * 65}{R}
""")


if __name__ == "__main__":
    main()
