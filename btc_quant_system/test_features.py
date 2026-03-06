#!/usr/bin/env python3
"""
Feature Engine — End-to-End Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Load 4H data from Parquet cache
2. Compute all 50+ features
3. Print comprehensive report
4. Verify no look-ahead bias
5. Save featured data to cache
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
from engines.data_pipeline import DataPipeline
from engines.feature_engine import FeatureEngine

# ── ANSI ─────────────────────────────────────────────────────
C  = "\033[36m"
G  = "\033[32m"
Y  = "\033[33m"
RED = "\033[31m"
B  = "\033[1m"
D  = "\033[2m"
R  = "\033[0m"


def lookahead_bias_test(df_original: pd.DataFrame, df_featured: pd.DataFrame, engine: FeatureEngine):
    """
    Verify no look-ahead bias by computing features on a truncated
    dataset and comparing the last row against the full computation.

    If row N's features only depend on data ≤ N, then truncating
    at row N and computing should yield identical values.
    """
    print(f"\n{C}{'═' * 65}")
    print(f"  LOOK-AHEAD BIAS TEST")
    print(f"{'═' * 65}{R}")

    # Pick a row in the middle (row 1000) — far enough for warmup
    test_idx = 1000
    truncated = df_original.iloc[: test_idx + 1].copy()

    engine_check = FeatureEngine()
    truncated_feat = engine_check.compute_all_features(truncated)

    all_feats = engine.get_all_feature_names()
    passed = 0
    failed = 0
    skipped = 0

    for feat in all_feats:
        if feat not in truncated_feat.columns or feat not in df_featured.columns:
            skipped += 1
            continue

        val_full = df_featured[feat].iloc[test_idx]
        val_trunc = truncated_feat[feat].iloc[test_idx]

        if pd.isna(val_full) and pd.isna(val_trunc):
            passed += 1
            continue

        if pd.isna(val_full) or pd.isna(val_trunc):
            print(f"  {RED}FAIL{R}  {feat}: full={val_full}, truncated={val_trunc}")
            failed += 1
            continue

        if np.isclose(val_full, val_trunc, rtol=1e-6, equal_nan=True):
            passed += 1
        else:
            print(f"  {RED}FAIL{R}  {feat}: full={val_full:.6f}, truncated={val_trunc:.6f}")
            failed += 1

    total = passed + failed + skipped
    print(f"\n  Results:  {G}{passed} passed{R}  |  {RED}{failed} failed{R}  |  {D}{skipped} skipped{R}  ({total} total)")

    if failed == 0:
        print(f"  {G}✓ ZERO look-ahead bias detected — all features are causal{R}")
    else:
        print(f"  {RED}✗ {failed} features may have look-ahead bias!{R}")

    print(f"{C}{'═' * 65}{R}")
    return failed == 0


def main():
    print(f"""
{C}{'═' * 65}
  FEATURE ENGINE — END-TO-END TEST
{'═' * 65}{R}
""")

    loggers = setup_all_loggers(
        log_dir=str(PROJECT_ROOT / "logs"),
        level="INFO",
    )

    total_t0 = time.perf_counter()

    # ══════════════════════════════════════════════════════════
    # STEP 1: Load 4H data from cache
    # ══════════════════════════════════════════════════════════
    print(f"{B}{C}[STEP 1/5]{R} Loading 4H data from cache...")
    print(f"{'─' * 65}")

    pipeline = DataPipeline()
    df = pipeline.load_processed("4h")

    if df is None:
        print(f"{Y}  Cache miss — running full pipeline...{R}")
        df = pipeline.get_data(timeframe="4h")

    print(f"  ✓ Loaded {format_number(len(df), 0)} candles")
    print(f"  ✓ Range: {df.index[0]} → {df.index[-1]}")
    print(f"  ✓ Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # ══════════════════════════════════════════════════════════
    # STEP 2: Compute all features
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 2/5]{R} Computing all features (5 layers)...")
    print(f"{'─' * 65}")

    engine = FeatureEngine()
    t0 = time.perf_counter()
    df_feat = engine.compute_all_features(df)
    compute_time = time.perf_counter() - t0

    print(f"\n  ✓ Computation done in {format_duration(compute_time)}")
    print(f"  ✓ Total columns: {len(df.columns)} OHLCV → {len(df_feat.columns)} with features")
    print(f"  ✓ Memory: {df_feat.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Print feature report
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 3/5]{R} Feature report...")
    print(f"{'─' * 65}")

    engine.print_feature_report(df_feat)

    # ══════════════════════════════════════════════════════════
    # STEP 4: Look-ahead bias test
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 4/5]{R} Look-ahead bias verification...")
    print(f"{'─' * 65}")

    bias_ok = lookahead_bias_test(df, df_feat, engine)

    # ══════════════════════════════════════════════════════════
    # STEP 5: Save featured data
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 5/5]{R} Saving featured data...")
    print(f"{'─' * 65}")

    out_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    df_feat.to_parquet(out_path, engine="pyarrow", compression="snappy")
    save_time = time.perf_counter() - t0

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {out_path.name}")
    print(f"  ✓ Size:  {size_mb:.2f} MB")
    print(f"  ✓ Time:  {format_duration(save_time)}")

    # ── Verify reload ─────────────────────────────────────
    t0 = time.perf_counter()
    df_reload = pd.read_parquet(out_path)
    reload_time = time.perf_counter() - t0

    match = (
        len(df_reload) == len(df_feat)
        and list(df_reload.columns) == list(df_feat.columns)
    )
    print(f"  ✓ Reload verified: {format_number(len(df_reload), 0)} rows, "
          f"{'MATCH' if match else 'MISMATCH!'} ({format_duration(reload_time)})")

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    total_time = time.perf_counter() - total_t0
    all_feats = engine.get_all_feature_names()
    total_features = len(all_feats)

    # Count NaN-free rows (valid for trading)
    warmup_rows = df_feat[all_feats].isna().any(axis=1).sum()
    valid_rows = len(df_feat) - warmup_rows
    valid_pct = valid_rows / len(df_feat) * 100

    print(f"""
{C}{'═' * 65}
  FINAL SUMMARY
{'═' * 65}{R}
  Input              : {format_number(len(df), 0)} candles (4H)
  Features computed  : {B}{total_features}{R}
  Output columns     : {len(df_feat.columns)} (5 OHLCV + {total_features} features)
  Warmup rows (NaN)  : {format_number(warmup_rows, 0)}
  Valid rows         : {format_number(valid_rows, 0)} ({valid_pct:.1f}%)
  Look-ahead bias    : {G + '✓ NONE' + R if bias_ok else RED + '✗ DETECTED' + R}
  Cache file         : {out_path.name} ({size_mb:.2f} MB)
  Total time         : {format_duration(total_time)}
{G}{'═' * 65}
  ✓ Feature engine test complete — ready for signal generation
{'═' * 65}{R}
""")


if __name__ == "__main__":
    main()
