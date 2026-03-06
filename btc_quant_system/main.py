#!/usr/bin/env python3
"""
BTC Quant Trading System — Main Entry Point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage:
    python main.py                    # Full pipeline test
    python main.py --timeframe 1h     # Specific timeframe
    python main.py --no-cache         # Force reload from CSV
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_all_loggers
from utils.helpers import (
    load_config,
    format_number,
    format_currency,
    format_pct,
    format_duration,
    format_large_number,
)
from engines.data_pipeline import DataPipeline


# ── ANSI Helpers ─────────────────────────────────────────────
G  = "\033[32m"   # Green
Y  = "\033[33m"   # Yellow
C  = "\033[36m"   # Cyan
B  = "\033[1m"    # Bold
D  = "\033[2m"    # Dim
R  = "\033[0m"    # Reset


def print_banner():
    """Print system banner."""
    print(f"""
{C}{'═' * 62}
 ██████╗ ████████╗ ██████╗     ██████╗ ██╗   ██╗ █████╗ ███╗  ██╗████████╗
 ██╔══██╗╚══██╔══╝██╔════╝    ██╔═══██╗██║   ██║██╔══██╗████╗ ██║╚══██╔══╝
 ██████╔╝   ██║   ██║         ██║   ██║██║   ██║███████║██╔██╗██║   ██║
 ██╔══██╗   ██║   ██║         ██║▄▄ ██║██║   ██║██╔══██║██║╚████║   ██║
 ██████╔╝   ██║   ╚██████╗    ╚██████╔╝╚██████╔╝██║  ██║██║ ╚███║   ██║
 ╚═════╝    ╚═╝    ╚═════╝     ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚══╝   ╚═╝
{'═' * 62}{R}
{B}  BTC Quant Trading System v1.0.0{R}
{D}  Data Pipeline • Feature Engine • ML Models • Risk Management{R}
{C}{'═' * 62}{R}
""")


def run_pipeline_test(timeframe: str = "4h", use_cache: bool = True):
    """
    Run a complete pipeline test:
    1. Load raw 1-min data
    2. Clean & validate
    3. Resample to target timeframe
    4. Print comprehensive summary
    5. Save to cache & verify
    """
    total_t0 = time.perf_counter()

    # ── Initialize ────────────────────────────────────────
    pipeline = DataPipeline()

    # ══════════════════════════════════════════════════════
    # STEP 1: Load Raw Data
    # ══════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 1/6]{R} Loading raw 1-minute data...")
    print(f"{'─' * 62}")

    t0 = time.perf_counter()
    df_raw = pipeline.load_raw_data()
    load_time = time.perf_counter() - t0

    print(f"  ✓ Loaded {format_number(len(df_raw), 0)} candles")
    print(f"  ✓ Time: {format_duration(load_time)}")
    print(f"  ✓ Memory: {df_raw.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # ══════════════════════════════════════════════════════
    # STEP 2: Clean Data
    # ══════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 2/6]{R} Cleaning data...")
    print(f"{'─' * 62}")

    t0 = time.perf_counter()
    df_clean = pipeline.clean_data(df_raw)
    clean_time = time.perf_counter() - t0

    removed = len(df_raw) - len(df_clean)
    print(f"  ✓ Cleaned: {format_number(len(df_clean), 0)} rows retained")
    print(f"  ✓ Removed: {format_number(removed, 0)} rows")
    print(f"  ✓ Time: {format_duration(clean_time)}")

    # ══════════════════════════════════════════════════════
    # STEP 3: Validate
    # ══════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 3/6]{R} Validating OHLCV integrity...")
    print(f"{'─' * 62}")

    is_valid, result = pipeline.validate(df_clean)

    print(f"  {'✓' if is_valid else '✗'} Status: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  ✓ Score: {result.score:.1f}/100")
    print(f"  ✓ Checks: {result.passed_checks}/{result.total_checks} passed")

    if result.warnings:
        for w in result.warnings:
            print(f"  ⚠ {w}")

    # ══════════════════════════════════════════════════════
    # STEP 4: Resample
    # ══════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 4/6]{R} Resampling to {timeframe}...")
    print(f"{'─' * 62}")

    t0 = time.perf_counter()
    df_resampled = pipeline.resample_timeframe(df_clean, timeframe)
    resample_time = time.perf_counter() - t0

    compression = len(df_clean) / max(len(df_resampled), 1)
    print(f"  ✓ 1min candles : {format_number(len(df_clean), 0)}")
    print(f"  ✓ {timeframe} candles  : {format_number(len(df_resampled), 0)}")
    print(f"  ✓ Compression  : {compression:.0f}:1")
    print(f"  ✓ Time: {format_duration(resample_time)}")

    # ══════════════════════════════════════════════════════
    # STEP 5: Save to Cache
    # ══════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 5/6]{R} Saving processed data to cache...")
    print(f"{'─' * 62}")

    t0 = time.perf_counter()
    cache_path = pipeline.save_processed(df_resampled, timeframe)
    save_time = time.perf_counter() - t0

    cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {cache_path.name}")
    print(f"  ✓ Size: {cache_size_mb:.2f} MB")
    print(f"  ✓ Time: {format_duration(save_time)}")

    # ══════════════════════════════════════════════════════
    # STEP 6: Verify Cache
    # ══════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 6/6]{R} Verifying cache load...")
    print(f"{'─' * 62}")

    t0 = time.perf_counter()
    df_cached = pipeline.load_processed(timeframe)
    cache_load_time = time.perf_counter() - t0

    if df_cached is not None:
        match = (
            len(df_cached) == len(df_resampled)
            and df_cached.index[0] == df_resampled.index[0]
            and df_cached.index[-1] == df_resampled.index[-1]
        )
        print(f"  ✓ Cache loaded: {format_number(len(df_cached), 0)} rows")
        print(f"  ✓ Match: {'IDENTICAL' if match else 'MISMATCH!'}")
        print(f"  ✓ Load time: {format_duration(cache_load_time)}")
    else:
        print(f"  ✗ Cache load failed!")

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    pipeline.print_summary(df_resampled, label=f" {timeframe.upper()} RESAMPLED ")

    total_time = time.perf_counter() - total_t0

    # Price statistics
    close = df_resampled["Close"]

    print(f"{C}{'═' * 62}")
    print(f" PRICE STATISTICS ({timeframe.upper()})")
    print(f"{'═' * 62}{R}")
    print(f"  Latest Close   : {format_currency(close.iloc[-1])}")
    print(f"  All-time High  : {format_currency(close.max())}")
    print(f"  All-time Low   : {format_currency(close.min())}")
    print(f"  Mean           : {format_currency(close.mean())}")
    print(f"  Median         : {format_currency(close.median())}")
    print(f"  Std Dev        : {format_currency(close.std())}")

    # Performance timings
    print(f"\n{C}{'═' * 62}")
    print(f" PIPELINE PERFORMANCE")
    print(f"{'═' * 62}{R}")
    print(f"  CSV Load       : {format_duration(load_time)}")
    print(f"  Clean          : {format_duration(clean_time)}")
    print(f"  Resample       : {format_duration(resample_time)}")
    print(f"  Save Cache     : {format_duration(save_time)}")
    print(f"  Load Cache     : {format_duration(cache_load_time)}")
    print(f"  {'─' * 40}")
    print(f"  {B}TOTAL          : {format_duration(total_time)}{R}")

    print(f"\n{G}{'═' * 62}")
    print(f" ✓ Pipeline test complete — all systems operational")
    print(f"{'═' * 62}{R}\n")

    return df_resampled


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="BTC Quant Trading System"
    )
    parser.add_argument(
        "--timeframe", "-tf",
        default="4h",
        choices=["1min", "5min", "15min", "30min", "1h", "4h", "1d", "1w"],
        help="Target timeframe (default: 4h)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force reload from CSV (ignore cache)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup
    print_banner()
    loggers = setup_all_loggers(
        log_dir=str(PROJECT_ROOT / "logs"),
        level=args.log_level,
    )

    # Run
    run_pipeline_test(
        timeframe=args.timeframe,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
