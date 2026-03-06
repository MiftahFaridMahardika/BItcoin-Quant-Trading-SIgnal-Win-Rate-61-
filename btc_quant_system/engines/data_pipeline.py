"""
BTC Quant Trading System — Data Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core engine for loading, cleaning, validating, resampling,
and caching Bitcoin OHLCV data.

Designed for the inspected dataset:
  - File: btcusd_1-min_data.csv (373 MB, 7.45M rows)
  - Columns: Timestamp (unix epoch), Open, High, Low, Close, Volume
  - Range: 2012-01-01 → 2026-03-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import logging

from utils.helpers import (
    load_config,
    ensure_dir,
    format_number,
    format_currency,
    format_pct,
    format_duration,
    format_large_number,
    tf_to_pandas,
    tf_to_minutes,
)
from utils.validators import validate_ohlcv, ValidationResult


logger = logging.getLogger("data")


# ── Constants ────────────────────────────────────────────────
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

OHLCV_DTYPES = {
    "Timestamp": "float64",
    "Open":      "float64",
    "High":      "float64",
    "Low":       "float64",
    "Close":     "float64",
    "Volume":    "float64",
}

RESAMPLE_RULES = {
    "Open":   "first",
    "High":   "max",
    "Low":    "min",
    "Close":  "last",
    "Volume": "sum",
}


class DataPipeline:
    """
    Pipeline for loading, cleaning, and processing BTC OHLCV data.

    Workflow:
        1. load_raw_data()   → Load CSV with auto timestamp detection
        2. clean_data()      → Handle missing, duplicates, anomalies
        3. validate_ohlcv()  → Validate data integrity
        4. resample()        → Convert to desired timeframe
        5. save_processed()  → Cache to Parquet for fast re-loading
        6. get_data()        → High-level method combining all steps
    """

    def __init__(self, config_path: str = "configs/trading_config.yaml"):
        """
        Initialize the DataPipeline.

        Parameters
        ----------
        config_path : str
            Path to the trading config YAML file.
        """
        self.config = load_config(config_path)
        self.data_config = self.config.get("data", {})
        self.source_path = self.data_config.get("source_path", "")
        self.symbol = self.data_config.get("symbol", "BTC/USDT")
        self.base_tf = self.data_config.get("base_timeframe", "1min")

        # Cache storage (in-memory)
        self._cache: Dict[str, pd.DataFrame] = {}

        # Directories
        self.raw_dir = ensure_dir("data/raw")
        self.processed_dir = ensure_dir("data/processed")
        self.features_dir = ensure_dir("data/features")

        logger.info(
            f"DataPipeline initialized — source: {self.source_path}, "
            f"symbol: {self.symbol}"
        )

    # ══════════════════════════════════════════════════════════
    # LOADING
    # ══════════════════════════════════════════════════════════

    def load_raw_data(
        self,
        filepath: Optional[str] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load raw CSV data with automatic timestamp conversion.

        Parameters
        ----------
        filepath : str, optional
            Override path to CSV file. Uses config source_path if None.
        nrows : int, optional
            Only read first N rows (useful for testing).

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex (UTC) and OHLCV columns.
        """
        path = Path(filepath or self.source_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        file_size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Loading {path.name} ({file_size_mb:.1f} MB)...")

        t0 = time.perf_counter()

        # Read CSV with optimized dtypes
        df = pd.read_csv(
            path,
            dtype=OHLCV_DTYPES,
            nrows=nrows,
        )

        # Validate basic structure
        if "Timestamp" not in df.columns:
            raise ValueError(
                f"Expected 'Timestamp' column, got: {list(df.columns)}"
            )

        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Auto-detect and convert timestamp format
        df = self._convert_timestamp(df)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Loaded {len(df):,} rows in {format_duration(elapsed)} "
            f"| Range: {df.index.min()} → {df.index.max()}"
        )

        return df

    def _convert_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-detect timestamp format and convert to DatetimeIndex.

        Handles:
        - Unix epoch (seconds as float/int)
        - ISO format strings (YYYY-MM-DD HH:MM:SS)
        """
        sample = df["Timestamp"].iloc[0]

        if isinstance(sample, (int, float)) or (
            isinstance(sample, str) and sample.replace(".", "").isdigit()
        ):
            # Unix epoch format
            logger.info("Detected timestamp format: Unix epoch (seconds)")
            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"], unit="s", utc=True
            )
        else:
            # String datetime format
            logger.info("Detected timestamp format: ISO string")
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

        df.set_index("Timestamp", inplace=True)
        df.index.name = "Timestamp"

        return df

    # ══════════════════════════════════════════════════════════
    # CLEANING
    # ══════════════════════════════════════════════════════════

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and sanitize OHLCV data.

        Steps:
        1. Sort by timestamp (ensure chronological order)
        2. Remove duplicate timestamps (keep last)
        3. Fix OHLC anomalies (High < Low)
        4. Handle missing values (forward-fill prices, zero-fill volume)
        5. Remove rows with negative prices

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
        """
        logger.info("Cleaning data...")
        t0 = time.perf_counter()
        initial_rows = len(df)

        # ── 1. Sort chronologically ──────────────────────
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            logger.info("  Sorted index chronologically")

        # ── 2. Remove duplicate timestamps ───────────────
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            df = df[~df.index.duplicated(keep="last")]
            logger.warning(f"  Removed {dup_count:,} duplicate timestamps")

        # ── 3. Fix OHLC anomalies ────────────────────────
        # Fix High < Low by swapping
        hl_mask = df["High"] < df["Low"]
        hl_count = hl_mask.sum()
        if hl_count > 0:
            df.loc[hl_mask, ["High", "Low"]] = df.loc[
                hl_mask, ["Low", "High"]
            ].values
            logger.warning(f"  Fixed {hl_count:,} High/Low swaps")

        # Ensure Open/Close within High/Low range
        df["Open"] = df["Open"].clip(lower=df["Low"], upper=df["High"])
        df["Close"] = df["Close"].clip(lower=df["Low"], upper=df["High"])

        # ── 4. Handle missing values ─────────────────────
        null_count = df[REQUIRED_COLUMNS].isnull().sum().sum()
        if null_count > 0:
            price_cols = ["Open", "High", "Low", "Close"]
            df[price_cols] = df[price_cols].ffill()
            df["Volume"] = df["Volume"].fillna(0)
            # Drop any remaining NaN at the start (can't forward fill)
            df.dropna(subset=price_cols, inplace=True)
            logger.info(f"  Filled {null_count:,} null values (ffill + zero)")

        # ── 5. Remove negative prices ────────────────────
        neg_mask = (df[["Open", "High", "Low", "Close"]] < 0).any(axis=1)
        neg_count = neg_mask.sum()
        if neg_count > 0:
            df = df[~neg_mask]
            logger.warning(f"  Removed {neg_count:,} rows with negative prices")

        # ── Summary ──────────────────────────────────────
        removed = initial_rows - len(df)
        elapsed = time.perf_counter() - t0

        logger.info(
            f"  Cleaning complete in {format_duration(elapsed)} — "
            f"{len(df):,} rows retained "
            f"({removed:,} removed, {format_pct(removed/max(initial_rows,1))})"
        )

        return df

    # ══════════════════════════════════════════════════════════
    # VALIDATION
    # ══════════════════════════════════════════════════════════

    def validate(self, df: pd.DataFrame) -> Tuple[bool, ValidationResult]:
        """
        Validate OHLCV data integrity.

        Returns
        -------
        (is_valid, ValidationResult)
        """
        logger.info("Validating OHLCV data...")
        result = validate_ohlcv(df)

        if result.is_valid:
            logger.info(f"  Validation PASSED — score: {result.score:.1f}/100")
        else:
            logger.error(
                f"  Validation FAILED — score: {result.score:.1f}/100"
            )
            for issue in result.issues:
                logger.error(f"    {issue}")

        for warn in result.warnings:
            logger.warning(f"    {warn}")

        return result.is_valid, result

    # ══════════════════════════════════════════════════════════
    # RESAMPLING
    # ══════════════════════════════════════════════════════════

    def resample_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Resample OHLCV data from base timeframe to a larger one.

        Parameters
        ----------
        df : pd.DataFrame
            Input OHLCV data (must have DatetimeIndex).
        timeframe : str
            Target timeframe ('5min', '15min', '1h', '4h', '1d', '1w').

        Returns
        -------
        pd.DataFrame
            Resampled OHLCV DataFrame.
        """
        if timeframe == self.base_tf:
            logger.info(f"Timeframe is already {timeframe}, skipping resample")
            return df.copy()

        pd_rule = tf_to_pandas(timeframe)
        logger.info(f"Resampling {len(df):,} rows → {timeframe} (rule={pd_rule})")

        t0 = time.perf_counter()

        resampled = df.resample(pd_rule).agg(RESAMPLE_RULES)

        # Drop rows where all OHLC are NaN (no data in that period)
        resampled.dropna(subset=["Open", "High", "Low", "Close"], how="all", inplace=True)

        # Fill any remaining volume NaN with 0
        resampled["Volume"] = resampled["Volume"].fillna(0)

        elapsed = time.perf_counter() - t0
        ratio = len(df) / max(len(resampled), 1)

        logger.info(
            f"  Resampled to {len(resampled):,} candles "
            f"(compression: {ratio:.0f}:1) in {format_duration(elapsed)}"
        )

        return resampled

    # ══════════════════════════════════════════════════════════
    # CACHING (Parquet)
    # ══════════════════════════════════════════════════════════

    def save_processed(
        self,
        df: pd.DataFrame,
        timeframe: str,
        fmt: str = "parquet",
    ) -> Path:
        """
        Save processed data to disk cache.

        Parameters
        ----------
        df : pd.DataFrame
            Processed OHLCV data.
        timeframe : str
            Timeframe label for the filename.
        fmt : str
            Output format ('parquet' or 'csv').

        Returns
        -------
        Path
            Path to the saved file.
        """
        filename = f"btcusd_{timeframe}.{fmt}"
        filepath = self.processed_dir / filename

        t0 = time.perf_counter()

        if fmt == "parquet":
            df.to_parquet(filepath, engine="pyarrow", compression="snappy")
        elif fmt == "csv":
            df.to_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        size_mb = filepath.stat().st_size / (1024 * 1024)
        elapsed = time.perf_counter() - t0

        logger.info(
            f"Saved {filepath.name} ({size_mb:.2f} MB) "
            f"in {format_duration(elapsed)}"
        )

        return filepath

    def load_processed(
        self,
        timeframe: str,
        fmt: str = "parquet",
    ) -> Optional[pd.DataFrame]:
        """
        Load processed data from disk cache.

        Returns None if cache file doesn't exist.
        """
        filename = f"btcusd_{timeframe}.{fmt}"
        filepath = self.processed_dir / filename

        if not filepath.exists():
            logger.debug(f"Cache miss: {filepath}")
            return None

        t0 = time.perf_counter()

        if fmt == "parquet":
            df = pd.read_parquet(filepath, engine="pyarrow")
        elif fmt == "csv":
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Cache hit: loaded {filepath.name} "
            f"({len(df):,} rows) in {format_duration(elapsed)}"
        )

        return df

    # ══════════════════════════════════════════════════════════
    # HIGH-LEVEL API
    # ══════════════════════════════════════════════════════════

    def get_data(
        self,
        timeframe: str = "4h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Main entry point — get processed OHLCV data.

        Logic:
        1. Check in-memory cache
        2. Check disk cache (Parquet)
        3. Load raw → clean → resample → cache

        Parameters
        ----------
        timeframe : str
            Desired timeframe ('1h', '4h', '1d', etc.).
        start_date : str, optional
            Filter start date (e.g., '2020-01-01').
        end_date : str, optional
            Filter end date (e.g., '2024-12-31').
        use_cache : bool
            Whether to use disk/memory cache (default True).

        Returns
        -------
        pd.DataFrame
            Processed, resampled OHLCV data.
        """
        cache_key = timeframe

        # ── 1. In-memory cache ────────────────────────────
        if use_cache and cache_key in self._cache:
            logger.info(f"Memory cache hit for {timeframe}")
            df = self._cache[cache_key]
            return self._filter_dates(df, start_date, end_date)

        # ── 2. Disk cache ─────────────────────────────────
        if use_cache:
            df = self.load_processed(timeframe)
            if df is not None:
                self._cache[cache_key] = df
                return self._filter_dates(df, start_date, end_date)

        # ── 3. Full pipeline ──────────────────────────────
        logger.info(f"No cache found — running full pipeline for {timeframe}")

        df = self.load_raw_data()
        df = self.clean_data(df)
        self.validate(df)

        if timeframe != self.base_tf:
            df = self.resample_timeframe(df, timeframe)

        # Cache to disk and memory
        self.save_processed(df, timeframe)
        self._cache[cache_key] = df

        return self._filter_dates(df, start_date, end_date)

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]
        return df

    # ══════════════════════════════════════════════════════════
    # INFO & DIAGNOSTICS
    # ══════════════════════════════════════════════════════════

    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Generate a summary dictionary about the data.

        Returns
        -------
        dict
            Data summary including shape, date range, price stats, etc.
        """
        info = {
            "rows": len(df),
            "columns": list(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "date_start": str(df.index.min()),
            "date_end": str(df.index.max()),
            "price_min": float(df["Close"].min()),
            "price_max": float(df["Close"].max()),
            "price_mean": float(df["Close"].mean()),
            "price_latest": float(df["Close"].iloc[-1]),
            "volume_total": float(df["Volume"].sum()),
            "volume_mean": float(df["Volume"].mean()),
            "zero_volume_pct": round(
                (df["Volume"] == 0).sum() / len(df) * 100, 2
            ),
            "null_count": int(df.isnull().sum().sum()),
        }

        return info

    def print_summary(self, df: pd.DataFrame, label: str = "") -> None:
        """
        Print a formatted summary of the DataFrame.
        """
        info = self.get_data_info(df)
        title = f" {label} " if label else " DATA SUMMARY "

        print()
        print(f"{'═' * 60}")
        print(f"{title:═^60}")
        print(f"{'═' * 60}")
        print(f"  Symbol         : {self.symbol}")
        print(f"  Rows           : {format_number(info['rows'], 0)}")
        print(f"  Memory         : {info['memory_mb']:.2f} MB")
        print(f"  Date Range     : {info['date_start']}")
        print(f"                 → {info['date_end']}")
        print(f"{'─' * 60}")
        print(f"  Close Min      : {format_currency(info['price_min'])}")
        print(f"  Close Max      : {format_currency(info['price_max'])}")
        print(f"  Close Mean     : {format_currency(info['price_mean'])}")
        print(f"  Close Latest   : {format_currency(info['price_latest'])}")
        print(f"{'─' * 60}")
        print(f"  Volume Total   : {format_large_number(info['volume_total'])} BTC")
        print(f"  Volume Mean    : {format_number(info['volume_mean'], 4)} BTC")
        print(f"  Zero Volume    : {info['zero_volume_pct']:.2f}%")
        print(f"  Null Values    : {info['null_count']}")
        print(f"{'═' * 60}")
        print()
