"""
BTC Quant Trading System — Data Validators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validation functions for OHLCV data integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


# ── Validation Result ────────────────────────────────────────
@dataclass
class ValidationResult:
    """Container for data validation results."""

    is_valid: bool = True
    total_checks: int = 0
    passed_checks: int = 0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Quality score from 0 to 100."""
        if self.total_checks == 0:
            return 100.0
        return (self.passed_checks / self.total_checks) * 100

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_check(self, passed: bool, name: str, detail: str = "") -> None:
        self.total_checks += 1
        if passed:
            self.passed_checks += 1
        else:
            self.add_issue(f"[FAIL] {name}: {detail}")

    def summary(self) -> str:
        lines = [
            f"{'PASSED' if self.is_valid else 'FAILED'} "
            f"— Score: {self.score:.1f}/100 "
            f"({self.passed_checks}/{self.total_checks} checks passed)",
        ]
        for issue in self.issues:
            lines.append(f"  ERROR   {issue}")
        for warn in self.warnings:
            lines.append(f"  WARN    {warn}")
        return "\n".join(lines)


# ── Validators ───────────────────────────────────────────────
def validate_columns(
    df: pd.DataFrame,
    required: List[str] = None,
) -> ValidationResult:
    """
    Validate that required columns exist in the DataFrame.
    """
    if required is None:
        required = ["Open", "High", "Low", "Close", "Volume"]

    result = ValidationResult()
    missing = [col for col in required if col not in df.columns]

    result.add_check(
        passed=len(missing) == 0,
        name="Required Columns",
        detail=f"Missing: {missing}" if missing else "",
    )

    return result


def validate_ohlcv(df: pd.DataFrame) -> ValidationResult:
    """
    Comprehensive OHLCV data validation.

    Checks:
    1. Required columns exist
    2. No null/NaN values
    3. No negative prices
    4. High >= Low for all rows
    5. Open/Close within High/Low range
    6. No negative volume
    7. Monotonically increasing index (if datetime)
    8. No duplicate timestamps
    """
    result = ValidationResult()

    # ── 1. Column existence ───────────────────────────────
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    result.add_check(
        passed=len(missing) == 0,
        name="Column Existence",
        detail=f"Missing: {missing}" if missing else "",
    )
    if missing:
        return result  # Can't proceed without columns

    # ── 2. Null values ────────────────────────────────────
    null_counts = df[required].isnull().sum()
    total_nulls = null_counts.sum()
    result.add_check(
        passed=total_nulls == 0,
        name="No Null Values",
        detail=f"Found {total_nulls} nulls: {null_counts[null_counts > 0].to_dict()}"
        if total_nulls > 0 else "",
    )
    result.stats["null_counts"] = null_counts.to_dict()

    # ── 3. No negative prices ─────────────────────────────
    price_cols = ["Open", "High", "Low", "Close"]
    neg_mask = (df[price_cols] < 0).any(axis=1)
    neg_count = neg_mask.sum()
    result.add_check(
        passed=neg_count == 0,
        name="No Negative Prices",
        detail=f"Found {neg_count} rows with negative prices",
    )
    result.stats["negative_prices"] = int(neg_count)

    # ── 4. High >= Low ────────────────────────────────────
    hl_violation = (df["High"] < df["Low"]).sum()
    result.add_check(
        passed=hl_violation == 0,
        name="High >= Low",
        detail=f"Found {hl_violation} violations",
    )
    result.stats["high_low_violations"] = int(hl_violation)

    # ── 5. Open/Close within High/Low ─────────────────────
    ohlc_violation = (
        (df["Open"] > df["High"]) | (df["Open"] < df["Low"])
        | (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
    ).sum()
    result.add_check(
        passed=ohlc_violation == 0,
        name="Open/Close Within Range",
        detail=f"Found {ohlc_violation} violations",
    )
    result.stats["ohlc_range_violations"] = int(ohlc_violation)

    # ── 6. No negative volume ─────────────────────────────
    neg_vol = (df["Volume"] < 0).sum()
    result.add_check(
        passed=neg_vol == 0,
        name="No Negative Volume",
        detail=f"Found {neg_vol} rows with negative volume",
    )
    result.stats["negative_volume"] = int(neg_vol)

    # ── 7. Zero volume (warning only) ─────────────────────
    zero_vol = (df["Volume"] == 0).sum()
    zero_pct = zero_vol / len(df) * 100
    result.stats["zero_volume"] = int(zero_vol)
    result.stats["zero_volume_pct"] = round(zero_pct, 2)
    if zero_pct > 30:
        result.add_warning(
            f"High zero-volume ratio: {zero_vol:,} rows ({zero_pct:.1f}%)"
        )

    # ── 8. Monotonic index ────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        is_monotonic = df.index.is_monotonic_increasing
        result.add_check(
            passed=is_monotonic,
            name="Monotonic Index",
            detail="Index is not sorted chronologically",
        )

        # ── 9. Duplicate timestamps ───────────────────────
        dup_count = df.index.duplicated().sum()
        result.add_check(
            passed=dup_count == 0,
            name="No Duplicate Timestamps",
            detail=f"Found {dup_count} duplicates",
        )
        result.stats["duplicate_timestamps"] = int(dup_count)

    return result


def validate_date_range(
    df: pd.DataFrame,
    expected_start: str = None,
    expected_end: str = None,
) -> ValidationResult:
    """Validate that data covers the expected date range."""
    result = ValidationResult()

    if not isinstance(df.index, pd.DatetimeIndex):
        result.add_issue("Index is not DatetimeIndex — cannot validate dates")
        return result

    actual_start = df.index.min()
    actual_end = df.index.max()

    result.stats["actual_start"] = str(actual_start)
    result.stats["actual_end"] = str(actual_end)

    if expected_start:
        exp = pd.Timestamp(expected_start)
        result.add_check(
            passed=actual_start <= exp,
            name="Start Date Coverage",
            detail=f"Data starts at {actual_start}, expected <= {exp}",
        )

    if expected_end:
        exp = pd.Timestamp(expected_end)
        result.add_check(
            passed=actual_end >= exp,
            name="End Date Coverage",
            detail=f"Data ends at {actual_end}, expected >= {exp}",
        )

    return result
