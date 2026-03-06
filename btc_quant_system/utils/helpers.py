"""
BTC Quant Trading System — Helper Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Common utility functions used across the system.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
import time
from functools import wraps


# ── Config Loading ───────────────────────────────────────────
def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    yaml.YAMLError
        If the YAML is malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    return config


def merge_configs(*configs: Dict) -> Dict:
    """
    Deep-merge multiple config dicts (later ones override earlier).
    """
    result = {}
    for cfg in configs:
        _deep_merge(result, cfg)
    return result


def _deep_merge(base: Dict, override: Dict) -> None:
    """Recursively merge override into base in-place."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ── Directory Helpers ────────────────────────────────────────
def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path to ensure.

    Returns
    -------
    Path
        The resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Formatting Functions ────────────────────────────────────
def format_number(n: float, decimals: int = 2) -> str:
    """
    Format a number with thousands separators and fixed decimals.

    Examples
    --------
    >>> format_number(1234567.891, 2)
    '1,234,567.89'
    >>> format_number(0.00045, 5)
    '0.00045'
    """
    if n is None:
        return "N/A"
    return f"{n:,.{decimals}f}"


def format_pct(n: float, decimals: int = 2) -> str:
    """
    Format a decimal fraction as a percentage string.

    Examples
    --------
    >>> format_pct(0.1567)
    '15.67%'
    >>> format_pct(-0.032, 1)
    '-3.2%'
    """
    if n is None:
        return "N/A"
    return f"{n * 100:,.{decimals}f}%"


def format_currency(n: float, symbol: str = "$", decimals: int = 2) -> str:
    """
    Format a number as currency.

    Examples
    --------
    >>> format_currency(68795.50)
    '$68,795.50'
    >>> format_currency(3.80, '$', 2)
    '$3.80'
    """
    if n is None:
        return "N/A"
    return f"{symbol}{n:,.{decimals}f}"


def format_duration(seconds: float) -> str:
    """
    Format seconds into a human-readable duration.

    Examples
    --------
    >>> format_duration(3661.5)
    '1h 1m 1.50s'
    >>> format_duration(45.2)
    '45.20s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.2f}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m {s:.2f}s"


def format_large_number(n: float) -> str:
    """
    Format large numbers with K/M/B suffixes.

    Examples
    --------
    >>> format_large_number(1_500_000)
    '1.50M'
    >>> format_large_number(2_300)
    '2.30K'
    """
    if n is None:
        return "N/A"
    abs_n = abs(n)
    sign = "-" if n < 0 else ""

    if abs_n >= 1_000_000_000:
        return f"{sign}{abs_n / 1_000_000_000:.2f}B"
    elif abs_n >= 1_000_000:
        return f"{sign}{abs_n / 1_000_000:.2f}M"
    elif abs_n >= 1_000:
        return f"{sign}{abs_n / 1_000:.2f}K"
    else:
        return f"{sign}{abs_n:.2f}"


# ── Timing Decorator ────────────────────────────────────────
def timer(func):
    """Decorator to measure and print function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  ⏱  {func.__name__}() completed in {format_duration(elapsed)}")
        return result

    return wrapper


# ── Timeframe Helpers ────────────────────────────────────────
TIMEFRAME_MAP = {
    "1min":  "1min",
    "5min":  "5min",
    "15min": "15min",
    "30min": "30min",
    "1h":    "1h",
    "4h":    "4h",
    "1d":    "1D",
    "1w":    "1W",
}

TIMEFRAME_MINUTES = {
    "1min":  1,
    "5min":  5,
    "15min": 15,
    "30min": 30,
    "1h":    60,
    "4h":    240,
    "1d":    1440,
    "1w":    10080,
}


def tf_to_pandas(timeframe: str) -> str:
    """Convert our timeframe string to pandas resample rule."""
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. "
            f"Valid: {list(TIMEFRAME_MAP.keys())}"
        )
    return TIMEFRAME_MAP[timeframe]


def tf_to_minutes(timeframe: str) -> int:
    """Convert a timeframe string to number of minutes."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. "
            f"Valid: {list(TIMEFRAME_MINUTES.keys())}"
        )
    return TIMEFRAME_MINUTES[timeframe]
