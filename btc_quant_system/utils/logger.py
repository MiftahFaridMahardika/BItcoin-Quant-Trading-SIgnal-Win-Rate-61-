"""
BTC Quant Trading System — Logging System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides structured logging with:
- File logging with rotation (max 10MB, keep 5 files)
- Console logging with colors
- Separate loggers for: main, trading, performance
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


# ── ANSI Color Codes ─────────────────────────────────────────
class Colors:
    """ANSI color codes for terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    BG_RED  = "\033[41m"

    LEVEL_COLORS = {
        "DEBUG":    CYAN,
        "INFO":     GREEN,
        "WARNING":  YELLOW,
        "ERROR":    RED,
        "CRITICAL": BG_RED + WHITE + BOLD,
    }


# ── Color Formatter ──────────────────────────────────────────
class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to console output."""

    FMT = (
        "{color}[{levelname:<8}]{reset} "
        "{dim}{asctime}{reset} "
        "{bold}{name}{reset} → "
        "{message}"
    )

    def format(self, record: logging.LogRecord) -> str:
        color = Colors.LEVEL_COLORS.get(record.levelname, Colors.WHITE)
        record.msg = self.FMT.format(
            color=color,
            reset=Colors.RESET,
            dim=Colors.DIM,
            bold=Colors.BOLD,
            levelname=record.levelname,
            asctime=self.formatTime(record, "%H:%M:%S"),
            name=record.name,
            message=record.getMessage(),
        )
        record.args = None  # Prevent double-formatting
        return record.msg


# ── File Formatter ───────────────────────────────────────────
class FileFormatter(logging.Formatter):
    """Clean formatter for log files (no color codes)."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ── Setup Functions ──────────────────────────────────────────
def setup_logger(
    name: str = "btc_quant",
    log_dir: str = "logs",
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create and configure a logger with console and file handlers.

    Parameters
    ----------
    name : str
        Logger name (e.g., 'main', 'trading', 'performance').
    log_dir : str
        Directory for log files.
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    console : bool
        Enable console (stdout) logging.
    file : bool
        Enable file logging with rotation.
    max_bytes : int
        Max size per log file before rotation (default 10 MB).
    backup_count : int
        Number of rotated log files to keep (default 5).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # ── Console Handler ───────────────────────────────────
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColorFormatter())
        logger.addHandler(console_handler)

    # ── File Handler (rotating) ───────────────────────────
    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / f"{name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    Falls back to creating a console-only logger if not yet set up.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name, console=True, file=False)
    return logger


# ── Pre-configured Loggers ───────────────────────────────────
def setup_all_loggers(
    log_dir: str = "logs",
    level: str = "INFO",
) -> dict:
    """
    Set up all system loggers at once.

    Returns dict of {name: logger} for:
      - main         : General system operations
      - trading      : Trade signals and execution
      - performance  : Backtest metrics and performance
      - data         : Data pipeline operations
    """
    logger_names = ["main", "trading", "performance", "data"]
    loggers = {}

    for name in logger_names:
        loggers[name] = setup_logger(
            name=name,
            log_dir=log_dir,
            level=level,
        )

    loggers["main"].info(
        f"Logger system initialized — {len(loggers)} loggers active "
        f"(dir={log_dir}, level={level})"
    )

    return loggers
