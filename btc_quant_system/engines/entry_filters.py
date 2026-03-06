"""
Entry Filters — Precision Entry Timing
========================================
4 independent filters applied BEFORE entry execution:

1. PullbackEntry     — RSI-5 overbought/oversold check → wait for pullback zone
2. CandlePatternFilter — only enter on strong body candles with good structure
3. SRClearanceFilter — avoid entry within 0.3×ATR of a S/R level
4. TimeFilter        — skip low-liquidity hours, prefer London/NY overlap

Each filter returns a FilterResult(allowed: bool, reason: str, details: dict).
All 4 are combined in EntryFilterManager.check_all().
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# DATA CLASS
# ══════════════════════════════════════════════════════════════

@dataclass
class FilterResult:
    allowed: bool
    reason: str
    details: Dict


# ══════════════════════════════════════════════════════════════
# 1. PULLBACK ENTRY FILTER
# ══════════════════════════════════════════════════════════════

class PullbackEntry:
    """
    Avoid chasing: if short-term RSI shows overbought (LONG) or oversold (SHORT),
    define a pullback entry zone and wait up to max_wait_candles bars.

    Logic:
      LONG  → if RSI-5 > 70: wait for pullback to [close - 0.5×ATR, close]
      SHORT → if RSI-5 < 30: wait for pullback to [close, close + 0.5×ATR]

    State machine:
      None → WAITING (signal detected but overbought)
           → READY   (pullback zone touched → entry OK)
    """

    def __init__(self, rsi_period: int = 5, max_wait_candles: int = 5):
        self.rsi_period = rsi_period
        self.max_wait_candles = max_wait_candles

    # ── RSI computation (pure numpy, no TA-Lib dependency) ──────
    @staticmethod
    def _rsi(close: np.ndarray, period: int) -> float:
        """Compute RSI for the last bar using last (period+1) close prices."""
        n = period + 1
        if len(close) < n:
            return 50.0
        prices = close[-n:]
        deltas = np.diff(prices)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def should_wait_for_pullback(
        self,
        df: pd.DataFrame,
        direction: str,
        atr: float,
    ) -> Dict:
        """
        Check if we should wait for a pullback before entering.

        Returns dict with:
          wait             : bool
          entry_zone_low   : float  (only if wait=True)
          entry_zone_high  : float
          max_wait_candles : int
          rsi5             : float  (diagnostic)
        """
        close_col = "Close" if "Close" in df.columns else "close"
        close_arr = df[close_col].values
        rsi5 = self._rsi(close_arr, self.rsi_period)
        price = float(close_arr[-1])

        if direction == "LONG":
            if rsi5 > 70:
                return {
                    "wait": True,
                    "entry_zone_low":   price - 0.5 * atr,
                    "entry_zone_high":  price,
                    "max_wait_candles": self.max_wait_candles,
                    "rsi5": rsi5,
                }
        else:  # SHORT
            if rsi5 < 30:
                return {
                    "wait": True,
                    "entry_zone_low":   price,
                    "entry_zone_high":  price + 0.5 * atr,
                    "max_wait_candles": self.max_wait_candles,
                    "rsi5": rsi5,
                }

        return {"wait": False, "rsi5": rsi5}

    def check_pullback_entry(
        self,
        pending: Dict,
        row: pd.Series,
    ) -> Dict:
        """
        Check if the current bar satisfies the pullback entry zone.

        pending : dict from should_wait_for_pullback (must have wait=True)
        row     : current OHLC bar as pd.Series

        Returns dict with:
          entry        : bool
          price        : float
          better_entry : bool  (True if entry was improved vs original)
        """
        direction = pending.get("direction", "LONG")
        lo = float(row.get("Low", row.get("low", 0)))
        hi = float(row.get("High", row.get("high", 0)))
        op = float(row.get("Open", row.get("open", 0)))
        cl = float(row.get("Close", row.get("close", 0)))

        zone_low  = pending.get("entry_zone_low", 0)
        zone_high = pending.get("entry_zone_high", float("inf"))

        if direction == "LONG":
            # Price dipped into pullback zone AND candle closed bullish
            touched_zone = lo <= zone_high and cl > zone_low
            if touched_zone and cl > op:
                return {"entry": True, "price": cl, "better_entry": True}
        else:
            # SHORT: price bounced up into zone AND candle closed bearish
            touched_zone = hi >= zone_low and cl < zone_high
            if touched_zone and cl < op:
                return {"entry": True, "price": cl, "better_entry": True}

        return {"entry": False}


# ══════════════════════════════════════════════════════════════
# 2. CANDLE PATTERN FILTER
# ══════════════════════════════════════════════════════════════

class CandlePatternFilter:
    """
    Filter entries based on entry candle direction and structure.

    Hard rules (always applied):
      LONG  → candle must close bullish (close > open)
      SHORT → candle must close bearish (close < open)

    Optional strict_mode checks (disabled by default — too noisy on 4H):
      • Lower wick < body × max_wick_body_ratio  (LONG)
      • (close - low) / (high - low) > min_close_strength  (LONG)
    """

    def __init__(
        self,
        max_wick_body_ratio: float = 0.7,
        min_close_strength:  float = 0.45,
        min_body_pct:        float = 0.002,
        strict_mode:         bool  = False,   # enable wick/strength checks
    ):
        self.max_wick_body_ratio = max_wick_body_ratio
        self.min_close_strength  = min_close_strength
        self.min_body_pct        = min_body_pct
        self.strict_mode         = strict_mode

    def check_entry_candle(self, row: pd.Series, direction: str) -> FilterResult:
        """
        Returns FilterResult(allowed=True) only if candle direction qualifies.
        """
        op = float(row.get("Open",  row.get("open",  0)))
        cl = float(row.get("Close", row.get("close", 0)))
        hi = float(row.get("High",  row.get("high",  0)))
        lo = float(row.get("Low",   row.get("low",   0)))

        candle_range = hi - lo
        body         = abs(cl - op)
        price        = (hi + lo) / 2 or cl

        # Doji guard — only block very tiny bodies (0.1% of price)
        if candle_range < 1e-8 or (price > 0 and body / price < 0.001):
            return FilterResult(False, "Doji candle — no conviction",
                                {"body_pct": body / price * 100 if price else 0})

        body_top    = max(cl, op)
        body_bottom = min(cl, op)
        upper_wick  = hi - body_top
        lower_wick  = body_bottom - lo

        if direction == "LONG":
            if not (cl > op):
                return FilterResult(False, "LONG on a bearish candle — skip",
                                    {"close": cl, "open": op})
            if self.strict_mode:
                wick_ok = lower_wick < body * self.max_wick_body_ratio
                close_strength = (cl - lo) / candle_range if candle_range > 0 else 0
                if not wick_ok:
                    return FilterResult(False,
                        f"Lower wick too large ({lower_wick:.2f} > body×{self.max_wick_body_ratio})",
                        {"lower_wick": lower_wick, "body": body})
                if close_strength < self.min_close_strength:
                    return FilterResult(False,
                        f"Weak close strength ({close_strength:.2f} < {self.min_close_strength})",
                        {"close_strength": close_strength})

        else:  # SHORT
            if not (cl < op):
                return FilterResult(False, "SHORT on a bullish candle — skip",
                                    {"close": cl, "open": op})
            if self.strict_mode:
                wick_ok = upper_wick < body * self.max_wick_body_ratio
                close_strength = (hi - cl) / candle_range if candle_range > 0 else 0
                if not wick_ok:
                    return FilterResult(False,
                        f"Upper wick too large ({upper_wick:.2f} > body×{self.max_wick_body_ratio})",
                        {"upper_wick": upper_wick, "body": body})
                if close_strength < self.min_close_strength:
                    return FilterResult(False,
                        f"Weak close strength ({close_strength:.2f} < {self.min_close_strength})",
                        {"close_strength": close_strength})

        return FilterResult(True, "Candle pattern OK",
                            {"body": body, "upper_wick": upper_wick,
                             "lower_wick": lower_wick})


# ══════════════════════════════════════════════════════════════
# 3. SUPPORT / RESISTANCE CLEARANCE FILTER
# ══════════════════════════════════════════════════════════════

class SRClearanceFilter:
    """
    Identify nearby support and resistance levels from recent
    swing highs/lows and avoid entering when price is too close.

    S/R levels are computed from the last `lookback` candles:
      - Swing high: local max in ±window bars
      - Swing low : local min in ±window bars

    Clearance rule:
      LONG  → nearest resistance must be ≥ min_clearance_atr × ATR above entry
      SHORT → nearest support   must be ≥ min_clearance_atr × ATR below entry
    """

    def __init__(
        self,
        lookback:           int   = 100,
        swing_window:       int   = 5,
        min_clearance_atr:  float = 0.3,
    ):
        self.lookback          = lookback
        self.swing_window      = swing_window
        self.min_clearance_atr = min_clearance_atr

    def _find_sr_levels(self, df: pd.DataFrame, idx: int) -> Tuple[List[float], List[float]]:
        """
        Find swing highs (resistance) and swing lows (support) in
        [idx - lookback, idx].

        Returns (support_levels, resistance_levels).
        """
        hi_col = "High" if "High" in df.columns else "high"
        lo_col = "Low"  if "Low"  in df.columns else "low"

        start = max(0, idx - self.lookback)
        end   = idx + 1
        highs = df[hi_col].values[start:end]
        lows  = df[lo_col].values[start:end]
        w = self.swing_window

        supports    = []
        resistances = []

        for i in range(w, len(highs) - w):
            # Swing high
            if highs[i] == max(highs[i - w: i + w + 1]):
                resistances.append(float(highs[i]))
            # Swing low
            if lows[i] == min(lows[i - w: i + w + 1]):
                supports.append(float(lows[i]))

        return supports, resistances

    def check_sr_clearance(
        self,
        df:          pd.DataFrame,
        idx:         int,
        entry_price: float,
        direction:   str,
        atr:         float,
    ) -> FilterResult:
        """
        Returns FilterResult(allowed=True) if entry has enough room before
        the nearest opposing S/R level.
        """
        supports, resistances = self._find_sr_levels(df, idx)
        min_clearance = self.min_clearance_atr * atr

        if direction == "LONG":
            above_entry = [r for r in resistances if r > entry_price]
            if not above_entry:
                return FilterResult(True, "No resistance above — clear",
                                    {"nearest_resistance": None})
            nearest_r = min(above_entry)
            clearance = nearest_r - entry_price
            if clearance < min_clearance:
                return FilterResult(False,
                    f"Too close to resistance ({clearance:.2f} < {min_clearance:.2f})",
                    {"nearest_resistance": nearest_r,
                     "clearance": clearance,
                     "min_clearance": min_clearance})
            return FilterResult(True,
                f"Resistance clearance OK ({clearance:.2f} ≥ {min_clearance:.2f})",
                {"nearest_resistance": nearest_r, "clearance": clearance})

        else:  # SHORT
            below_entry = [s for s in supports if s < entry_price]
            if not below_entry:
                return FilterResult(True, "No support below — clear",
                                    {"nearest_support": None})
            nearest_s = max(below_entry)
            clearance = entry_price - nearest_s
            if clearance < min_clearance:
                return FilterResult(False,
                    f"Too close to support ({clearance:.2f} < {min_clearance:.2f})",
                    {"nearest_support": nearest_s,
                     "clearance": clearance,
                     "min_clearance": min_clearance})
            return FilterResult(True,
                f"Support clearance OK ({clearance:.2f} ≥ {min_clearance:.2f})",
                {"nearest_support": nearest_s, "clearance": clearance})


# ══════════════════════════════════════════════════════════════
# 4. TIME FILTER
# ══════════════════════════════════════════════════════════════

class TimeFilter:
    """
    Filter entries by UTC hour.

    BTC trades 24/7 but liquidity varies substantially:
      - Dead zone (00:00–05:59 UTC): low Asia/Pacific liquidity — BLOCK
      - London open (07:00–09:59 UTC): high liquidity — PREFER
      - NY open / NY-London overlap (13:00–17:59 UTC): highest liquidity — PREFER
      - Other hours: ALLOW (normal)

    Configurable via blocked_hours / preferred_hours.
    """

    # Default settings (UTC hours)
    BLOCKED_HOURS:   List[int] = [1, 2, 3]      # narrowed: only deepest dead zone
    PREFERRED_HOURS: List[int] = [7, 8, 9, 13, 14, 15, 16, 17]

    def __init__(
        self,
        blocked_hours:   Optional[List[int]] = None,
        preferred_hours: Optional[List[int]] = None,
        enabled:         bool = True,
    ):
        self.blocked_hours   = blocked_hours   or self.BLOCKED_HOURS
        self.preferred_hours = preferred_hours or self.PREFERRED_HOURS
        self.enabled         = enabled

    def check_time_filter(self, timestamp) -> FilterResult:
        """
        Returns FilterResult indicating whether this timestamp is
        suitable for entry.
        """
        if not self.enabled:
            return FilterResult(True, "Time filter disabled", {})

        try:
            hour = int(timestamp.hour)
        except Exception:
            return FilterResult(True, "Could not parse timestamp hour", {})

        if hour in self.blocked_hours:
            return FilterResult(False,
                f"Low-liquidity hour (UTC {hour:02d}:xx) — skipping",
                {"hour": hour, "zone": "dead_zone"})

        if hour in self.preferred_hours:
            return FilterResult(True,
                f"High-liquidity window (UTC {hour:02d}:xx) — preferred",
                {"hour": hour, "zone": "preferred"})

        return FilterResult(True,
            f"Normal liquidity (UTC {hour:02d}:xx)",
            {"hour": hour, "zone": "normal"})


# ══════════════════════════════════════════════════════════════
# ENTRY FILTER MANAGER (combines all 4)
# ══════════════════════════════════════════════════════════════

class EntryFilterManager:
    """
    Orchestrates all 4 entry filters.

    Usage:
        from engines.entry_filters import EntryFilterManager

        efm = EntryFilterManager(config)
        result = efm.check_all(df, idx, signal, row)
        if result['allowed']:
            # proceed to entry
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        ef_cfg = cfg.get("entry_filters", {})
        self.enabled = ef_cfg.get("enabled", True)

        self.pullback_filter = PullbackEntry(
            rsi_period       = ef_cfg.get("rsi_period", 5),
            max_wait_candles = ef_cfg.get("max_wait_candles", 5),
        )
        self.candle_filter = CandlePatternFilter(
            max_wick_body_ratio = ef_cfg.get("max_wick_body_ratio", 0.7),
            min_close_strength  = ef_cfg.get("min_close_strength",  0.45),
            min_body_pct        = ef_cfg.get("min_body_pct",         0.002),
        )
        self.sr_filter = SRClearanceFilter(
            lookback          = ef_cfg.get("sr_lookback",          100),
            swing_window      = ef_cfg.get("sr_swing_window",        5),
            min_clearance_atr = ef_cfg.get("sr_min_clearance_atr",  0.3),
        )
        self.time_filter = TimeFilter(
            blocked_hours   = ef_cfg.get("blocked_hours",   None),
            preferred_hours = ef_cfg.get("preferred_hours", None),
            enabled         = ef_cfg.get("time_filter_enabled", True),
        )

        # Pullback waiting state
        self._waiting:    bool       = False
        self._wait_signal: Dict      = {}
        self._wait_bars:  int        = 0

        # Stats tracking
        self.stats = {
            "checked":           0,
            "passed":            0,
            "blocked_pullback":  0,
            "pullback_improved": 0,
            "blocked_candle":    0,
            "blocked_sr":        0,
            "blocked_time":      0,
        }

    def reset_wait(self) -> None:
        """Call when a trade is opened or a new signal is generated."""
        self._waiting    = False
        self._wait_signal = {}
        self._wait_bars  = 0

    def check_all(
        self,
        df:        pd.DataFrame,
        idx:       int,
        signal:    Dict,
        row:       pd.Series,
    ) -> Dict:
        """
        Run all entry filters for the current bar.

        Parameters
        ----------
        df     : full DataFrame
        idx    : current bar index
        signal : dict from signal_engine.generate_trading_signal()
        row    : df.iloc[idx]

        Returns
        -------
        dict with:
          allowed          : bool  — True = proceed to entry
          reason           : str   — human-readable summary
          is_pullback_wait : bool  — if True, engine should wait (don't enter yet)
          improved_price   : float | None — pullback-improved entry price
          filter_details   : dict  — per-filter results
        """
        if not self.enabled:
            return {"allowed": True, "reason": "Entry filters disabled",
                    "is_pullback_wait": False, "improved_price": None, "filter_details": {}}

        self.stats["checked"] += 1

        direction = signal.get("direction", "LONG")
        entry_price = float(signal.get("entry_price", row.get("Close", row.get("close", 0))))
        atr = float(row.get("atr_14", entry_price * 0.02) or entry_price * 0.02)
        timestamp = row.name

        # ── 1. TIME FILTER ───────────────────────────────────────────
        time_result = self.time_filter.check_time_filter(timestamp)
        if not time_result.allowed:
            self.stats["blocked_time"] += 1
            return {
                "allowed": False,
                "reason": time_result.reason,
                "is_pullback_wait": False,
                "improved_price": None,
                "filter_details": {"time": time_result.reason},
            }

        # ── 2. CANDLE PATTERN FILTER ─────────────────────────────────
        candle_result = self.candle_filter.check_entry_candle(row, direction)
        if not candle_result.allowed:
            self.stats["blocked_candle"] += 1
            return {
                "allowed": False,
                "reason": candle_result.reason,
                "is_pullback_wait": False,
                "improved_price": None,
                "filter_details": {
                    "time":   time_result.reason,
                    "candle": candle_result.reason,
                },
            }

        # ── 3. PULLBACK FILTER ───────────────────────────────────────
        # If we are in a wait state, check if pullback zone is now reached
        if self._waiting:
            self._wait_bars += 1
            if self._wait_bars > self._wait_signal.get("max_wait_candles", 5):
                # Timeout — give up waiting, allow normal entry next signal
                self.reset_wait()
                # Don't enter now; let next signal trigger
                return {
                    "allowed": False,
                    "reason": "Pullback wait timed out — resetting",
                    "is_pullback_wait": False,
                    "improved_price": None,
                    "filter_details": {"pullback": "Wait timeout"},
                }
            pb_check = self.pullback_filter.check_pullback_entry(
                {**self._wait_signal, "direction": direction}, row
            )
            if pb_check["entry"]:
                self.reset_wait()
                self.stats["pullback_improved"] += 1
                improved = pb_check["price"]
            else:
                return {
                    "allowed": False,
                    "reason": "Waiting for pullback entry zone",
                    "is_pullback_wait": True,
                    "improved_price": None,
                    "filter_details": {"pullback": "Waiting for pullback"},
                }
        else:
            # Check if we should start waiting
            pb_state = self.pullback_filter.should_wait_for_pullback(df, direction, atr)
            improved = None
            if pb_state["wait"]:
                self.stats["blocked_pullback"] += 1
                self._waiting    = True
                self._wait_bars  = 0
                self._wait_signal = {**pb_state, "direction": direction}
                return {
                    "allowed": False,
                    "reason": f"RSI-5 at {pb_state['rsi5']:.1f} — waiting for pullback",
                    "is_pullback_wait": True,
                    "improved_price": None,
                    "filter_details": {
                        "time":    time_result.reason,
                        "candle":  candle_result.reason,
                        "pullback": f"RSI-5={pb_state['rsi5']:.1f} → waiting",
                    },
                }

        # ── 4. S/R CLEARANCE FILTER ──────────────────────────────────
        sr_result = self.sr_filter.check_sr_clearance(df, idx, entry_price, direction, atr)
        if not sr_result.allowed:
            self.stats["blocked_sr"] += 1
            return {
                "allowed": False,
                "reason": sr_result.reason,
                "is_pullback_wait": False,
                "improved_price": None,
                "filter_details": {
                    "time":   time_result.reason,
                    "candle": candle_result.reason,
                    "sr":     sr_result.reason,
                },
            }

        # ── All filters passed ───────────────────────────────────────
        self.stats["passed"] += 1
        return {
            "allowed":          True,
            "reason":           "All entry filters passed",
            "is_pullback_wait": False,
            "improved_price":   improved,
            "filter_details": {
                "time":   time_result.reason,
                "candle": candle_result.reason,
                "sr":     sr_result.reason,
            },
        }

    def summary(self) -> Dict:
        """Return filter pass/block statistics."""
        total = self.stats["checked"]
        if total == 0:
            return self.stats
        return {
            **self.stats,
            "pass_rate":            self.stats["passed"] / total * 100,
            "block_rate_time":      self.stats["blocked_time"] / total * 100,
            "block_rate_candle":    self.stats["blocked_candle"] / total * 100,
            "block_rate_pullback":  self.stats["blocked_pullback"] / total * 100,
            "block_rate_sr":        self.stats["blocked_sr"] / total * 100,
            "pullback_improved":    self.stats["pullback_improved"],
        }
