"""
BTC Quant Trading System — Signal Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
6-Layer Weighted Scoring System for trade signal generation.

Aggregates 71 technical indicators into a composite score
ranging from -19 to +19 using hedge-fund-style methodology.

Weight Allocation (total: 19):
  Layer 1  Trend       :  7  (EMA stack 3, HMA 2, Supertrend 2)
  Layer 2  Momentum    :  5  (RSI 2, MACD 2, Z-score 1)
  Layer 3  Volatility  :  2  (Bollinger 1, Keltner 1)
  Layer 4  Volume      :  3  (Volume confirm 2, OBV 1)
  Layer 5  Price Action:  2  (Momentum 1, Structure 1)

  Regime filter (ATR ratio) acts as a BLOCKER, not a scorer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger("trading")


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    """Container for a single-bar signal computation."""
    timestamp: pd.Timestamp
    score: int
    max_score: int
    signal: str         # STRONG_LONG | LONG | SKIP | SHORT | STRONG_SHORT
    confidence: float   # 0.0 – 1.0
    regime: str         # LOW | NORMAL | HIGH | EXTREME
    individual_signals: Dict[str, int] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════

class SignalEngine:
    """
    Multi-Layer Weighted Scoring Signal Generator.

    Workflow:
      1. For each bar, evaluate 11 signal checkers across 5 layers.
      2. Each checker returns +1 (bullish), -1 (bearish), or 0 (neutral).
      3. Multiply by the checker's weight and sum.
      4. The volatility regime acts as a blocker (EXTREME → force SKIP).
      5. Classify the total score into a trading signal.
    """

    # ── Weight table ──────────────────────────────────────
    WEIGHTS = {
        # Layer 1: Trend (7)
        "ema_structure":   3,
        "hma_direction":   2,
        "supertrend":      2,
        # Layer 2: Momentum (5)
        "rsi_divergence":  2,
        "macd_histogram":  2,
        "zscore":          1,
        # Layer 3: Volatility (2)
        "bollinger":       1,
        "keltner":         1,
        # Layer 4: Volume (3)
        "volume_confirm":  2,
        "obv_trend":       1,
        # Layer 5: Price Action (2)
        "price_momentum":  1,
        "trend_structure": 1,
    }

    MAX_SCORE = sum(WEIGHTS.values())   # 19
    MIN_SCORE = -MAX_SCORE              # -19

    # Classification thresholds
    STRONG_LONG_THRESHOLD  =  8
    LONG_THRESHOLD         =  4
    SHORT_THRESHOLD        = -4
    STRONG_SHORT_THRESHOLD = -8

    LAYER_MAP = {
        "L1_Trend":       ["ema_structure", "hma_direction", "supertrend"],
        "L2_Momentum":    ["rsi_divergence", "macd_histogram", "zscore"],
        "L3_Volatility":  ["bollinger", "keltner"],
        "L4_Volume":      ["volume_confirm", "obv_trend"],
        "L5_PriceAction": ["price_momentum", "trend_structure"],
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.min_confidence = self.config.get("min_confidence", 0.60)

    # ══════════════════════════════════════════════════════════
    # INDIVIDUAL SIGNAL CHECKERS
    # Each returns (direction: int, reason: str)
    #   direction ∈ {+1, 0, -1}
    # ══════════════════════════════════════════════════════════

    # ── Layer 1: Trend ────────────────────────────────────

    @staticmethod
    def check_ema_structure(row: pd.Series) -> Tuple[int, str]:
        """
        EMA 21/55/200 stack alignment.

        +1: EMA21 > EMA55 > EMA200 (bullish)
        -1: EMA21 < EMA55 < EMA200 (bearish)
         0: mixed / no clear stack
        """
        sig = int(row.get("ema_stack_signal", 0))
        if sig == 1:
            return 1, "EMA bullish stack (21>55>200)"
        elif sig == -1:
            return -1, "EMA bearish stack (21<55<200)"
        return 0, ""

    @staticmethod
    def check_hma_direction(row: pd.Series) -> Tuple[int, str]:
        """
        Hull MA slope direction.

        +1: HMA rising
        -1: HMA falling
        """
        sig = int(row.get("hma_signal", 0))
        if sig == 1:
            return 1, "HMA rising"
        elif sig == -1:
            return -1, "HMA falling"
        return 0, ""

    @staticmethod
    def check_supertrend(row: pd.Series) -> Tuple[int, str]:
        """
        Supertrend direction.

        +1: uptrend
        -1: downtrend
        """
        sig = int(row.get("supertrend_dir", 0))
        if sig == 1:
            return 1, "Supertrend UP"
        elif sig == -1:
            return -1, "Supertrend DOWN"
        return 0, ""

    # ── Layer 2: Momentum ─────────────────────────────────

    @staticmethod
    def check_rsi_divergence(row: pd.Series) -> Tuple[int, str]:
        """
        RSI divergence-aware signal.

        +1: RSI < 35 AND RSI slope positive (bullish reversal)
        -1: RSI > 65 AND RSI slope negative (bearish reversal)
        """
        sig = int(row.get("rsi_signal", 0))
        rsi = row.get("rsi", 50)
        if sig == 1:
            return 1, f"RSI bullish reversal ({rsi:.0f})"
        elif sig == -1:
            return -1, f"RSI bearish reversal ({rsi:.0f})"
        return 0, ""

    @staticmethod
    def check_macd_histogram(row: pd.Series, prev_row: pd.Series) -> Tuple[int, str]:
        """
        MACD histogram zero-line cross with trend confirmation.

        +1: histogram flips positive AND MACD line > 0
        -1: histogram flips negative AND MACD line < 0
        """
        hist_now = row.get("macd_hist", 0)
        hist_prev = prev_row.get("macd_hist", 0)
        macd_line = row.get("macd_line", 0)

        if hist_prev <= 0 < hist_now and macd_line > 0:
            return 1, f"MACD hist bullish cross (hist={hist_now:.0f})"
        if hist_prev >= 0 > hist_now and macd_line < 0:
            return -1, f"MACD hist bearish cross (hist={hist_now:.0f})"
        return 0, ""

    @staticmethod
    def check_zscore(row: pd.Series) -> Tuple[int, str]:
        """
        Z-Score mean reversion.

        +1: Z < -2.0  (oversold — expect bounce)
        -1: Z > +2.0  (overbought — expect pullback)
        """
        z = row.get("zscore_20", 0)
        if pd.isna(z):
            return 0, ""
        if z < -2.0:
            return 1, f"Z-score oversold ({z:.2f})"
        if z > 2.0:
            return -1, f"Z-score overbought ({z:.2f})"
        return 0, ""

    # ── Layer 3: Volatility ───────────────────────────────

    @staticmethod
    def check_bollinger(row: pd.Series) -> Tuple[int, str]:
        """
        Bollinger %B with bandwidth expansion filter.

        +1: %B < 0.2 AND bandwidth expanding
        -1: %B > 0.8 AND bandwidth expanding
         0: squeeze / neutral
        """
        sig = int(row.get("bb_signal", 0))
        pct_b = row.get("bb_pct_b", 0.5)
        if sig == 1:
            return 1, f"BB oversold (%B={pct_b:.2f})"
        elif sig == -1:
            return -1, f"BB overbought (%B={pct_b:.2f})"
        return 0, ""

    @staticmethod
    def check_keltner(row: pd.Series) -> Tuple[int, str]:
        """
        Keltner Channel breakout.

        +1: Close > upper KC (bullish breakout)
        -1: Close < lower KC (bearish breakout)
        """
        sig = int(row.get("kc_signal", 0))
        if sig == 1:
            return 1, "KC bullish breakout"
        elif sig == -1:
            return -1, "KC bearish breakout"
        return 0, ""

    # ── Layer 4: Volume ───────────────────────────────────

    @staticmethod
    def check_volume_confirm(row: pd.Series) -> Tuple[int, str]:
        """
        Volume confirmation — above-average volume in price direction.

        +1: vol_ratio > 1.5 AND price up (Close > Open)
        -1: vol_ratio > 1.5 AND price down (Close < Open)
         0: below-average volume
        """
        vol_ratio = row.get("vol_ratio", 0)
        if pd.isna(vol_ratio):
            return 0, ""
        if vol_ratio > 1.5:
            price_up = row.get("Close", 0) > row.get("Open", 0)
            if price_up:
                return 1, f"High vol bullish ({vol_ratio:.1f}x avg)"
            else:
                return -1, f"High vol bearish ({vol_ratio:.1f}x avg)"
        return 0, ""

    @staticmethod
    def check_obv_trend(row: pd.Series) -> Tuple[int, str]:
        """
        OBV divergence signal.

        +1: OBV rising while price flat/down (accumulation)
        -1: OBV falling while price flat/up (distribution)
        """
        sig = int(row.get("obv_signal", 0))
        if sig == 1:
            return 1, "OBV hidden bullish divergence"
        elif sig == -1:
            return -1, "OBV bearish divergence"
        return 0, ""

    # ── Layer 5: Price Action ─────────────────────────────

    @staticmethod
    def check_price_momentum(row: pd.Series) -> Tuple[int, str]:
        """
        Multi-horizon momentum alignment.

        +1: ret_1 > 0 AND ret_6 > 0 AND ret_42 > 0 (all positive)
        -1: ret_1 < 0 AND ret_6 < 0 AND ret_42 < 0 (all negative)
         0: mixed
        """
        r1 = row.get("ret_1", 0)
        r6 = row.get("ret_6", 0)
        r42 = row.get("ret_42", 0)

        if pd.isna(r1) or pd.isna(r6) or pd.isna(r42):
            return 0, ""

        if r1 > 0 and r6 > 0 and r42 > 0:
            return 1, f"Momentum aligned bullish ({r1:.1f}%/{r6:.1f}%/{r42:.1f}%)"
        elif r1 < 0 and r6 < 0 and r42 < 0:
            return -1, f"Momentum aligned bearish ({r1:.1f}%/{r6:.1f}%/{r42:.1f}%)"
        return 0, ""

    @staticmethod
    def check_trend_structure(row: pd.Series) -> Tuple[int, str]:
        """
        HH/HL vs LH/LL trend structure.

        +1: Higher highs + higher lows (uptrend)
        -1: Lower highs + lower lows (downtrend)
         0: ranging
        """
        sig = int(row.get("trend_structure", 0))
        if sig == 1:
            return 1, "Trend structure: HH/HL (uptrend)"
        elif sig == -1:
            return -1, "Trend structure: LH/LL (downtrend)"
        return 0, ""

    # ══════════════════════════════════════════════════════════
    # REGIME FILTER (BLOCKER)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def check_volatility_regime(row: pd.Series) -> str:
        """
        Volatility regime from ATR ratio.

        0 → LOW      (< 0.8× baseline) — favour mean-reversion
        1 → NORMAL   (0.8–1.2×)
        2 → HIGH     (1.2–2.0×) — favour trend-following
        3 → EXTREME  (> 2.0×) — BLOCK ALL TRADING
        """
        regime_code = row.get("vol_regime", 1)
        return {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}.get(
            int(regime_code), "NORMAL"
        )

    # ══════════════════════════════════════════════════════════
    # MASTER SIGNAL CALCULATION (single bar)
    # ══════════════════════════════════════════════════════════

    def calculate_signal_score(
        self,
        df: pd.DataFrame,
        idx: int,
    ) -> SignalResult:
        """
        Calculate the weighted composite signal score for one bar.

        Parameters
        ----------
        df : pd.DataFrame
            Featured OHLCV data (output of FeatureEngine).
        idx : int
            Positional index of the bar to evaluate.

        Returns
        -------
        SignalResult
        """
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1] if idx > 0 else row

        # ── Regime blocker ────────────────────────────────
        regime = self.check_volatility_regime(row)
        if regime == "EXTREME":
            return SignalResult(
                timestamp=row.name,
                score=0,
                max_score=self.MAX_SCORE,
                signal="SKIP",
                confidence=0.0,
                regime=regime,
                individual_signals={},
                reasons=["BLOCKED: Volatility regime EXTREME — no trading"],
            )

        # ── Evaluate all checkers ─────────────────────────
        signals: Dict[str, int] = {}
        reasons: List[str] = []
        total_score = 0

        checkers = [
            # (key,               callable,                          needs_prev)
            ("ema_structure",     lambda r, _: self.check_ema_structure(r),       False),
            ("hma_direction",     lambda r, _: self.check_hma_direction(r),       False),
            ("supertrend",        lambda r, _: self.check_supertrend(r),          False),
            ("rsi_divergence",    lambda r, _: self.check_rsi_divergence(r),      False),
            ("macd_histogram",    lambda r, p: self.check_macd_histogram(r, p),   True),
            ("zscore",            lambda r, _: self.check_zscore(r),              False),
            ("bollinger",         lambda r, _: self.check_bollinger(r),           False),
            ("keltner",           lambda r, _: self.check_keltner(r),             False),
            ("volume_confirm",    lambda r, _: self.check_volume_confirm(r),      False),
            ("obv_trend",         lambda r, _: self.check_obv_trend(r),           False),
            ("price_momentum",    lambda r, _: self.check_price_momentum(r),      False),
            ("trend_structure",   lambda r, _: self.check_trend_structure(r),     False),
        ]

        for key, checker_fn, _ in checkers:
            sig, reason = checker_fn(row, prev_row)
            signals[key] = sig
            weighted = sig * self.WEIGHTS[key]
            total_score += weighted
            if sig != 0 and reason:
                reasons.append(
                    f"{'+'if weighted>0 else ''}{weighted:d} {reason}"
                )

        # ── Classify ──────────────────────────────────────
        signal_type, confidence = self._classify_signal(total_score)

        return SignalResult(
            timestamp=row.name,
            score=total_score,
            max_score=self.MAX_SCORE,
            signal=signal_type,
            confidence=confidence,
            regime=regime,
            individual_signals=signals,
            reasons=reasons,
        )

    # ══════════════════════════════════════════════════════════
    # CLASSIFICATION
    # ══════════════════════════════════════════════════════════

    def _classify_signal(self, score: int) -> Tuple[str, float]:
        """
        Map a raw score to a signal label + confidence.

        Score ≥  +8  →  STRONG_LONG   conf = score / MAX
        Score +4…+7  →  LONG          conf = score / 8
        Score -3…+3  →  SKIP          conf = 0.0
        Score -7…-4  →  SHORT         conf = |score| / 8
        Score ≤  -8  →  STRONG_SHORT  conf = |score| / MAX
        """
        if score >= self.STRONG_LONG_THRESHOLD:
            return "STRONG_LONG", min(score / self.MAX_SCORE, 1.0)
        if score >= self.LONG_THRESHOLD:
            return "LONG", score / self.STRONG_LONG_THRESHOLD
        if score <= self.STRONG_SHORT_THRESHOLD:
            return "STRONG_SHORT", min(abs(score) / self.MAX_SCORE, 1.0)
        if score <= self.SHORT_THRESHOLD:
            return "SHORT", abs(score) / abs(self.STRONG_SHORT_THRESHOLD)
        return "SKIP", 0.0

    # ══════════════════════════════════════════════════════════
    # COMPLETE TRADING SIGNAL (with SL / TP)
    # ══════════════════════════════════════════════════════════

    def generate_trading_signal(
        self,
        df: pd.DataFrame,
        idx: int,
    ) -> Dict:
        """
        Generate a full trading signal with entry / SL / TP levels.

        Uses ATR from the feature set for dynamic SL/TP sizing.
        TP levels follow a tiered exit plan (33% / 33% / 34%).

        Returns
        -------
        dict with keys: signal, direction, entry_price, stop_loss,
                        tp1, tp2, tp3, risk_reward, confidence,
                        score, regime, reasons, timestamp
        """
        result = self.calculate_signal_score(df, idx)

        if result.signal == "SKIP":
            return {
                "signal": "SKIP",
                "score": result.score,
                "regime": result.regime,
                "timestamp": result.timestamp,
                "reasons": result.reasons,
            }

        row = df.iloc[idx]
        price = float(row["Close"])
        atr = float(row.get("atr_14", 0))

        # Fallback if ATR is missing
        if pd.isna(atr) or atr <= 0:
            atr = price * 0.02  # 2% fallback

        direction = "LONG" if result.score > 0 else "SHORT"

        # SL / TP multipliers (from risk_config defaults)
        sl_mult  = 1.5
        tp1_mult = 2.0
        tp2_mult = 3.5
        tp3_mult = 5.0

        sign = 1 if direction == "LONG" else -1
        sl  = price - sign * atr * sl_mult
        tp1 = price + sign * atr * tp1_mult
        tp2 = price + sign * atr * tp2_mult
        tp3 = price + sign * atr * tp3_mult

        risk = abs(price - sl)
        rr = abs(tp2 - price) / risk if risk > 0 else 0.0

        return {
            "signal":      result.signal,
            "direction":   direction,
            "entry_price": round(price, 2),
            "stop_loss":   round(sl, 2),
            "tp1":         round(tp1, 2),
            "tp2":         round(tp2, 2),
            "tp3":         round(tp3, 2),
            "risk_reward": round(rr, 2),
            "confidence":  round(result.confidence, 4),
            "score":       result.score,
            "regime":      result.regime,
            "reasons":     result.reasons,
            "timestamp":   result.timestamp,
            "individual":  result.individual_signals,
        }

    # ══════════════════════════════════════════════════════════
    # BATCH SIGNAL GENERATION (vectorised where possible)
    # ══════════════════════════════════════════════════════════

    def generate_signals_batch(
        self,
        df: pd.DataFrame,
        start_idx: int = 1,
    ) -> pd.DataFrame:
        """
        Generate signal scores for every bar in the DataFrame.

        Adds columns:
          signal_score, signal_type, signal_confidence, signal_regime

        Parameters
        ----------
        df : pd.DataFrame
            Featured OHLCV data.
        start_idx : int
            First bar to evaluate (skip warmup rows).

        Returns
        -------
        pd.DataFrame  (same df with new signal columns)
        """
        logger.info(
            f"Generating batch signals for {len(df):,} bars "
            f"(start_idx={start_idx})..."
        )
        t0 = time.perf_counter()

        n = len(df)
        scores = np.zeros(n, dtype=int)
        types = np.full(n, "SKIP", dtype=object)
        confs = np.zeros(n, dtype=float)
        regimes = np.full(n, "NORMAL", dtype=object)

        # ── Vectorised regime (blocker) ───────────────────
        regime_map = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
        vol_regime = df["vol_regime"].fillna(1).astype(int)
        regimes_arr = vol_regime.map(regime_map).values
        extreme_mask = regimes_arr == "EXTREME"

        # ── Vectorised individual signals ─────────────────
        # Layer 1
        sig_ema = df["ema_stack_signal"].fillna(0).astype(int).values
        sig_hma = df["hma_signal"].fillna(0).astype(int).values
        sig_st = df["supertrend_dir"].fillna(0).astype(int).values

        # Layer 2
        sig_rsi = df["rsi_signal"].fillna(0).astype(int).values
        # MACD histogram cross: need prev bar comparison
        hist = df["macd_hist"].fillna(0).values
        macd_l = df["macd_line"].fillna(0).values
        hist_prev = np.roll(hist, 1)
        hist_prev[0] = 0
        sig_macd = np.where(
            (hist_prev <= 0) & (hist > 0) & (macd_l > 0), 1,
            np.where(
                (hist_prev >= 0) & (hist < 0) & (macd_l < 0), -1, 0
            ),
        )
        sig_z = df["zscore_signal"].fillna(0).astype(int).values

        # Layer 3
        sig_bb = df["bb_signal"].fillna(0).astype(int).values
        sig_kc = df["kc_signal"].fillna(0).astype(int).values

        # Layer 4
        vol_r = df["vol_ratio"].fillna(0).values
        price_up = (df["Close"].values > df["Open"].values).astype(int)
        price_dn = (df["Close"].values < df["Open"].values).astype(int)
        sig_vol = np.where(
            (vol_r > 1.5) & (price_up == 1), 1,
            np.where((vol_r > 1.5) & (price_dn == 1), -1, 0),
        )
        sig_obv = df["obv_signal"].fillna(0).astype(int).values

        # Layer 5
        r1 = df["ret_1"].fillna(0).values
        r6 = df["ret_6"].fillna(0).values
        r42 = df["ret_42"].fillna(0).values
        sig_mom = np.where(
            (r1 > 0) & (r6 > 0) & (r42 > 0), 1,
            np.where((r1 < 0) & (r6 < 0) & (r42 < 0), -1, 0),
        )
        sig_ts = df["trend_structure"].fillna(0).astype(int).values

        # ── Composite score ───────────────────────────────
        w = self.WEIGHTS
        raw = (
            sig_ema * w["ema_structure"]
            + sig_hma * w["hma_direction"]
            + sig_st * w["supertrend"]
            + sig_rsi * w["rsi_divergence"]
            + sig_macd * w["macd_histogram"]
            + sig_z * w["zscore"]
            + sig_bb * w["bollinger"]
            + sig_kc * w["keltner"]
            + sig_vol * w["volume_confirm"]
            + sig_obv * w["obv_trend"]
            + sig_mom * w["price_momentum"]
            + sig_ts * w["trend_structure"]
        )

        # Apply extreme blocker
        raw[extreme_mask] = 0

        # ── Classify ──────────────────────────────────────
        scores = raw

        SL = self.STRONG_LONG_THRESHOLD
        LT = self.LONG_THRESHOLD
        ST = self.SHORT_THRESHOLD
        SS = self.STRONG_SHORT_THRESHOLD
        MX = self.MAX_SCORE

        types = np.select(
            [
                extreme_mask,
                scores >= SL,
                scores >= LT,
                scores <= SS,
                scores <= ST,
            ],
            ["SKIP", "STRONG_LONG", "LONG", "STRONG_SHORT", "SHORT"],
            default="SKIP",
        )

        confs = np.select(
            [
                extreme_mask,
                scores >= SL,
                (scores >= LT) & (scores < SL),
                scores <= SS,
                (scores <= ST) & (scores > SS),
            ],
            [
                0.0,
                np.minimum(scores / MX, 1.0),
                scores / SL,
                np.minimum(np.abs(scores) / MX, 1.0),
                np.abs(scores) / abs(SS),
            ],
            default=0.0,
        )

        # Skip warmup rows
        types[:start_idx] = "SKIP"
        confs[:start_idx] = 0.0
        scores[:start_idx] = 0

        # ── Write columns ────────────────────────────────
        df["signal_score"] = scores
        df["signal_type"] = types
        df["signal_confidence"] = np.round(confs, 4)
        df["signal_regime"] = regimes_arr

        elapsed = time.perf_counter() - t0
        non_skip = (types != "SKIP").sum()
        logger.info(
            f"  Batch complete in {elapsed:.2f}s — "
            f"{non_skip:,} actionable signals out of {n:,} bars"
        )

        return df

    # ══════════════════════════════════════════════════════════
    # REPORTING
    # ══════════════════════════════════════════════════════════

    def print_signal_report(self, df: pd.DataFrame) -> None:
        """Print a comprehensive signal distribution report."""
        C  = "\033[36m"
        G  = "\033[32m"
        Y  = "\033[33m"
        RED = "\033[31m"
        B  = "\033[1m"
        D  = "\033[2m"
        R  = "\033[0m"

        if "signal_type" not in df.columns:
            print(f"{RED}No signal columns found — run generate_signals_batch first.{R}")
            return

        total = len(df)
        dist = df["signal_type"].value_counts()

        print(f"\n{C}{'═' * 65}")
        print(f"  SIGNAL ENGINE REPORT")
        print(f"{'═' * 65}{R}")

        # ── Signal distribution ──────────────────────────
        print(f"\n{B}  Signal Distribution:{R}")
        print(f"  {'─' * 50}")

        colors = {
            "STRONG_LONG": G, "LONG": G,
            "SHORT": RED, "STRONG_SHORT": RED,
            "SKIP": D,
        }
        order = ["STRONG_LONG", "LONG", "SKIP", "SHORT", "STRONG_SHORT"]

        for sig_type in order:
            count = dist.get(sig_type, 0)
            pct = count / total * 100
            bar_len = int(pct / 2)
            color = colors.get(sig_type, D)
            bar = "█" * bar_len
            print(
                f"  {sig_type:<14s} │ {count:>7,} │ {pct:5.1f}% │ "
                f"{color}{bar}{R}"
            )

        actionable = total - dist.get("SKIP", 0)
        print(f"  {'─' * 50}")
        print(
            f"  {B}Actionable     │ {actionable:>7,} │ "
            f"{actionable/total*100:5.1f}%{R}"
        )

        # ── Score statistics ─────────────────────────────
        scores = df["signal_score"]
        print(f"\n{B}  Score Statistics:{R}")
        print(f"  {'─' * 50}")
        print(f"  Mean           : {scores.mean():+.2f}")
        print(f"  Std Dev        : {scores.std():.2f}")
        print(f"  Min / Max      : {scores.min():+d} / {scores.max():+d}")
        print(f"  Median         : {scores.median():+.0f}")

        # ── Score histogram (text) ────────────────────────
        print(f"\n{B}  Score Histogram:{R}")
        print(f"  {'─' * 50}")
        bins = range(self.MIN_SCORE, self.MAX_SCORE + 2)
        hist_counts = pd.cut(scores, bins=bins, right=False).value_counts().sort_index()

        max_count = hist_counts.max() if hist_counts.max() > 0 else 1
        for interval, count in hist_counts.items():
            score_val = int(interval.left)
            if count == 0:
                continue
            bar_len = int(count / max_count * 30)
            bar = "█" * max(bar_len, 1)

            if score_val >= self.STRONG_LONG_THRESHOLD:
                color = G
            elif score_val >= self.LONG_THRESHOLD:
                color = G
            elif score_val <= self.STRONG_SHORT_THRESHOLD:
                color = RED
            elif score_val <= self.SHORT_THRESHOLD:
                color = RED
            else:
                color = D

            print(f"  {score_val:+3d} │ {count:>6,} │ {color}{bar}{R}")

        # ── Regime distribution ──────────────────────────
        if "signal_regime" in df.columns:
            print(f"\n{B}  Volatility Regime:{R}")
            print(f"  {'─' * 50}")
            for regime, cnt in df["signal_regime"].value_counts().items():
                pct = cnt / total * 100
                print(f"  {regime:<10s} │ {cnt:>7,} ({pct:5.1f}%)")

        # ── Confidence stats ─────────────────────────────
        actionable_df = df[df["signal_type"] != "SKIP"]
        if len(actionable_df) > 0:
            conf = actionable_df["signal_confidence"]
            print(f"\n{B}  Confidence (actionable only):{R}")
            print(f"  {'─' * 50}")
            print(f"  Mean           : {conf.mean():.2%}")
            print(f"  Min / Max      : {conf.min():.2%} / {conf.max():.2%}")

        print(f"\n{C}{'═' * 65}{R}\n")

    def print_sample_signals(
        self,
        df: pd.DataFrame,
        n: int = 5,
        signal_type: Optional[str] = None,
    ) -> None:
        """Print N sample signals with full breakdown."""
        C = "\033[36m"
        G = "\033[32m"
        RED = "\033[31m"
        B = "\033[1m"
        D = "\033[2m"
        R = "\033[0m"

        if "signal_type" not in df.columns:
            print("No signal columns — run generate_signals_batch first.")
            return

        if signal_type:
            sample_df = df[df["signal_type"] == signal_type]
        else:
            sample_df = df[df["signal_type"] != "SKIP"]

        if len(sample_df) == 0:
            print(f"No signals of type '{signal_type}' found.")
            return

        # Pick N evenly spaced samples
        indices = np.linspace(0, len(sample_df) - 1, min(n, len(sample_df)), dtype=int)
        samples = sample_df.iloc[indices]

        print(f"\n{C}{'═' * 65}")
        print(f"  SAMPLE SIGNALS ({signal_type or 'ALL ACTIONABLE'})")
        print(f"{'═' * 65}{R}")

        for _, row in samples.iterrows():
            sig_type = row["signal_type"]
            color = G if "LONG" in sig_type else RED if "SHORT" in sig_type else D

            # Generate full signal for this bar
            pos = df.index.get_loc(row.name)
            full_sig = self.generate_trading_signal(df, pos)

            print(f"\n  {B}{row.name}{R}")
            print(f"  {color}{sig_type}{R}  score={row['signal_score']:+d}/{self.MAX_SCORE}  "
                  f"conf={row['signal_confidence']:.0%}  regime={row['signal_regime']}")
            print(f"  Close={row['Close']:,.2f}")

            if full_sig.get("direction"):
                d = full_sig
                print(
                    f"  Entry={d['entry_price']:,.2f}  "
                    f"SL={d['stop_loss']:,.2f}  "
                    f"TP1={d['tp1']:,.2f}  TP2={d['tp2']:,.2f}  TP3={d['tp3']:,.2f}  "
                    f"R:R={d['risk_reward']:.1f}"
                )

            if full_sig.get("reasons"):
                for reason in full_sig["reasons"]:
                    print(f"    {reason}")

            print(f"  {'─' * 55}")

        print(f"\n{C}{'═' * 65}{R}\n")


# ══════════════════════════════════════════════════════════════
# SIGNAL QUALITY ENGINE
# ══════════════════════════════════════════════════════════════

class SignalQualityEngine:
    """
    High-selectivity filter layer on top of SignalEngine.

    Applies 4 confirmation layers to produce a 0-100 quality score:
      Base score    (max 40) — raw signal strength
      ML confidence (max 20) — model probability
      Volume        (max 15) — vol_ratio thresholds
      Trend align   (max 15) — 4H direction vs 1D trend proxy
      Regime bonus  (max 10) — regime / direction alignment

    Trade decision gate:
      quality >= 70 → TAKE
      quality 50-69 → WAIT  (skip this bar, re-evaluate next)
      quality <  50 → SKIP
    """

    MIN_SCORE_LONG  =  10   # raised from 7/8
    MIN_SCORE_SHORT = -10

    QUALITY_TAKE = 70
    QUALITY_WAIT = 50

    # ── Core quality formula ───────────────────────────────────────────────

    @staticmethod
    def calculate_signal_quality(
        signal_score: float,
        ml_confidence: float,
        volume_ratio: float,
        trend_alignment: str,
        regime: str,
        signal_direction: str,
    ) -> float:
        """
        Return a 0-100 quality score for a single signal.

        Parameters
        ----------
        signal_score     : raw composite score (-19..+19)
        ml_confidence    : ML ensemble confidence (0.0–1.0)
        volume_ratio     : current vol / 20-bar average vol
        trend_alignment  : "ALIGNED" | "NEUTRAL" | "AGAINST"
        regime           : "BULL" | "BEAR" | "SIDEWAYS" | other
        signal_direction : "LONG" | "SHORT"
        """
        quality = 0.0

        # Base score (max 40)
        quality += min(abs(signal_score), 15) * 2.5

        # ML confidence (max 20)
        quality += ml_confidence * 20

        # Volume confirmation (max 15)
        if volume_ratio > 2.0:
            quality += 15
        elif volume_ratio > 1.5:
            quality += 10
        elif volume_ratio > 1.2:
            quality += 5

        # Trend alignment (max 15)
        if trend_alignment == "ALIGNED":
            quality += 15
        elif trend_alignment == "NEUTRAL":
            quality += 5

        # Regime bonus (max 10)
        if regime == "BULL" and signal_direction == "LONG":
            quality += 10
        elif regime == "BEAR" and signal_direction == "SHORT":
            quality += 10
        elif regime == "SIDEWAYS":
            quality += 5

        return float(quality)

    @staticmethod
    def get_trade_decision(quality: float) -> str:
        """Map quality score to TAKE / WAIT / SKIP."""
        if quality >= 70:
            return "TAKE"
        if quality >= 50:
            return "WAIT"
        return "SKIP"

    # ── Batch computation ──────────────────────────────────────────────────

    def generate_quality_signals_batch(
        self,
        df: pd.DataFrame,
        ml_confidence_col: Optional[str] = None,
        regime_col: Optional[str] = None,
        default_ml_confidence: float = 0.70,
    ) -> pd.DataFrame:
        """
        Compute quality scores for every bar and append three columns:
          quality_score     float 0-100
          quality_decision  "TAKE" | "WAIT" | "SKIP"
          trend_alignment   "ALIGNED" | "NEUTRAL" | "AGAINST"

        Must be called after ``SignalEngine.generate_signals_batch()``.
        """
        if "signal_score" not in df.columns:
            raise RuntimeError(
                "Run SignalEngine.generate_signals_batch() before quality scoring."
            )

        n = len(df)
        scores_arr = df["signal_score"].values.astype(float)

        # ── Signal direction ──────────────────────────────────────────────
        dir_arr = np.where(scores_arr > 0, "LONG",
                  np.where(scores_arr < 0, "SHORT", "FLAT"))

        # ── Volume ratio ─────────────────────────────────────────────────
        if "vol_ratio" in df.columns:
            vol_ratio = df["vol_ratio"].fillna(1.0).values.astype(float)
        else:
            vol_ratio = np.ones(n)

        # ── 1D trend alignment (proxy: ret_6 = 24 h, ret_42 = 7 d) ──────
        close = df["Close"].values.astype(float)
        if "ret_6" in df.columns:
            ret_1d = df["ret_6"].fillna(0).values.astype(float)
        else:
            ret_1d = np.concatenate(
                [np.zeros(6), (close[6:] / close[:-6] - 1) * 100]
            )

        if "ret_42" in df.columns:
            ret_7d = df["ret_42"].fillna(0).values.astype(float)
        else:
            ret_7d = np.concatenate(
                [np.zeros(42), (close[42:] / close[:-42] - 1) * 100]
            )

        trend_align = np.full(n, "NEUTRAL", dtype=object)
        long_m  = scores_arr > 0
        trend_align[long_m & (ret_1d > 0) & (ret_7d > 0)] = "ALIGNED"
        trend_align[long_m & (ret_1d < 0) & (ret_7d < 0)] = "AGAINST"
        short_m = scores_arr < 0
        trend_align[short_m & (ret_1d < 0) & (ret_7d < 0)] = "ALIGNED"
        trend_align[short_m & (ret_1d > 0) & (ret_7d > 0)] = "AGAINST"

        # ── ML confidence ────────────────────────────────────────────────
        if ml_confidence_col and ml_confidence_col in df.columns:
            ml_conf = df[ml_confidence_col].fillna(default_ml_confidence).values.astype(float)
        else:
            ml_conf = np.full(n, default_ml_confidence)

        # ── HMM regime (or ATR-based fallback) ───────────────────────────
        if regime_col and regime_col in df.columns:
            regime_arr = df[regime_col].fillna("SIDEWAYS").values.astype(str)
        else:
            if "vol_regime" in df.columns:
                vr = df["vol_regime"].fillna(1).astype(int).values
                regime_arr = np.select(
                    [vr == 3],
                    ["HIGH_VOL"],
                    default="SIDEWAYS",
                ).astype(str)
            else:
                regime_arr = np.full(n, "SIDEWAYS", dtype=object)

        # ── ATR extreme blocker ──────────────────────────────────────────
        if "vol_regime" in df.columns:
            atr_extreme = df["vol_regime"].fillna(1).astype(int).values == 3
        else:
            atr_extreme = np.zeros(n, dtype=bool)

        sig_type_arr = (
            df["signal_type"].values
            if "signal_type" in df.columns
            else np.full(n, "SKIP", dtype=object)
        )

        # ── Vectorised quality components ────────────────────────────────
        abs_scores = np.abs(scores_arr)

        q_base = np.minimum(abs_scores, 15) * 2.5                          # max 40

        q_ml = ml_conf * 20                                                  # max 20

        q_vol = np.select(                                                   # max 15
            [vol_ratio > 2.0, vol_ratio > 1.5, vol_ratio > 1.2],
            [15.0, 10.0, 5.0],
            default=0.0,
        )

        q_trend = np.select(                                                 # max 15
            [trend_align == "ALIGNED", trend_align == "NEUTRAL"],
            [15.0, 5.0],
            default=0.0,
        )

        q_regime = np.select(                                                # max 10
            [
                (regime_arr == "BULL")    & (dir_arr == "LONG"),
                (regime_arr == "BEAR")    & (dir_arr == "SHORT"),
                (regime_arr == "SIDEWAYS"),
            ],
            [10.0, 10.0, 5.0],
            default=0.0,
        )

        quality_scores = q_base + q_ml + q_vol + q_trend + q_regime

        # ── Hard blockers ────────────────────────────────────────────────
        block_mask = (
            atr_extreme
            | (sig_type_arr == "SKIP")
            | (abs_scores < abs(self.MIN_SCORE_LONG))
            | (trend_align == "AGAINST")
        )
        quality_scores[block_mask] = 0.0

        decisions = np.select(
            [block_mask,
             quality_scores >= self.QUALITY_TAKE,
             quality_scores >= self.QUALITY_WAIT],
            ["SKIP", "TAKE", "WAIT"],
            default="SKIP",
        )

        df["quality_score"]    = np.round(quality_scores, 1)
        df["quality_decision"] = decisions
        df["trend_alignment"]  = trend_align

        take_count = (decisions == "TAKE").sum()
        wait_count = (decisions == "WAIT").sum()
        logger.info(
            "Quality filter: %d TAKE, %d WAIT out of %d bars",
            take_count, wait_count, n,
        )
        return df

    # ── Comparison report ─────────────────────────────────────────────────

    @staticmethod
    def print_quality_report(df: pd.DataFrame, period_label: str = "") -> None:
        """Print old-vs-new signal count comparison + quality distribution."""
        C   = "\033[36m"
        G   = "\033[32m"
        Y   = "\033[33m"
        RED = "\033[31m"
        B   = "\033[1m"
        D   = "\033[2m"
        R   = "\033[0m"

        header = f"  SIGNAL QUALITY REPORT{' — ' + period_label if period_label else ''}"
        print(f"\n{C}{'═' * 65}")
        print(header)
        print(f"{'═' * 65}{R}")

        total = len(df)

        # ── Old system ───────────────────────────────────────────────────
        if "signal_type" in df.columns:
            old_dist = df["signal_type"].value_counts()
            old_actionable = total - old_dist.get("SKIP", 0)
            old_strong = (
                old_dist.get("STRONG_LONG", 0) + old_dist.get("STRONG_SHORT", 0)
            )
            print(f"\n{B}  OLD System (score threshold ≥ 4):{R}")
            print(f"  {'─' * 50}")
            for t in ["STRONG_LONG", "LONG", "SHORT", "STRONG_SHORT"]:
                cnt = old_dist.get(t, 0)
                pct = cnt / total * 100
                color = G if "LONG" in t else RED
                print(f"  {color}{t:<14s}{R} │ {cnt:>7,} ({pct:5.1f}%)")
            print(f"  {'─' * 50}")
            print(f"  {B}Actionable     │ {old_actionable:>7,} ({old_actionable/total*100:.1f}%)  [score ≥ 4]{R}")
            print(f"  {B}Strong only    │ {old_strong:>7,} ({old_strong/total*100:.1f}%)  [score ≥ 8]{R}")
        else:
            old_actionable = 0

        # ── New system ───────────────────────────────────────────────────
        if "quality_decision" in df.columns:
            new_dist   = df["quality_decision"].value_counts()
            take_count = new_dist.get("TAKE", 0)
            wait_count = new_dist.get("WAIT", 0)
            skip_count = new_dist.get("SKIP", 0)

            print(f"\n{B}  NEW System (quality ≥ 70, score ≥ 10, MTF aligned):{R}")
            print(f"  {'─' * 50}")
            print(f"  {G}TAKE           │ {take_count:>7,} ({take_count/total*100:5.1f}%){R}")
            print(f"  {Y}WAIT           │ {wait_count:>7,} ({wait_count/total*100:5.1f}%){R}")
            print(f"  {D}SKIP           │ {skip_count:>7,} ({skip_count/total*100:5.1f}%){R}")

            if old_actionable > 0:
                reduction = (1 - take_count / old_actionable) * 100
                print(f"\n  Signal reduction : {RED}{reduction:.1f}%{R} fewer entries")
                print(f"  Expected outcome : higher win-rate due to quality gate")

        # ── Quality score distribution ────────────────────────────────────
        if "quality_score" in df.columns:
            qs = df.loc[df["quality_score"] > 0, "quality_score"]
            if len(qs) > 0:
                print(f"\n{B}  Quality Score Distribution (signals with score > 0):{R}")
                print(f"  {'─' * 50}")
                for bucket, label, color in [
                    ((70, 101), "TAKE  (≥70)",  G),
                    ((50,  70), "WAIT  (50-69)", Y),
                    (( 0,  50), "SKIP  (<50)",   D),
                ]:
                    cnt = int(((qs >= bucket[0]) & (qs < bucket[1])).sum())
                    print(f"  {color}{label:<14s}{R} │ {cnt:>7,} ({cnt/len(qs)*100:5.1f}%)")
                print(f"  {'─' * 50}")
                print(f"  Mean   : {qs.mean():.1f}   Median : {qs.median():.1f}   Max : {qs.max():.1f}")

        # ── Trend alignment ───────────────────────────────────────────────
        if "trend_alignment" in df.columns:
            ta = df["trend_alignment"].value_counts()
            print(f"\n{B}  Trend Alignment (all bars):{R}")
            print(f"  {'─' * 50}")
            for label in ["ALIGNED", "NEUTRAL", "AGAINST"]:
                cnt = ta.get(label, 0)
                print(f"  {label:<10s} │ {cnt:>7,} ({cnt/total*100:5.1f}%)")

        print(f"\n{C}{'═' * 65}{R}\n")
