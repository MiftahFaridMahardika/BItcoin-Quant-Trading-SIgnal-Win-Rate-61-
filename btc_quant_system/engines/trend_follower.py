"""
Trend-Following Mode
====================
Kapitalisasi bull market dengan tiga komponen utama:

1. MarketBiasDetector  — deteksi STRONG_BULL / BULL / NEUTRAL / BEAR
2. get_dynamic_thresholds() — adjust scoring thresholds & position sizing
3. TrendAwareSignalEngine   — wraps SignalEngine, injects bias-adjusted classification
4. TrendFollowerSLTP        — widened TP3 + adapted multipliers per bias
5. TrendFollowerBacktest     — OLD vs NEW backtest comparison per year
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# MARKET BIAS DETECTOR
# ══════════════════════════════════════════════════════════════

@dataclass
class BiasResult:
    """Result of market bias detection."""
    bias: str           # STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR
    score: int          # 0..6
    details: Dict[str, bool]


class MarketBiasDetector:
    """
    Detects macro market bias for adaptive trend-following.

    Score breakdown (max 6):
      +1  Price > EMA50
      +1  Price > EMA200
      +1  EMA50 > EMA200  (golden cross territory)
      +2  Higher Highs AND Higher Lows (last 20 bars)
      +1  30-bar ROC > +10%

    Bias:
      score 5-6  → STRONG_BULL
      score 3-4  → BULL
      score 2    → NEUTRAL
      score 1    → BEAR
      score 0    → STRONG_BEAR
    """

    def __init__(self, lookback: int = 200):
        self.lookback = lookback

    def detect_bias(self, df: pd.DataFrame, idx: int = -1) -> BiasResult:
        """
        Detect market bias at a specific bar.

        Parameters
        ----------
        df  : pd.DataFrame with Close, High, Low columns
        idx : bar index to evaluate (-1 = last bar)
        """
        close_col = "Close" if "Close" in df.columns else "close"
        high_col  = "High"  if "High"  in df.columns else "high"
        low_col   = "Low"   if "Low"   in df.columns else "low"

        close = df[close_col]
        high  = df[high_col]
        low   = df[low_col]

        if idx == -1:
            idx = len(df) - 1

        # Need at least 200 bars for EMA200
        if idx < 200:
            return BiasResult("NEUTRAL", 3, {})

        # ── 1. EMA50 / EMA200 ─────────────────────────────
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        price_now        = close.iloc[idx]
        above_ema50      = price_now > ema50.iloc[idx]
        above_ema200     = price_now > ema200.iloc[idx]
        ema50_above_ema200 = ema50.iloc[idx] > ema200.iloc[idx]

        # ── 2. Higher Highs / Higher Lows (20-bar window) ─
        window = min(20, idx)
        if idx >= window:
            recent_high_now  = high.iloc[max(0, idx - window):idx + 1].max()
            recent_high_then = high.iloc[max(0, idx - window * 2):idx - window + 1].max()
            recent_low_now   = low.iloc[max(0, idx - window):idx + 1].min()
            recent_low_then  = low.iloc[max(0, idx - window * 2):idx - window + 1].min()
            hh = recent_high_now > recent_high_then
            hl = recent_low_now  > recent_low_then
        else:
            hh = hl = False

        # ── 3. 30-bar ROC ─────────────────────────────────
        roc_bars = min(30, idx)
        roc30 = (price_now - close.iloc[idx - roc_bars]) / close.iloc[idx - roc_bars] if idx >= roc_bars else 0.0
        strong_momentum = roc30 > 0.10   # +10% in 30 bars

        # ── Score ─────────────────────────────────────────
        bull_score = (
            int(above_ema50)
            + int(above_ema200)
            + int(ema50_above_ema200)
            + (2 if (hh and hl) else 0)
            + int(strong_momentum)
        )

        if bull_score >= 5:
            bias = "STRONG_BULL"
        elif bull_score >= 3:
            bias = "BULL"
        elif bull_score <= 0:
            bias = "STRONG_BEAR"
        elif bull_score == 1:
            bias = "BEAR"
        else:
            bias = "NEUTRAL"

        return BiasResult(
            bias=bias,
            score=bull_score,
            details={
                "price_above_ema50":      above_ema50,
                "price_above_ema200":     above_ema200,
                "ema50_above_ema200":     ema50_above_ema200,
                "higher_highs":           hh,
                "higher_lows":            hl,
                "strong_momentum_30bar":  strong_momentum,
                "roc30_pct":              round(roc30 * 100, 2),
            },
        )

    def detect_bias_series(self, df: pd.DataFrame, recalc_every: int = 24) -> pd.Series:
        """
        Compute bias for every bar, recalculating every N bars (default 24h at 4H).
        Returns a pd.Series of bias strings aligned to df.index.
        """
        biases = pd.Series("NEUTRAL", index=df.index, dtype=str)

        cached_bias = "NEUTRAL"
        for pos in range(200, len(df)):
            if pos % recalc_every == 0:
                result = self.detect_bias(df, idx=pos)
                cached_bias = result.bias
            biases.iloc[pos] = cached_bias

        return biases


# ══════════════════════════════════════════════════════════════
# DYNAMIC THRESHOLDS
# ══════════════════════════════════════════════════════════════

def get_dynamic_thresholds(market_bias: str) -> Dict:
    """
    Adjust signal classification thresholds and position sizing
    based on macro market bias.

    In STRONG_BULL:
      - Lower LONG threshold (easier to go long)
      - Raise SHORT threshold (much harder to go short)
      - Bigger LONG positions, smaller SHORT positions

    Returns dict with keys:
      long_threshold        : int  (signal score needed for LONG)
      strong_long_threshold : int  (signal score needed for STRONG_LONG)
      short_threshold       : int  (signal score needed for SHORT, negative)
      strong_short_threshold: int  (signal score needed for STRONG_SHORT, negative)
      position_mult_long    : float (multiply position size for longs)
      position_mult_short   : float (multiply position size for shorts)
      tp3_atr_mult          : float (TP3 ATR multiplier — wider in bull)
      max_trade_bars        : int  (max hold bars — longer in bull)
    """
    configs = {
        "STRONG_BULL": {
            # Lower bar for LONG, very high bar for SHORT
            "long_threshold":          3,    # was 4 — easier LONG entry
            "strong_long_threshold":   7,    # was 8
            "short_threshold":        -12,   # was -4 — much harder SHORT
            "strong_short_threshold": -15,   # was -8 — practically blocked
            "position_mult_long":      1.3,  # +30% size on longs
            "position_mult_short":     0.5,  # -50% size on shorts (risk control)
            "tp3_atr_mult":            8.0,  # wider TP3 to ride the trend
            "max_trade_bars":          250,  # hold longer (~41 days at 4H)
        },
        "BULL": {
            "long_threshold":          4,
            "strong_long_threshold":   8,
            "short_threshold":        -10,
            "strong_short_threshold": -12,
            "position_mult_long":      1.1,
            "position_mult_short":     0.7,
            "tp3_atr_mult":            6.0,
            "max_trade_bars":          200,
        },
        "NEUTRAL": {
            "long_threshold":          4,
            "strong_long_threshold":   8,
            "short_threshold":         -4,
            "strong_short_threshold":  -8,
            "position_mult_long":      1.0,
            "position_mult_short":     1.0,
            "tp3_atr_mult":            5.0,
            "max_trade_bars":          168,
        },
        "BEAR": {
            "long_threshold":          10,   # harder LONG
            "strong_long_threshold":   13,
            "short_threshold":         -4,   # easier SHORT
            "strong_short_threshold":  -7,
            "position_mult_long":      0.6,
            "position_mult_short":     1.2,
            "tp3_atr_mult":            5.0,
            "max_trade_bars":          168,
        },
        "STRONG_BEAR": {
            "long_threshold":          14,   # almost blocked
            "strong_long_threshold":   17,
            "short_threshold":         -3,
            "strong_short_threshold":  -7,
            "position_mult_long":      0.4,
            "position_mult_short":     1.3,
            "tp3_atr_mult":            5.0,
            "max_trade_bars":          168,
        },
    }
    return configs.get(market_bias, configs["NEUTRAL"])


# ══════════════════════════════════════════════════════════════
# TREND-AWARE SIGNAL ENGINE (WRAPPER)
# ══════════════════════════════════════════════════════════════

class TrendAwareSignalEngine:
    """
    Wraps SignalEngine and applies market-bias-adjusted classification.

    The bias is refreshed every `bias_recalc_bars` bars (default 24 = 4H × 24 = daily).

    Usage:
        from engines.signal_engine import SignalEngine
        from engines.trend_follower import TrendAwareSignalEngine

        base_engine = SignalEngine()
        trend_engine = TrendAwareSignalEngine(base_engine)
        # Then use trend_engine.generate_trading_signal(df, idx)
    """

    def __init__(self, signal_engine, bias_recalc_bars: int = 24):
        self.signal_engine = signal_engine
        self.bias_detector = MarketBiasDetector()
        self.bias_recalc_bars = bias_recalc_bars

        # Cache
        self._cached_bias: str = "NEUTRAL"
        self._cached_thresholds: Dict = get_dynamic_thresholds("NEUTRAL")
        self._last_bias_idx: int = -1

    def _refresh_bias(self, df: pd.DataFrame, idx: int) -> None:
        """Refresh bias and thresholds if needed."""
        if (idx - self._last_bias_idx) >= self.bias_recalc_bars or self._last_bias_idx < 0:
            result = self.bias_detector.detect_bias(df, idx=idx)
            self._cached_bias = result.bias
            self._cached_thresholds = get_dynamic_thresholds(result.bias)
            self._last_bias_idx = idx

    def get_current_bias(self) -> str:
        return self._cached_bias

    def get_current_thresholds(self) -> Dict:
        return self._cached_thresholds

    def _classify_with_bias(self, score: int, thresholds: Dict) -> Tuple[str, float]:
        """
        Classify signal score using bias-adjusted thresholds.
        """
        max_score = 19  # SignalEngine.MAX_SCORE
        sl  = thresholds["strong_long_threshold"]
        lt  = thresholds["long_threshold"]
        st  = thresholds["short_threshold"]
        ss  = thresholds["strong_short_threshold"]

        if score >= sl:
            return "STRONG_LONG", min(score / max_score, 1.0)
        if score >= lt:
            return "LONG", score / sl if sl > 0 else 0.5
        if score <= ss:
            return "STRONG_SHORT", min(abs(score) / max_score, 1.0)
        if score <= st:
            return "SHORT", abs(score) / abs(ss) if ss != 0 else 0.5
        return "SKIP", 0.0

    def generate_trading_signal(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        Generate trading signal with trend-bias-adjusted thresholds.

        Returns same dict as SignalEngine.generate_trading_signal() but with:
          - bias-adjusted direction classification
          - wider TP3 in bull bias
          - position_mult key for execution engine
          - market_bias key
        """
        import pandas as _pd

        # Refresh bias periodically
        self._refresh_bias(df, idx)
        thresholds = self._cached_thresholds
        bias = self._cached_bias

        # Get raw score from base signal engine
        result = self.signal_engine.calculate_signal_score(df, idx)

        if result.signal == "SKIP":
            return {
                "signal": "SKIP",
                "score": result.score,
                "regime": result.regime,
                "timestamp": result.timestamp,
                "reasons": result.reasons,
                "market_bias": bias,
            }

        # Bias-adjusted classification
        signal_type, confidence = self._classify_with_bias(result.score, thresholds)

        if signal_type == "SKIP":
            return {
                "signal": "SKIP",
                "score": result.score,
                "regime": result.regime,
                "timestamp": result.timestamp,
                "reasons": result.reasons + [f"[TrendFilter] Bias={bias} filtered (score={result.score})"],
                "market_bias": bias,
            }

        row = df.iloc[idx]
        price = float(row["Close"])
        atr   = float(row.get("atr_14", 0))
        if _pd.isna(atr) or atr <= 0:
            atr = price * 0.02

        direction = "LONG" if result.score > 0 else "SHORT"

        # Position multiplier from bias
        pos_mult = (thresholds["position_mult_long"]
                    if direction == "LONG"
                    else thresholds["position_mult_short"])

        # Adaptive SL/TP: wider TP3 in bull
        sl_mult  = 1.95 if bias in ("STRONG_BULL", "BULL") else 1.5
        tp1_mult = 1.65 if bias in ("STRONG_BULL", "BULL") else 2.0
        tp2_mult = 2.75 if bias in ("STRONG_BULL", "BULL") else 3.5
        tp3_mult = thresholds["tp3_atr_mult"]   # 8× in STRONG_BULL, 5× neutral

        sign = 1 if direction == "LONG" else -1
        sl  = price - sign * atr * sl_mult
        tp1 = price + sign * atr * tp1_mult
        tp2 = price + sign * atr * tp2_mult
        tp3 = price + sign * atr * tp3_mult

        risk = abs(price - sl)
        rr   = abs(tp2 - price) / risk if risk > 0 else 0.0

        return {
            "signal":         signal_type,
            "direction":      direction,
            "entry_price":    round(price, 2),
            "stop_loss":      round(sl, 2),
            "tp1":            round(tp1, 2),
            "tp2":            round(tp2, 2),
            "tp3":            round(tp3, 2),
            "risk_reward":    round(rr, 2),
            "confidence":     round(confidence, 4),
            "score":          result.score,
            "regime":         result.regime,
            "reasons":        result.reasons,
            "timestamp":      result.timestamp,
            "individual":     result.individual_signals,
            "market_bias":    bias,
            "position_mult":  round(pos_mult, 2),
            "tp3_atr_mult":   tp3_mult,
        }


# ══════════════════════════════════════════════════════════════
# TREND-FOLLOWING BACKTEST SIMULATION
# Simulates OLD (neutral) vs NEW (trend-aware) per bull-market years
# ══════════════════════════════════════════════════════════════

MAX_BARS = 250   # ~41 days at 4H (extended for trend mode)


def _simulate_trade(
    entry_price: float,
    direction: str,
    atr: float,
    sl_mult: float,
    tp1_mult: float,
    tp2_mult: float,
    tp3_mult: float,
    high_arr: np.ndarray,
    low_arr:  np.ndarray,
    close_arr: np.ndarray,
    atr_arr:  np.ndarray,
    pos_mult: float = 1.0,   # position size multiplier
) -> Dict:
    """
    Simulate one trade with tiered partial exits.

    Returns dict with: exit, pnl_r (adjusted by pos_mult), bars, tp1, tp2, bias_mult
    """
    sign    = 1 if direction == "LONG" else -1
    sl      = entry_price - sign * atr * sl_mult
    tp1     = entry_price + sign * atr * tp1_mult
    tp2     = entry_price + sign * atr * tp2_mult
    tp3     = entry_price + sign * atr * tp3_mult
    sl_dist = abs(entry_price - sl)

    cur_sl   = sl
    tp1_hit  = False
    tp2_hit  = False
    breakeven = False
    total_r  = 0.0
    n        = len(high_arr)

    for i in range(n):
        h = high_arr[i]
        l = low_arr[i]
        c = close_arr[i]
        bar_atr = float(atr_arr[i]) if (i < len(atr_arr) and not np.isnan(atr_arr[i]) and atr_arr[i] > 0) else atr

        if direction == "LONG":
            if not tp1_hit and h >= tp1:
                total_r += (tp1 - entry_price) / sl_dist * 0.40
                tp1_hit  = True
                cur_sl   = max(cur_sl, entry_price)
                breakeven = True
                continue

            if tp1_hit and not tp2_hit and h >= tp2:
                total_r += (tp2 - entry_price) / sl_dist * 0.30
                tp2_hit  = True
                new_sl   = max(c - bar_atr, entry_price)
                if new_sl > cur_sl:
                    cur_sl = new_sl
                continue

            if tp2_hit and h >= tp3:
                total_r += (tp3 - entry_price) / sl_dist * 0.30
                return {"exit": "TP3", "pnl_r": total_r * pos_mult, "bars": i,
                        "tp1": tp1_hit, "tp2": tp2_hit}

            if l <= cur_sl:
                remain   = 1.0 - (0.40 if tp1_hit else 0) - (0.30 if tp2_hit else 0)
                total_r += (cur_sl - entry_price) / sl_dist * remain
                reason   = "TRAIL_STOP" if breakeven else "STOP_LOSS"
                return {"exit": reason, "pnl_r": total_r * pos_mult, "bars": i,
                        "tp1": tp1_hit, "tp2": tp2_hit}

            # Tiered trailing — only after 0.5× ATR profit
            profit = c - entry_price
            if profit >= 3 * bar_atr:
                new_sl = c - 0.7 * bar_atr
            elif profit >= 2 * bar_atr:
                new_sl = c - 1.0 * bar_atr
            elif profit >= bar_atr:
                new_sl = max(entry_price, c - 1.5 * bar_atr)
                if not breakeven:
                    breakeven = True
            elif profit >= 0.5 * bar_atr:
                new_sl = entry_price
                if not breakeven:
                    breakeven = True
            else:
                continue
            if new_sl > cur_sl:
                cur_sl = new_sl

        else:   # SHORT (symmetric)
            if not tp1_hit and l <= tp1:
                total_r += (entry_price - tp1) / sl_dist * 0.40
                tp1_hit  = True
                cur_sl   = min(cur_sl, entry_price)
                breakeven = True
                continue

            if tp1_hit and not tp2_hit and l <= tp2:
                total_r += (entry_price - tp2) / sl_dist * 0.30
                tp2_hit  = True
                new_sl   = min(c + bar_atr, entry_price)
                if new_sl < cur_sl:
                    cur_sl = new_sl
                continue

            if tp2_hit and l <= tp3:
                total_r += (entry_price - tp3) / sl_dist * 0.30
                return {"exit": "TP3", "pnl_r": total_r * pos_mult, "bars": i,
                        "tp1": tp1_hit, "tp2": tp2_hit}

            if h >= cur_sl:
                remain   = 1.0 - (0.40 if tp1_hit else 0) - (0.30 if tp2_hit else 0)
                total_r += (entry_price - cur_sl) / sl_dist * remain
                reason   = "TRAIL_STOP" if breakeven else "STOP_LOSS"
                return {"exit": reason, "pnl_r": total_r * pos_mult, "bars": i,
                        "tp1": tp1_hit, "tp2": tp2_hit}

            profit = entry_price - c
            if profit >= 3 * bar_atr:
                new_sl = c + 0.7 * bar_atr
            elif profit >= 2 * bar_atr:
                new_sl = c + 1.0 * bar_atr
            elif profit >= bar_atr:
                new_sl = min(entry_price, c + 1.5 * bar_atr)
                if not breakeven:
                    breakeven = True
            elif profit >= 0.5 * bar_atr:
                new_sl = entry_price
                if not breakeven:
                    breakeven = True
            else:
                continue
            if new_sl < cur_sl:
                cur_sl = new_sl

    # Time exit
    remain  = 1.0 - (0.40 if tp1_hit else 0) - (0.30 if tp2_hit else 0)
    last_c  = close_arr[-1]
    pnl_r = (last_c - entry_price) / sl_dist * remain * sign
    return {"exit": "TIME_EXIT", "pnl_r": pnl_r * pos_mult, "bars": n,
            "tp1": tp1_hit, "tp2": tp2_hit}


def run_trend_follower_comparison(
    df: pd.DataFrame,
    signal_engine,
    years: Optional[List[int]] = None,
    bias_recalc_bars: int = 24,
) -> Dict:
    """
    Run OLD (neutral thresholds, fixed TP) vs NEW (trend-bias-adjusted)
    backtest comparison.

    Parameters
    ----------
    df              : featured DataFrame with signal columns already computed
    signal_engine   : SignalEngine instance
    years           : list of years to include (None = all)
    bias_recalc_bars: how often to recalculate bias

    Returns
    -------
    dict: {
      'old': list of trade results,
      'new': list of trade results,
      'bias_series': pd.Series of bias per bar,
    }
    """
    bias_detector  = MarketBiasDetector()
    trend_engine   = TrendAwareSignalEngine(signal_engine, bias_recalc_bars)

    close_arr = df["Close"].values
    high_arr  = df["High"].values
    low_arr   = df["Low"].values
    atr_arr   = df.get("atr_14", df["Close"] * 0.02).fillna(df["Close"] * 0.02).values
    n_total   = len(df)

    old_results: List[Dict] = []
    new_results: List[Dict] = []
    meta: List[Dict] = []

    # Pre-compute bias series for reporting
    bias_series = bias_detector.detect_bias_series(df, recalc_every=bias_recalc_bars)

    # Iterate over actionable signals
    actionable_mask = df["signal_type"].isin(
        ["STRONG_LONG", "LONG", "SHORT", "STRONG_SHORT"]
    )
    signal_df = df[actionable_mask].copy()

    if years:
        signal_df = signal_df[signal_df.index.year.isin(years)]

    for ts, row in signal_df.iterrows():
        idx = df.index.get_loc(ts)
        if idx < 200:
            continue

        direction = str(row["signal_type"]).replace("STRONG_", "")
        if direction not in ("LONG", "SHORT"):
            continue

        entry_price = float(row["Close"])
        atr         = float(atr_arr[idx])
        if np.isnan(atr) or atr <= 0:
            atr = entry_price * 0.02

        end_idx   = min(idx + MAX_BARS + 1, n_total)
        fwd_high  = high_arr[idx + 1: end_idx]
        fwd_low   = low_arr[idx + 1: end_idx]
        fwd_close = close_arr[idx + 1: end_idx]
        fwd_atr   = atr_arr[idx + 1: end_idx]

        if len(fwd_high) == 0:
            continue

        # ── OLD (neutral, fixed thresholds, fixed 1.5/2.0/3.5/5.0) ──
        old_r = _simulate_trade(
            entry_price, direction, atr,
            sl_mult=1.5, tp1_mult=2.0, tp2_mult=3.5, tp3_mult=5.0,
            high_arr=fwd_high, low_arr=fwd_low, close_arr=fwd_close, atr_arr=fwd_atr,
            pos_mult=1.0,
        )
        old_results.append({**old_r, "direction": direction, "year": ts.year})

        # ── NEW (trend-bias, adaptive thresholds) ──
        # Get bias at this bar
        bias_result = bias_detector.detect_bias(df, idx=idx)
        bias        = bias_result.bias
        thresholds  = get_dynamic_thresholds(bias)

        # Check new direction with bias-adjusted thresholds
        score     = row["signal_score"]
        sig_type, confidence = trend_engine._classify_with_bias(score, thresholds)

        # If filtered out by bias, record as SKIP
        if sig_type == "SKIP":
            new_results.append({
                "exit": "BIAS_FILTERED", "pnl_r": 0.0, "bars": 0,
                "tp1": False, "tp2": False,
                "direction": direction, "year": ts.year,
                "bias": bias, "score": score,
            })
            meta.append({"ts": ts, "direction": direction, "bias": bias,
                         "new_dir": "FILTERED", "score": score})
            continue

        new_direction = "LONG" if score > 0 else "SHORT"
        pos_mult  = (thresholds["position_mult_long"]
                     if new_direction == "LONG"
                     else thresholds["position_mult_short"])
        sl_m  = 1.95 if bias in ("STRONG_BULL", "BULL") else 1.5
        tp1_m = 1.65 if bias in ("STRONG_BULL", "BULL") else 2.0
        tp2_m = 2.75 if bias in ("STRONG_BULL", "BULL") else 3.5
        tp3_m = thresholds["tp3_atr_mult"]

        new_r = _simulate_trade(
            entry_price, new_direction, atr,
            sl_mult=sl_m, tp1_mult=tp1_m, tp2_mult=tp2_m, tp3_mult=tp3_m,
            high_arr=fwd_high, low_arr=fwd_low, close_arr=fwd_close, atr_arr=fwd_atr,
            pos_mult=pos_mult,
        )
        new_results.append({**new_r, "direction": new_direction,
                            "year": ts.year, "bias": bias,
                            "pos_mult": pos_mult, "score": score})
        meta.append({"ts": ts, "direction": direction, "bias": bias,
                     "new_dir": new_direction, "score": score})

    return {
        "old": old_results,
        "new": new_results,
        "meta": meta,
        "bias_series": bias_series,
    }
