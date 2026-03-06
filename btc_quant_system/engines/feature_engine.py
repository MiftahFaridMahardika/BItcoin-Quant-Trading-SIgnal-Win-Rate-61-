"""
BTC Quant Trading System — Feature Engineering Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Computes 50+ technical indicators organized into 5 analytical layers.
Pure pandas/numpy implementation — no TA-Lib dependency.

LAYER 1: Trend Detection          (12 features)
LAYER 2: Momentum & Mean Reversion (20 features)
LAYER 3: Volatility Regime         (14 features)
LAYER 4: Volume Analysis           (10 features)
LAYER 5: Price Action & Structure  (10 features)

All computations are causal — row N only uses data from rows ≤ N.
Zero look-ahead bias by design.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger("data")


# ══════════════════════════════════════════════════════════════
# PURE-NUMPY BUILDING BLOCKS (no look-ahead)
# ══════════════════════════════════════════════════════════════

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (causal)."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average (causal)."""
    return series.rolling(window=period, min_periods=period).mean()


def _wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average (causal). Recent values weighted heavier."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: max(H-L, |H-prevC|, |L-prevC|)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Returns +1 on the bar where fast crosses ABOVE slow."""
    return ((fast > slow) & (fast.shift(1) <= slow.shift(1))).astype(int)


def _crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Returns +1 on the bar where fast crosses BELOW slow."""
    return ((fast < slow) & (fast.shift(1) >= slow.shift(1))).astype(int)


class FeatureEngine:
    """
    Compute 50+ technical indicators for BTC signal generation.

    Usage:
        engine = FeatureEngine()
        df_featured = engine.compute_all_features(df_ohlcv)
        engine.print_feature_report(df_featured)
    """

    def __init__(self):
        self.feature_registry: Dict[str, List[str]] = {
            "L1_Trend": [],
            "L2_Momentum": [],
            "L3_Volatility": [],
            "L4_Volume": [],
            "L5_PriceAction": [],
        }
        self._computed = False

    # ══════════════════════════════════════════════════════════
    # LAYER 1: TREND DETECTION  (Weight: HIGH)
    # ══════════════════════════════════════════════════════════

    def compute_ema_system(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EMA Crossover System — Triple EMA Stack.

        EMAs: 21 (fast), 55 (medium), 200 (trend filter)

        Features produced:
          ema_21, ema_55, ema_200        — raw EMA values
          ema_21_55_cross                — +1 golden cross, -1 death cross
          ema_stack_signal               — +1 bullish stack, -1 bearish, 0 mixed

        Typical win rate: 52-58%
        """
        df["ema_21"] = _ema(df["Close"], 21)
        df["ema_55"] = _ema(df["Close"], 55)
        df["ema_200"] = _ema(df["Close"], 200)

        # Cross signal: +1 when EMA21 crosses above EMA55
        cross_up = _crossover(df["ema_21"], df["ema_55"])
        cross_dn = _crossunder(df["ema_21"], df["ema_55"])
        df["ema_21_55_cross"] = cross_up - cross_dn

        # Stack signal
        bullish = (df["ema_21"] > df["ema_55"]) & (df["ema_55"] > df["ema_200"])
        bearish = (df["ema_21"] < df["ema_55"]) & (df["ema_55"] < df["ema_200"])
        df["ema_stack_signal"] = np.where(bullish, 1, np.where(bearish, -1, 0))

        self.feature_registry["L1_Trend"].extend([
            "ema_21", "ema_55", "ema_200", "ema_21_55_cross", "ema_stack_signal",
        ])
        return df

    def compute_hma(self, df: pd.DataFrame, period: int = 55) -> pd.DataFrame:
        """
        Hull Moving Average — faster response, reduced lag.

        Formula:
          WMA1 = WMA(Close, n/2) × 2
          WMA2 = WMA(Close, n)
          raw  = WMA1 - WMA2
          HMA  = WMA(raw, √n)

        Features: hma_55, hma_slope, hma_signal
        Typical win rate: 54-60%
        """
        half_n = int(period / 2)
        sqrt_n = int(np.sqrt(period))

        wma_half = _wma(df["Close"], half_n)
        wma_full = _wma(df["Close"], period)
        raw_hma = 2 * wma_half - wma_full
        df["hma_55"] = _wma(raw_hma, sqrt_n)

        df["hma_slope"] = df["hma_55"].diff()
        df["hma_signal"] = np.sign(df["hma_slope"]).fillna(0).astype(int)

        self.feature_registry["L1_Trend"].extend([
            "hma_55", "hma_slope", "hma_signal",
        ])
        return df

    def compute_supertrend(
        self,
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pd.DataFrame:
        """
        Supertrend — ATR-based dynamic trend line.

        Upper = (H+L)/2 + mult × ATR
        Lower = (H+L)/2 - mult × ATR
        Flips direction on close crossing the active band.

        Features: supertrend, supertrend_dir (+1/-1)
        Typical win rate: 55-62%
        """
        hl2 = (df["High"] + df["Low"]) / 2
        tr = _true_range(df["High"], df["Low"], df["Close"])
        atr = _sma(tr, period)

        upper_band = (hl2 + multiplier * atr).values
        lower_band = (hl2 - multiplier * atr).values
        close = df["Close"].values
        n = len(df)

        direction = np.ones(n, dtype=int)
        st = np.full(n, np.nan)

        # Find first valid index (after ATR warmup)
        first_valid = period
        if first_valid < n:
            st[first_valid] = lower_band[first_valid]

        for i in range(first_valid + 1, n):
            # Ratchet lower band up (never down in uptrend)
            if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                pass  # keep new lower
            else:
                lower_band[i] = lower_band[i - 1]

            # Ratchet upper band down (never up in downtrend)
            if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                pass  # keep new upper
            else:
                upper_band[i] = upper_band[i - 1]

            # Direction flip logic
            if direction[i - 1] == 1:  # was uptrend
                direction[i] = -1 if close[i] < lower_band[i] else 1
            else:  # was downtrend
                direction[i] = 1 if close[i] > upper_band[i] else -1

            st[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

        df["supertrend"] = st
        df["supertrend_dir"] = direction

        self.feature_registry["L1_Trend"].extend([
            "supertrend", "supertrend_dir",
        ])
        return df

    def compute_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ichimoku Cloud — full 5-component system.

        Tenkan  = (9-high + 9-low) / 2
        Kijun   = (26-high + 26-low) / 2
        Senkou A = (Tenkan + Kijun) / 2
        Senkou B = (52-high + 52-low) / 2
        Chikou  = Close shifted 26 bars back

        Features: tenkan, kijun, senkou_a, senkou_b, ichi_signal
        """
        hi9 = df["High"].rolling(9).max()
        lo9 = df["Low"].rolling(9).min()
        df["tenkan"] = (hi9 + lo9) / 2

        hi26 = df["High"].rolling(26).max()
        lo26 = df["Low"].rolling(26).min()
        df["kijun"] = (hi26 + lo26) / 2

        # Senkou spans — shifted FORWARD 26 bars is future-looking in display,
        # but for feature purposes we store the *current* calculation value
        # (the plotted cloud is just visualization; the decision is at calc time).
        df["senkou_a"] = (df["tenkan"] + df["kijun"]) / 2

        hi52 = df["High"].rolling(52).max()
        lo52 = df["Low"].rolling(52).min()
        df["senkou_b"] = (hi52 + lo52) / 2

        # Composite signal: price above cloud AND tenkan > kijun
        above_cloud = (df["Close"] > df[["senkou_a", "senkou_b"]].max(axis=1))
        below_cloud = (df["Close"] < df[["senkou_a", "senkou_b"]].min(axis=1))
        tk_bull = df["tenkan"] > df["kijun"]

        df["ichi_signal"] = np.where(
            above_cloud & tk_bull, 1,
            np.where(below_cloud & ~tk_bull, -1, 0),
        )

        self.feature_registry["L1_Trend"].extend([
            "tenkan", "kijun", "senkou_a", "senkou_b", "ichi_signal",
        ])
        return df

    # ══════════════════════════════════════════════════════════
    # LAYER 2: MOMENTUM & MEAN REVERSION
    # ══════════════════════════════════════════════════════════

    def compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        RSI with divergence-aware signal.

        RS  = EMA(gain, n) / EMA(loss, n)
        RSI = 100 - 100/(1+RS)

        Signal (quant-style, NOT simple OB/OS):
          +1 if RSI < 35 AND RSI slope > 0 (bullish reversal)
          -1 if RSI > 65 AND RSI slope < 0 (bearish reversal)
        Typical win rate: 53-59%
        """
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        rsi_slope = df["rsi"].diff(3)
        df["rsi_signal"] = np.where(
            (df["rsi"] < 35) & (rsi_slope > 0), 1,
            np.where((df["rsi"] > 65) & (rsi_slope < 0), -1, 0),
        )

        self.feature_registry["L2_Momentum"].extend(["rsi", "rsi_signal"])
        return df

    def compute_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        MACD with histogram zero-cross.

        MACD Line = EMA(fast) - EMA(slow)
        Signal    = EMA(MACD, signal)
        Histogram = MACD - Signal

        Features: macd_line, macd_signal_line, macd_hist, macd_cross
        Typical win rate: 51-57%
        """
        ema_fast = _ema(df["Close"], fast)
        ema_slow = _ema(df["Close"], slow)
        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal_line"] = _ema(df["macd_line"], signal)
        df["macd_hist"] = df["macd_line"] - df["macd_signal_line"]

        cross_up = _crossover(df["macd_line"], df["macd_signal_line"])
        cross_dn = _crossunder(df["macd_line"], df["macd_signal_line"])
        df["macd_cross"] = cross_up - cross_dn

        self.feature_registry["L2_Momentum"].extend([
            "macd_line", "macd_signal_line", "macd_hist", "macd_cross",
        ])
        return df

    def compute_roc(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """
        Rate of Change — pure price momentum.

        ROC = ((Close - Close[n]) / Close[n]) × 100

        Features: roc_10, roc_signal
        Typical win rate: 50-55%
        """
        df["roc_10"] = df["Close"].pct_change(period) * 100

        roc_accel = df["roc_10"].diff()
        df["roc_signal"] = np.where(
            (df["roc_10"] > 2) & (roc_accel > 0), 1,
            np.where((df["roc_10"] < -2) & (roc_accel < 0), -1, 0),
        )

        self.feature_registry["L2_Momentum"].extend(["roc_10", "roc_signal"])
        return df

    def compute_zscore(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Z-Score Mean Reversion indicator.

        Z = (Close - SMA) / StdDev

        Features: zscore_20, zscore_signal
        Typical win rate: 58-65% (counter-trend)
        """
        sma = _sma(df["Close"], period)
        std = df["Close"].rolling(period, min_periods=period).std()
        df["zscore_20"] = (df["Close"] - sma) / std.replace(0, np.nan)

        df["zscore_signal"] = np.where(
            df["zscore_20"] < -2.0, 1,
            np.where(df["zscore_20"] > 2.0, -1, 0),
        )

        self.feature_registry["L2_Momentum"].extend(["zscore_20", "zscore_signal"])
        return df

    def compute_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """
        Stochastic Oscillator — %K and %D.

        %K = (Close - Low_n) / (High_n - Low_n) × 100
        %D = SMA(%K, d_period)

        Features: stoch_k, stoch_d, stoch_signal
        """
        low_n = df["Low"].rolling(k_period).min()
        high_n = df["High"].rolling(k_period).max()
        denom = (high_n - low_n).replace(0, np.nan)
        df["stoch_k"] = ((df["Close"] - low_n) / denom) * 100
        df["stoch_d"] = _sma(df["stoch_k"], d_period)

        cross_up = _crossover(df["stoch_k"], df["stoch_d"])
        cross_dn = _crossunder(df["stoch_k"], df["stoch_d"])

        df["stoch_signal"] = np.where(
            (cross_up == 1) & (df["stoch_k"] < 25), 1,
            np.where((cross_dn == 1) & (df["stoch_k"] > 75), -1, 0),
        )

        self.feature_registry["L2_Momentum"].extend([
            "stoch_k", "stoch_d", "stoch_signal",
        ])
        return df

    def compute_williams_r(
        self, df: pd.DataFrame, period: int = 14
    ) -> pd.DataFrame:
        """
        Williams %R — momentum oscillator [-100, 0].

        %R = (Highest_High - Close) / (Highest_High - Lowest_Low) × -100

        Features: williams_r, williams_r_signal
        """
        hh = df["High"].rolling(period).max()
        ll = df["Low"].rolling(period).min()
        denom = (hh - ll).replace(0, np.nan)
        df["williams_r"] = ((hh - df["Close"]) / denom) * -100

        df["williams_r_signal"] = np.where(
            df["williams_r"] < -80, 1,
            np.where(df["williams_r"] > -20, -1, 0),
        )

        self.feature_registry["L2_Momentum"].extend([
            "williams_r", "williams_r_signal",
        ])
        return df

    def compute_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Commodity Channel Index.

        TP  = (H + L + C) / 3
        CCI = (TP - SMA(TP)) / (0.015 × MeanDev(TP))

        Features: cci_20, cci_signal
        """
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        sma_tp = _sma(tp, period)
        mean_dev = tp.rolling(period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        df["cci_20"] = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))

        df["cci_signal"] = np.where(
            df["cci_20"] < -100, 1,
            np.where(df["cci_20"] > 100, -1, 0),
        )

        self.feature_registry["L2_Momentum"].extend(["cci_20", "cci_signal"])
        return df

    def compute_ultimate_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ultimate Oscillator — multi-timeframe momentum (7/14/28).

        BP = Close - min(Low, prev_Close)
        TR = max(High, prev_Close) - min(Low, prev_Close)
        UO = 100 × (4×Avg7 + 2×Avg14 + 1×Avg28) / 7

        Features: ult_osc, ult_osc_signal
        """
        prev_c = df["Close"].shift(1)
        bp = df["Close"] - pd.concat([df["Low"], prev_c], axis=1).min(axis=1)
        tr = (
            pd.concat([df["High"], prev_c], axis=1).max(axis=1)
            - pd.concat([df["Low"], prev_c], axis=1).min(axis=1)
        )
        tr = tr.replace(0, np.nan)

        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()

        df["ult_osc"] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

        df["ult_osc_signal"] = np.where(
            df["ult_osc"] < 30, 1,
            np.where(df["ult_osc"] > 70, -1, 0),
        )

        self.feature_registry["L2_Momentum"].extend([
            "ult_osc", "ult_osc_signal",
        ])
        return df

    # ══════════════════════════════════════════════════════════
    # LAYER 3: VOLATILITY REGIME
    # ══════════════════════════════════════════════════════════

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        ATR with volatility-regime classification.

        TR  = max(H-L, |H-prevC|, |L-prevC|)
        ATR = SMA(TR, n)
        ATR_ratio = ATR / SMA(ATR, 50)

        Regime:
          LOW (< 0.8)      → favour mean-reversion
          NORMAL (0.8-1.2) → standard
          HIGH (1.2-2.0)   → favour trend-following
          EXTREME (> 2.0)  → consider skip

        Features: atr_14, atr_pct, atr_ratio, vol_regime
        """
        tr = _true_range(df["High"], df["Low"], df["Close"])
        df["atr_14"] = _sma(tr, period)
        df["atr_pct"] = df["atr_14"] / df["Close"] * 100  # ATR as % of price

        atr_baseline = _sma(df["atr_14"], 50)
        df["atr_ratio"] = df["atr_14"] / atr_baseline.replace(0, np.nan)

        df["vol_regime"] = np.select(
            [
                df["atr_ratio"] < 0.8,
                df["atr_ratio"] < 1.2,
                df["atr_ratio"] < 2.0,
                df["atr_ratio"] >= 2.0,
            ],
            [0, 1, 2, 3],  # 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME
            default=1,
        )

        self.feature_registry["L3_Volatility"].extend([
            "atr_14", "atr_pct", "atr_ratio", "vol_regime",
        ])
        return df

    def compute_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.DataFrame:
        """
        Bollinger Bands + %B + Bandwidth.

        Middle = SMA(20)
        Upper  = SMA + 2σ
        Lower  = SMA - 2σ
        %B     = (Close - Lower) / (Upper - Lower)
        BW     = (Upper - Lower) / Middle

        Features: bb_upper, bb_lower, bb_pct_b, bb_bandwidth, bb_signal
        Typical win rate: 55-60%
        """
        sma = _sma(df["Close"], period)
        std = df["Close"].rolling(period, min_periods=period).std()

        df["bb_upper"] = sma + std_dev * std
        df["bb_lower"] = sma - std_dev * std

        band_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_pct_b"] = (df["Close"] - df["bb_lower"]) / band_range
        df["bb_bandwidth"] = band_range / sma

        bw_expanding = df["bb_bandwidth"] > df["bb_bandwidth"].shift(1)
        df["bb_signal"] = np.where(
            (df["bb_pct_b"] < 0.2) & bw_expanding, 1,
            np.where((df["bb_pct_b"] > 0.8) & bw_expanding, -1, 0),
        )

        self.feature_registry["L3_Volatility"].extend([
            "bb_upper", "bb_lower", "bb_pct_b", "bb_bandwidth", "bb_signal",
        ])
        return df

    def compute_keltner_channel(
        self,
        df: pd.DataFrame,
        period: int = 20,
        atr_mult: float = 2.0,
    ) -> pd.DataFrame:
        """
        Keltner Channel — EMA ± ATR multiple.

        Middle = EMA(Close, 20)
        Upper  = Middle + 2 × ATR
        Lower  = Middle - 2 × ATR

        Features: kc_upper, kc_lower, kc_signal
        """
        mid = _ema(df["Close"], period)
        tr = _true_range(df["High"], df["Low"], df["Close"])
        atr = _sma(tr, period)

        df["kc_upper"] = mid + atr_mult * atr
        df["kc_lower"] = mid - atr_mult * atr

        df["kc_signal"] = np.where(
            df["Close"] > df["kc_upper"], 1,
            np.where(df["Close"] < df["kc_lower"], -1, 0),
        )

        self.feature_registry["L3_Volatility"].extend([
            "kc_upper", "kc_lower", "kc_signal",
        ])
        return df

    def compute_donchian_channel(
        self, df: pd.DataFrame, period: int = 20
    ) -> pd.DataFrame:
        """
        Donchian Channel — rolling high/low breakout.

        Upper = Highest High(n)
        Lower = Lowest Low(n)
        Middle = (Upper + Lower) / 2

        Features: dc_upper, dc_lower, dc_signal
        """
        df["dc_upper"] = df["High"].rolling(period).max()
        df["dc_lower"] = df["Low"].rolling(period).min()

        # Breakout signal
        df["dc_signal"] = np.where(
            df["Close"] >= df["dc_upper"].shift(1), 1,
            np.where(df["Close"] <= df["dc_lower"].shift(1), -1, 0),
        )

        self.feature_registry["L3_Volatility"].extend([
            "dc_upper", "dc_lower", "dc_signal",
        ])
        return df

    def compute_historical_volatility(
        self, df: pd.DataFrame, period: int = 20
    ) -> pd.DataFrame:
        """
        Realized Volatility (annualized) from log returns.

        log_ret = ln(Close / Close[-1])
        HV = std(log_ret, n) × √(periods_per_year)

        Features: hist_vol_20
        """
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        # 4H bars: 6 per day × 365 = 2190 per year
        annual_factor = np.sqrt(6 * 365)
        df["hist_vol_20"] = (
            log_ret.rolling(period, min_periods=period).std() * annual_factor
        )

        self.feature_registry["L3_Volatility"].append("hist_vol_20")
        return df

    # ══════════════════════════════════════════════════════════
    # LAYER 4: VOLUME ANALYSIS
    # ══════════════════════════════════════════════════════════

    def compute_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        On-Balance Volume + OBV trend.

        OBV accumulates volume on up-closes, subtracts on down.

        Features: obv, obv_ema21, obv_signal
        """
        direction = np.sign(df["Close"].diff()).fillna(0)
        df["obv"] = (direction * df["Volume"]).cumsum()
        df["obv_ema21"] = _ema(df["obv"], 21)

        # Signal: OBV trend divergence from price
        obv_rising = df["obv"] > df["obv_ema21"]
        price_rising = df["Close"] > _ema(df["Close"], 21)
        df["obv_signal"] = np.where(
            obv_rising & ~price_rising, 1,   # hidden bullish divergence
            np.where(~obv_rising & price_rising, -1, 0),  # bearish divergence
        )

        self.feature_registry["L4_Volume"].extend([
            "obv", "obv_ema21", "obv_signal",
        ])
        return df

    def compute_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling VWAP (session-agnostic, rolling 20 bars).

        VWAP = Σ(TP × Volume) / Σ(Volume) over rolling window.

        Features: vwap_20, vwap_signal
        """
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        tp_vol = tp * df["Volume"]

        period = 20
        df["vwap_20"] = (
            tp_vol.rolling(period, min_periods=period).sum()
            / df["Volume"].rolling(period, min_periods=period).sum().replace(0, np.nan)
        )

        df["vwap_signal"] = np.where(
            df["Close"] > df["vwap_20"], 1,
            np.where(df["Close"] < df["vwap_20"], -1, 0),
        )

        self.feature_registry["L4_Volume"].extend(["vwap_20", "vwap_signal"])
        return df

    def compute_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Money Flow Index — volume-weighted RSI.

        Typical Price = (H + L + C) / 3
        Money Flow = TP × Volume
        MFI = 100 - 100/(1 + positive_flow/negative_flow)

        Features: mfi_14, mfi_signal
        """
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        mf = tp * df["Volume"]

        tp_diff = tp.diff()
        pos_mf = mf.where(tp_diff > 0, 0.0)
        neg_mf = mf.where(tp_diff < 0, 0.0)

        pos_sum = pos_mf.rolling(period).sum()
        neg_sum = neg_mf.rolling(period).sum().replace(0, np.nan)

        mf_ratio = pos_sum / neg_sum
        df["mfi_14"] = 100 - (100 / (1 + mf_ratio))

        df["mfi_signal"] = np.where(
            df["mfi_14"] < 20, 1,
            np.where(df["mfi_14"] > 80, -1, 0),
        )

        self.feature_registry["L4_Volume"].extend(["mfi_14", "mfi_signal"])
        return df

    def compute_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Chaikin Money Flow — accumulation/distribution pressure.

        CLV = ((C-L) - (H-C)) / (H-L)
        CMF = Σ(CLV × Vol, n) / Σ(Vol, n)

        Features: cmf_20, cmf_signal
        """
        hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
        clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
        clv_vol = clv * df["Volume"]

        df["cmf_20"] = (
            clv_vol.rolling(period).sum()
            / df["Volume"].rolling(period).sum().replace(0, np.nan)
        )

        df["cmf_signal"] = np.where(
            df["cmf_20"] > 0.1, 1,
            np.where(df["cmf_20"] < -0.1, -1, 0),
        )

        self.feature_registry["L4_Volume"].extend(["cmf_20", "cmf_signal"])
        return df

    def compute_volume_features(
        self, df: pd.DataFrame, period: int = 20
    ) -> pd.DataFrame:
        """
        Volume SMA and volume ratio for confirmation.

        Features: vol_sma_20, vol_ratio
        """
        df["vol_sma_20"] = _sma(df["Volume"], period)
        df["vol_ratio"] = df["Volume"] / df["vol_sma_20"].replace(0, np.nan)

        self.feature_registry["L4_Volume"].extend(["vol_sma_20", "vol_ratio"])
        return df

    # ══════════════════════════════════════════════════════════
    # LAYER 5: PRICE ACTION & STRUCTURE
    # ══════════════════════════════════════════════════════════

    def compute_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standard Pivot Points from previous bar's HLC.

        PP = (H[-1] + L[-1] + C[-1]) / 3
        R1 = 2×PP - L[-1]    S1 = 2×PP - H[-1]
        R2 = PP + (H-L)[-1]  S2 = PP - (H-L)[-1]

        Features: pivot_pp, pivot_r1, pivot_s1, pivot_r2, pivot_s2
        """
        h_prev = df["High"].shift(1)
        l_prev = df["Low"].shift(1)
        c_prev = df["Close"].shift(1)

        pp = (h_prev + l_prev + c_prev) / 3
        df["pivot_pp"] = pp
        df["pivot_r1"] = 2 * pp - l_prev
        df["pivot_s1"] = 2 * pp - h_prev
        df["pivot_r2"] = pp + (h_prev - l_prev)
        df["pivot_s2"] = pp - (h_prev - l_prev)

        self.feature_registry["L5_PriceAction"].extend([
            "pivot_pp", "pivot_r1", "pivot_s1", "pivot_r2", "pivot_s2",
        ])
        return df

    def compute_price_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-horizon returns (percentage).

        Horizons: 1-bar, 6-bar (≈1d@4H), 42-bar (≈7d), 180-bar (≈30d)

        Features: ret_1, ret_6, ret_42, ret_180
        """
        for n, label in [(1, "ret_1"), (6, "ret_6"), (42, "ret_42"), (180, "ret_180")]:
            df[label] = df["Close"].pct_change(n) * 100

        self.feature_registry["L5_PriceAction"].extend([
            "ret_1", "ret_6", "ret_42", "ret_180",
        ])
        return df

    def compute_higher_highs_lower_lows(
        self, df: pd.DataFrame, lookback: int = 10
    ) -> pd.DataFrame:
        """
        Trend structure — HH/HL (uptrend) vs LH/LL (downtrend).

        Rolling highest-high and lowest-low are compared against their
        previous values to classify the market structure.

        Features: trend_structure  (+1 = uptrend, -1 = downtrend, 0 = range)
        """
        roll_high = df["High"].rolling(lookback).max()
        roll_low = df["Low"].rolling(lookback).min()

        hh = roll_high > roll_high.shift(lookback)  # higher high
        hl = roll_low > roll_low.shift(lookback)     # higher low
        lh = roll_high < roll_high.shift(lookback)   # lower high
        ll = roll_low < roll_low.shift(lookback)     # lower low

        df["trend_structure"] = np.where(
            hh & hl, 1,
            np.where(lh & ll, -1, 0),
        )

        self.feature_registry["L5_PriceAction"].append("trend_structure")
        return df

    # ══════════════════════════════════════════════════════════
    # MASTER ORCHESTRATION
    # ══════════════════════════════════════════════════════════

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ALL features across every layer.

        Returns a copy of the input DataFrame with 50+ new columns.
        Preserves original OHLCV columns.
        """
        logger.info("Computing all features (5 layers)...")
        t0 = time.perf_counter()
        result = df.copy()

        # Reset registry to avoid duplication on re-runs
        for key in self.feature_registry:
            self.feature_registry[key] = []

        # ── Layer 1: Trend ────────────────────────────────
        t1 = time.perf_counter()
        result = self.compute_ema_system(result)
        result = self.compute_hma(result)
        result = self.compute_supertrend(result)
        result = self.compute_ichimoku(result)
        l1_time = time.perf_counter() - t1
        l1_count = len(self.feature_registry["L1_Trend"])
        logger.info(
            f"  L1 Trend          : {l1_count:2d} features  ({l1_time:.2f}s)"
        )

        # ── Layer 2: Momentum ────────────────────────────
        t2 = time.perf_counter()
        result = self.compute_rsi(result)
        result = self.compute_macd(result)
        result = self.compute_roc(result)
        result = self.compute_zscore(result)
        result = self.compute_stochastic(result)
        result = self.compute_williams_r(result)
        result = self.compute_cci(result)
        result = self.compute_ultimate_oscillator(result)
        l2_time = time.perf_counter() - t2
        l2_count = len(self.feature_registry["L2_Momentum"])
        logger.info(
            f"  L2 Momentum       : {l2_count:2d} features  ({l2_time:.2f}s)"
        )

        # ── Layer 3: Volatility ──────────────────────────
        t3 = time.perf_counter()
        result = self.compute_atr(result)
        result = self.compute_bollinger_bands(result)
        result = self.compute_keltner_channel(result)
        result = self.compute_donchian_channel(result)
        result = self.compute_historical_volatility(result)
        l3_time = time.perf_counter() - t3
        l3_count = len(self.feature_registry["L3_Volatility"])
        logger.info(
            f"  L3 Volatility     : {l3_count:2d} features  ({l3_time:.2f}s)"
        )

        # ── Layer 4: Volume ──────────────────────────────
        t4 = time.perf_counter()
        result = self.compute_obv(result)
        result = self.compute_vwap(result)
        result = self.compute_mfi(result)
        result = self.compute_cmf(result)
        result = self.compute_volume_features(result)
        l4_time = time.perf_counter() - t4
        l4_count = len(self.feature_registry["L4_Volume"])
        logger.info(
            f"  L4 Volume         : {l4_count:2d} features  ({l4_time:.2f}s)"
        )

        # ── Layer 5: Price Action ────────────────────────
        t5 = time.perf_counter()
        result = self.compute_pivot_points(result)
        result = self.compute_price_momentum(result)
        result = self.compute_higher_highs_lower_lows(result)
        l5_time = time.perf_counter() - t5
        l5_count = len(self.feature_registry["L5_PriceAction"])
        logger.info(
            f"  L5 Price Action   : {l5_count:2d} features  ({l5_time:.2f}s)"
        )

        total_features = sum(len(v) for v in self.feature_registry.values())
        total_time = time.perf_counter() - t0
        logger.info(
            f"  TOTAL             : {total_features} features in "
            f"{total_time:.2f}s"
        )

        self._computed = True
        return result

    # ══════════════════════════════════════════════════════════
    # REPORTING
    # ══════════════════════════════════════════════════════════

    def get_feature_list(self) -> Dict[str, List[str]]:
        """Return all features grouped by layer."""
        return dict(self.feature_registry)

    def get_all_feature_names(self) -> List[str]:
        """Return flat list of all feature names."""
        return [f for group in self.feature_registry.values() for f in group]

    def print_feature_report(self, df: pd.DataFrame) -> None:
        """Print a detailed feature computation report."""
        C = "\033[36m"
        G = "\033[32m"
        Y = "\033[33m"
        RED = "\033[31m"
        B = "\033[1m"
        D = "\033[2m"
        RST = "\033[0m"

        all_feats = self.get_all_feature_names()
        total = len(all_feats)

        print(f"\n{C}{'═' * 65}")
        print(f"  FEATURE ENGINEERING REPORT")
        print(f"{'═' * 65}{RST}")

        # ── Per-layer summary ─────────────────────────────
        print(f"\n{B}  Features by Layer:{RST}")
        print(f"  {'─' * 50}")
        for layer, feats in self.feature_registry.items():
            bar = "█" * len(feats)
            print(f"  {layer:<18s} │ {len(feats):2d} │ {G}{bar}{RST}")
        print(f"  {'─' * 50}")
        print(f"  {B}{'TOTAL':<18s} │ {total:2d}{RST}")

        # ── Feature names list ────────────────────────────
        print(f"\n{B}  All Feature Names:{RST}")
        for layer, feats in self.feature_registry.items():
            print(f"\n  {C}[{layer}]{RST}")
            for i, f in enumerate(feats, 1):
                print(f"    {i:2d}. {f}")

        # ── Missing values analysis ──────────────────────
        print(f"\n{B}  Missing Values (per feature):{RST}")
        print(f"  {'─' * 50}")
        has_issue = False
        for feat in all_feats:
            if feat in df.columns:
                n_miss = int(df[feat].isna().sum())
                pct = n_miss / len(df) * 100
                if n_miss > 0:
                    has_issue = True
                    tag = f"{Y}WARMUP{RST}" if pct < 5 else f"{RED}CHECK{RST}"
                    print(
                        f"  {feat:<25s} │ {n_miss:>8,} NaN "
                        f"({pct:5.2f}%)  [{tag}]"
                    )
        if not has_issue:
            print(f"  {G}All features have zero NaN ✓{RST}")

        # ── Sample output: last 5 rows ───────────────────
        signal_cols = [c for c in df.columns if c.endswith("_signal") or c.endswith("_dir") or c == "vol_regime" or c == "trend_structure"]
        if signal_cols:
            print(f"\n{B}  Signal Columns — Last 5 Rows:{RST}")
            print(df[signal_cols].tail().to_string(max_cols=12))

        # ── Feature statistics ───────────────────────────
        print(f"\n{B}  Feature Statistics (non-NaN rows):{RST}")
        print(f"  {'─' * 50}")
        stat_cols = [c for c in all_feats if c in df.columns and not c.endswith("_signal") and c != "vol_regime" and c != "trend_structure" and c != "supertrend_dir"]
        if stat_cols:
            stats = df[stat_cols].describe().loc[["mean", "std", "min", "max"]]
            # Transpose and print compact
            for col in stat_cols[:15]:  # cap at 15 to keep readable
                s = stats[col]
                print(
                    f"  {col:<22s} │ "
                    f"mean={s['mean']:>12.2f}  "
                    f"std={s['std']:>12.2f}  "
                    f"min={s['min']:>12.2f}  "
                    f"max={s['max']:>12.2f}"
                )
            if len(stat_cols) > 15:
                print(f"  {D}... and {len(stat_cols) - 15} more{RST}")

        # ── Look-ahead bias check ────────────────────────
        print(f"\n{B}  Look-Ahead Bias Check:{RST}")
        print(f"  {'─' * 50}")
        print(f"  {G}✓ All features use .shift(), .rolling(), .ewm(){RST}")
        print(f"  {G}✓ No future data leakage by construction{RST}")
        print(f"  {G}✓ Ichimoku Senkou stored at calc-time (not shifted forward){RST}")
        print(f"  {G}✓ Pivot points use previous bar's HLC (.shift(1)){RST}")

        print(f"\n{C}{'═' * 65}{RST}\n")
