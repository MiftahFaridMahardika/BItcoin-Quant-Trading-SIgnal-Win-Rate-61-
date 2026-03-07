"""
Market Regime Detection — Hidden Markov Model
===============================================

Identifies 4 market regimes from BTC OHLCV (4H) data:

  BULL     — persistent uptrend, lower volatility
  BEAR     — persistent downtrend, lower volatility
  SIDEWAYS — range-bound, directionless, low volatility
  HIGH_VOL — extremely high volatility (crisis / blow-off)

Architecture
------------
  4-state GaussianHMM (full covariance) trained on 7 regime features:

    1. ret_1d          — 1-day     log return
    2. ret_7d          — 7-day     log return
    3. ret_30d         — 30-day    log return
    4. realized_vol    — 5-day realised volatility  (annualised %)
    5. vol_ratio       — realised vol / long-run vol  (>1 = high relative vol)
    6. vol_trend       — volume deviation from 20-bar mean
    7. trend_strength  — (close − EMA50) / ATR20      (signed momentum)

Feature normalization: RobustScaler (robust to BTC's fat tails).

State label assignment (post-hoc, data-driven):
  ① State with highest realised_vol            → HIGH_VOL
  ② Highest ret_30d among remaining            → BULL
  ③ Lowest  ret_30d among remaining            → BEAR
  ④ Remaining state                            → SIDEWAYS

Usage
-----
  detector = RegimeDetector()
  detector.fit(df, "2015-01-01", "2022-12-31")
  regimes  = detector.predict(df)         # pd.Series of "BULL" / "BEAR" / ...
  proba    = detector.predict_proba(df)   # pd.DataFrame (n_bars × 4)
  detector.save("models/regime_hmm.pkl")
  detector = RegimeDetector.load("models/regime_hmm.pkl")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# ── Bars per period (4-hour data) ─────────────────────────────────────────────
_BARS_1D  = 6
_BARS_7D  = 42
_BARS_30D = 180
_BARS_VOL = 30    # 5-day realized vol window
_BARS_YEAR = 2190  # 252 trading days × 6 bars/day (approx)

# ── Regime constants ──────────────────────────────────────────────────────────
BULL     = "BULL"
BEAR     = "BEAR"
SIDEWAYS = "SIDEWAYS"
HIGH_VOL = "HIGH_VOL"

ALL_REGIMES = [BULL, BEAR, SIDEWAYS, HIGH_VOL]

# ── Regime-based trading adjustments ─────────────────────────────────────────
# score_bias: added to raw signal score BEFORE reclassification
# min_score:  minimum |adjusted score| required to take a trade
# size_mult:  position size multiplier (applied to base risk amount)
# skip:       if True, skip ALL signals in this regime

REGIME_CONFIG: Dict[str, Dict] = {
    BULL: {
        "score_bias":    +2,     # shift score upward → favour LONG signals
        "long_min_score": 4,     # same as default LONG threshold
        "short_min_score": 9,    # only STRONG_SHORT acceptable in bull market
        "size_mult":     1.00,
        "skip":          False,
        "description":   "Favour long signals, avoid shorts",
    },
    BEAR: {
        "score_bias":    -2,     # shift score downward → favour SHORT signals
        "long_min_score": 9,     # only STRONG_LONG acceptable in bear market
        "short_min_score": 4,    # same as default SHORT threshold
        "size_mult":     1.00,
        "skip":          False,
        "description":   "Favour short signals, avoid longs",
    },
    SIDEWAYS: {
        "score_bias":     0,
        "long_min_score": 6,     # tighter threshold — need clearer signal
        "short_min_score": 6,
        "size_mult":     0.75,
        "skip":          False,
        "description":   "Both directions OK, smaller size, tighter filter",
    },
    HIGH_VOL: {
        "score_bias":     0,
        "long_min_score": 99,    # effectively never trade
        "short_min_score": 99,
        "size_mult":     0.00,
        "skip":          True,
        "description":   "Skip all signals — extreme volatility, capital preservation",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 7 regime-detection features from raw OHLCV data.

    All features are dimensionless / return-space so they are comparable
    across BTC price epochs from 2015 ($200) to 2024 ($100 000+).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume (case-insensitive).

    Returns
    -------
    pd.DataFrame with 7 feature columns.  Rows before the longest look-back
    (_BARS_30D = 180 bars ≈ 30 days) will contain NaN.
    """
    close  = _col(df, "close")
    high   = _col(df, "high")
    low    = _col(df, "low")
    volume = _col(df, "volume")

    log_ret = np.log(close / close.shift(1))

    # ── 1–3  Multi-horizon returns ────────────────────────────────────────────
    ret_1d  = close.pct_change(_BARS_1D)
    ret_7d  = close.pct_change(_BARS_7D)
    ret_30d = close.pct_change(_BARS_30D)

    # ── 4  Realized volatility (annualised) ───────────────────────────────────
    realized_vol = log_ret.rolling(_BARS_VOL).std() * np.sqrt(_BARS_YEAR)

    # ── 5  Volatility ratio: realized / long-run ──────────────────────────────
    long_vol  = log_ret.rolling(_BARS_30D).std() * np.sqrt(_BARS_YEAR)
    vol_ratio = realized_vol / long_vol.clip(lower=1e-9)

    # ── 6  Volume trend: deviation from 20-bar mean (normalised) ─────────────
    if volume is not None and not volume.isnull().all():
        vol_ma     = volume.rolling(20).mean()
        vol_trend  = (volume / vol_ma.clip(lower=1.0)) - 1.0
    else:
        vol_trend = pd.Series(0.0, index=df.index)

    # ── 7  Trend strength: (close − EMA50) / ATR20 ────────────────────────────
    ema50 = close.ewm(span=50, adjust=False).mean()
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr20 = tr.rolling(20).mean()
    trend_strength = (close - ema50) / atr20.clip(lower=1e-9)

    return pd.DataFrame({
        "ret_1d":         ret_1d,
        "ret_7d":         ret_7d,
        "ret_30d":        ret_30d,
        "realized_vol":   realized_vol,
        "vol_ratio":      vol_ratio,
        "vol_trend":      vol_trend,
        "trend_strength": trend_strength,
    }, index=df.index)


def _col(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Return column by case-insensitive name (checks name, Name, NAME)."""
    for variant in [name, name.capitalize(), name.upper()]:
        if variant in df.columns:
            return df[variant]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    HMM-based Market Regime Detector for BTC.

    Parameters
    ----------
    n_states    : number of hidden states (default 4)
    n_iter      : maximum EM iterations per fit attempt
    n_init      : number of random restarts (best log-likelihood selected)
    random_state: base random seed
    """

    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 200,
        n_init: int = 5,
        random_state: int = 42,
    ):
        self.n_states     = n_states
        self.n_iter       = n_iter
        self.n_init       = n_init
        self.random_state = random_state

        self.model: Optional[object]       = None
        self.scaler: Optional[RobustScaler] = None
        self.state_to_label: Dict[int, str] = {}
        self.label_to_state: Dict[str, int] = {}
        self.feature_names: List[str]       = []
        self.train_start: str = ""
        self.train_end: str   = ""
        self.is_fitted: bool  = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, start: str, end: str) -> "RegimeDetector":
        """
        Train the Gaussian HMM on data within [start, end].

        Makes n_init attempts with different random seeds and keeps
        the model with the highest log-likelihood.

        Parameters
        ----------
        df    : raw OHLCV DataFrame (full history, any timezone)
        start : training start date  ("2015-01-01")
        end   : training end date    ("2022-12-31")
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            raise ImportError(
                "hmmlearn is required: pip install hmmlearn"
            )

        # Normalise index to tz-naive for slicing
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        df_work = df.copy()
        df_work.index = idx

        df_train = df_work.loc[start:end]
        logger.info(
            "Training HMM on %d bars (%s → %s)",
            len(df_train), df_train.index[0].date(), df_train.index[-1].date(),
        )

        feats     = compute_hmm_features(df_train).dropna()
        X_raw     = feats.values
        scaler    = RobustScaler()
        X_scaled  = scaler.fit_transform(X_raw)

        # ── Multiple restarts ──────────────────────────────────────────────
        best_model = None
        best_score = -np.inf

        for i in range(self.n_init):
            try:
                model = hmm.GaussianHMM(
                    n_components    = self.n_states,
                    covariance_type = "full",
                    n_iter          = self.n_iter,
                    random_state    = self.random_state + i,
                    verbose         = False,
                )
                model.fit(X_scaled)
                score = model.score(X_scaled)
                logger.debug("  Attempt %d/%d: log-likelihood = %.2f", i + 1, self.n_init, score)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as exc:
                logger.warning("HMM attempt %d failed: %s", i + 1, exc)

        if best_model is None:
            raise RuntimeError("All HMM training attempts failed.")

        logger.info("Best log-likelihood: %.2f", best_score)

        self.model        = best_model
        self.scaler       = scaler
        self.feature_names = list(feats.columns)
        self.train_start  = start
        self.train_end    = end

        # ── Assign human-readable labels to states ─────────────────────────
        self.state_to_label = self._assign_state_labels(X_scaled, feats.columns.tolist())
        self.label_to_state = {v: k for k, v in self.state_to_label.items()}

        self.is_fitted = True
        logger.info("State labels: %s", self.state_to_label)
        return self

    def _assign_state_labels(self, X_scaled: np.ndarray, feature_names: List[str]) -> Dict[int, str]:
        """
        Assign BULL / BEAR / SIDEWAYS / HIGH_VOL to HMM states.

        Uses each state's means (in standardised space):
          1. Highest realised_vol   → HIGH_VOL
          2. Highest ret_30d        → BULL
          3. Lowest  ret_30d        → BEAR
          4. Remaining              → SIDEWAYS
        """
        means   = self.model.means_          # (n_states, n_features)
        states  = list(range(self.n_states))

        def _feat_idx(name: str) -> int:
            return feature_names.index(name) if name in feature_names else 0

        vol_idx = _feat_idx("realized_vol")
        ret_idx = _feat_idx("ret_30d")

        state_vols = means[:, vol_idx]
        state_rets = means[:, ret_idx]

        labels: Dict[int, str] = {}
        remaining = list(states)

        # HIGH_VOL — highest vol
        hv = int(np.argmax([state_vols[s] for s in remaining]))
        labels[remaining[hv]] = HIGH_VOL
        remaining.pop(hv)

        # BULL — highest return
        bu = int(np.argmax([state_rets[s] for s in remaining]))
        labels[remaining[bu]] = BULL
        remaining.pop(bu)

        # BEAR — lowest return
        be = int(np.argmin([state_rets[s] for s in remaining]))
        labels[remaining[be]] = BEAR
        remaining.pop(be)

        # SIDEWAYS — whatever's left
        labels[remaining[0]] = SIDEWAYS

        return labels

    # ── Prediction ────────────────────────────────────────────────────────────

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        """Compute, clean, and scale HMM features for prediction."""
        # Normalise index
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        df_work = df.copy()
        df_work.index = idx

        feats      = compute_hmm_features(df_work)
        valid_mask = feats.notna().all(axis=1)
        feats_clean = feats[valid_mask]

        X_scaled = self.scaler.transform(feats_clean.values)
        return X_scaled, feats_clean.index, df_work.index

    def predict(self, df: pd.DataFrame, fallback: str = SIDEWAYS) -> pd.Series:
        """
        Predict the regime label for every bar in df.

        Bars in the warmup period (first ~180 bars) that lack full
        feature history are filled with `fallback`.

        Returns
        -------
        pd.Series[str]  index = df.index,  values ∈ {BULL, BEAR, SIDEWAYS, HIGH_VOL}
        """
        self._check_fitted()
        X, valid_idx, full_idx = self._prepare_features(df)
        states = self.model.predict(X)
        labels = pd.Series(
            [self.state_to_label[int(s)] for s in states],
            index=valid_idx,
        )
        # Reconstruct to original index with fallback for warmup
        result = pd.Series(fallback, index=full_idx)
        result.loc[labels.index] = labels.values
        # Restore original timezone-aware index if needed
        result.index = df.index
        return result

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Posterior probability of each regime for every bar.

        Returns
        -------
        pd.DataFrame  shape (n_bars, 4)
        Columns ordered: BULL, BEAR, SIDEWAYS, HIGH_VOL
        Rows for warmup period are NaN.
        """
        self._check_fitted()
        X, valid_idx, full_idx = self._prepare_features(df)
        proba = self.model.predict_proba(X)

        cols    = [self.state_to_label.get(s, f"state_{s}") for s in range(self.n_states)]
        proba_df = pd.DataFrame(proba, index=valid_idx, columns=cols)

        # Reindex to full df.index (NaN for warmup rows)
        proba_df = proba_df.reindex(full_idx)
        proba_df.index = df.index

        # Reorder columns to canonical order
        ordered = [c for c in ALL_REGIMES if c in proba_df.columns]
        return proba_df[ordered]

    def current_regime(self, df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """
        Return the regime and probability dict for the most recent bar.

        Returns
        -------
        (regime_label, {BULL: p, BEAR: p, SIDEWAYS: p, HIGH_VOL: p})
        """
        proba_df = self.predict_proba(df)
        latest   = proba_df.dropna().iloc[-1]
        return str(latest.idxmax()), latest.to_dict()

    # ── Analytics ─────────────────────────────────────────────────────────────

    def regime_stats(
        self, df: pd.DataFrame, regime_series: pd.Series
    ) -> pd.DataFrame:
        """
        Compute per-regime statistics.

        Returns pd.DataFrame with columns:
          n_bars, pct_time, mean_ann_ret, ann_vol, sharpe, avg_consecutive_bars
        """
        close   = _col(df, "close")
        log_ret = np.log(close / close.shift(1))

        rows = []
        for regime in ALL_REGIMES:
            mask = regime_series == regime
            n = int(mask.sum())
            if n == 0:
                rows.append({
                    "regime": regime, "n_bars": 0, "pct_time": 0.0,
                    "mean_ann_ret_pct": 0.0, "ann_vol_pct": 0.0,
                    "sharpe": 0.0, "avg_run_length": 0.0,
                })
                continue

            r = log_ret[mask]
            ann_ret = float(r.mean() * _BARS_YEAR * 100)
            ann_vol = float(r.std() * np.sqrt(_BARS_YEAR) * 100)
            sharpe  = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0

            # Average run length (consecutive bars in this regime)
            groups  = (regime_series != regime_series.shift()).cumsum()
            run_len = float(
                (regime_series == regime)
                .groupby(groups)
                .sum()
                .pipe(lambda s: s[s > 0].mean())
            )

            rows.append({
                "regime":          regime,
                "n_bars":          n,
                "pct_time":        round(n / len(df) * 100, 1),
                "mean_ann_ret_pct": round(ann_ret, 2),
                "ann_vol_pct":     round(ann_vol, 2),
                "sharpe":          round(sharpe, 3),
                "avg_run_length":  round(run_len, 1),
            })

        return pd.DataFrame(rows).set_index("regime")

    def transition_matrix(self, regime_series: pd.Series) -> pd.DataFrame:
        """Empirical state-transition probability matrix."""
        trans = pd.DataFrame(0, index=ALL_REGIMES, columns=ALL_REGIMES, dtype=float)
        for i in range(1, len(regime_series)):
            frm = regime_series.iloc[i - 1]
            to  = regime_series.iloc[i]
            if frm in ALL_REGIMES and to in ALL_REGIMES:
                trans.loc[frm, to] += 1
        row_sums = trans.sum(axis=1).replace(0, 1)
        return (trans.T / row_sums).T.round(4)

    # ── Signal adjustment ─────────────────────────────────────────────────────

    @staticmethod
    def adjust_signal(
        raw_score: int,
        signal_type: str,
        confidence: float,
        regime: str,
    ) -> Tuple[str, float, float]:
        """
        Apply regime bias to a raw signal score and re-classify.

        Parameters
        ----------
        raw_score   : original signal engine score
        signal_type : original classification
        confidence  : original confidence  [0, 1]
        regime      : current market regime label

        Returns
        -------
        (new_signal_type, new_confidence, size_multiplier)
        """
        cfg = REGIME_CONFIG.get(regime, REGIME_CONFIG[SIDEWAYS])

        # Skip entire regime
        if cfg["skip"]:
            return "SKIP", 0.0, 0.0

        # Apply score bias
        adj_score = raw_score + cfg["score_bias"]

        # Re-classify with regime thresholds
        long_thr  = cfg["long_min_score"]
        short_thr = cfg["short_min_score"]

        if adj_score >= 8:
            new_sig = "STRONG_LONG"
            new_conf = min(adj_score / 19, 1.0)
        elif adj_score >= long_thr:
            new_sig = "LONG"
            new_conf = adj_score / 8.0
        elif adj_score <= -8:
            new_sig = "STRONG_SHORT"
            new_conf = min(abs(adj_score) / 19, 1.0)
        elif adj_score <= -short_thr:
            new_sig = "SHORT"
            new_conf = abs(adj_score) / 8.0
        else:
            new_sig  = "SKIP"
            new_conf = 0.0

        return new_sig, min(max(new_conf, 0.0), 1.0), cfg["size_mult"]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise detector to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "model":          self.model,
                "scaler":         self.scaler,
                "state_to_label": self.state_to_label,
                "label_to_state": self.label_to_state,
                "feature_names":  self.feature_names,
                "train_start":    self.train_start,
                "train_end":      self.train_end,
                "n_states":       self.n_states,
            }, fh)
        logger.info("RegimeDetector saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeDetector":
        """Load a previously saved RegimeDetector."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls(n_states=data.get("n_states", 4))
        obj.model          = data["model"]
        obj.scaler         = data["scaler"]
        obj.state_to_label = data["state_to_label"]
        obj.label_to_state = data["label_to_state"]
        obj.feature_names  = data["feature_names"]
        obj.train_start    = data.get("train_start", "")
        obj.train_end      = data.get("train_end", "")
        obj.is_fitted      = True
        return obj

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("RegimeDetector must be fitted before prediction. Call .fit() first.")
