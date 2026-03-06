"""
Walk-Forward Optimization Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests robustness of the signal system by re-running the backtest
across rolling non-overlapping out-of-sample windows.

Methodology
-----------
* Each window has a TRAIN period (indicators/signal context) and
  a TEST period (out-of-sample execution / trade recording).
* Rule-based SignalEngine is stateless — no retraining needed.
* 60-day warmup buffer prepended to each window so ATR/EMA/etc.
  are properly initialised before the test period begins.

Default Schedule  (configurable)
-----------------
  train_months = 24  (2 years of context)
  test_months  = 12  (1 year OOS per window)
  step_months  = 12  (non-overlapping windows)
  n_windows    = 6   (2019 → 2024)

Stability Metrics
-----------------
  • Coefficient of Variation (CV) per key metric
  • Overall Consistency Score  = Σ(w_i × max(0, 1-CV_i))
  • Signal-type distribution consistency
  • Exit-reason distribution consistency
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from engines.execution_engine import BacktestEngine
from engines.risk_engine import RiskEngine
from engines.signal_engine import SignalEngine

logger = logging.getLogger("trading")

# ── WARMUP: 200 × 4H candles ≈ 33 days; we use 70 days to be safe ───────────
_WARMUP_DAYS = 70


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class WindowDef:
    """Defines one walk-forward window."""
    idx: int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp

    @property
    def label(self) -> str:
        return (f"W{self.idx:02d}  "
                f"train[{self.train_start.strftime('%Y-%m')}→"
                f"{self.train_end.strftime('%Y-%m')}]  "
                f"test[{self.test_start.strftime('%Y-%m')}→"
                f"{self.test_end.strftime('%Y-%m')}]")

    @property
    def test_label(self) -> str:
        return (f"{self.test_start.strftime('%Y-%m')} → "
                f"{self.test_end.strftime('%Y-%m')}")


@dataclass
class WindowResult:
    """Backtest + analysis result for one walk-forward window."""
    window:       WindowDef
    metrics:      Dict                      # BacktestEngine._calculate_results
    signal_dist:  Dict[str, float]          # % of each signal type
    exit_dist:    Dict[str, float]          # % of each exit reason
    elapsed_s:    float
    n_signals:    int = 0
    error:        Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.metrics.get("total_trades", 0) > 0


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ══════════════════════════════════════════════════════════════

class WalkForwardEngine:
    """
    Walk-Forward Optimization (WFO) over rolling OOS windows.

    Parameters
    ----------
    df : pd.DataFrame
        Full featured OHLCV (output of FeatureEngine with signal columns).
    bt_config : dict
        Config passed to BacktestEngine (slippage, fees, initial_capital, …).
    risk_config_path : str
        Path to risk_config.yaml.
    train_months : int
    test_months  : int
    step_months  : int
    n_windows    : int
    first_test_start : str   ISO date for the first window's test period start.
    """

    # Metrics that participate in the consistency score + their weights
    _STABILITY_WEIGHTS: Dict[str, float] = {
        "cagr_pct":          0.25,
        "sharpe_ratio":      0.25,
        "win_rate_pct":      0.20,
        "max_drawdown_pct":  0.15,
        "profit_factor":     0.15,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        bt_config: Dict,
        risk_config_path: str = "configs/risk_config.yaml",
        train_months: int = 24,
        test_months:  int = 12,
        step_months:  int = 12,
        n_windows:    int = 6,
        first_test_start: str = "2019-01-01",
    ) -> None:
        self.df = df
        self.bt_config = bt_config
        self.risk_config_path = risk_config_path
        self.train_months = train_months
        self.test_months  = test_months
        self.step_months  = step_months
        self.n_windows    = n_windows
        self.first_test_start = pd.Timestamp(first_test_start)

        self.windows:  List[WindowDef]    = []
        self.results:  List[WindowResult] = []

    # ──────────────────────────────────────────────────────────
    # WINDOW GENERATION
    # ──────────────────────────────────────────────────────────

    def _tz_strip(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Return tz-naive version of ts for cross-comparison."""
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    def _localise(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Return ts converted/localised to match df index timezone."""
        idx_tz = self.df.index.tz
        if idx_tz is None:
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        return ts.tz_convert(idx_tz) if ts.tzinfo is not None else ts.tz_localize(idx_tz)

    def _localise_pair(
        self, ts: pd.Timestamp, te: pd.Timestamp
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self._localise(ts), self._localise(te)

    def generate_windows(self) -> List[WindowDef]:
        """Auto-generate n_windows WindowDefs from first_test_start."""
        windows: List[WindowDef] = []
        test_start = self.first_test_start
        df_last_naive = self._tz_strip(self.df.index[-1])

        for i in range(self.n_windows):
            test_end   = test_start + pd.DateOffset(months=self.test_months) - pd.Timedelta(days=1)
            train_end  = test_start - pd.Timedelta(days=1)
            train_start = train_end - pd.DateOffset(months=self.train_months) + pd.Timedelta(days=1)

            # Skip window if test_end is beyond the dataset
            if self._tz_strip(test_end) > df_last_naive:
                logger.info(f"Window {i+1} extends beyond data — stopping at {i} windows.")
                break

            windows.append(WindowDef(
                idx=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))
            test_start += pd.DateOffset(months=self.step_months)

        self.windows = windows
        return windows

    # ──────────────────────────────────────────────────────────
    # SINGLE WINDOW EXECUTION
    # ──────────────────────────────────────────────────────────

    def run_window(self, window: WindowDef) -> WindowResult:
        """
        Run one OOS backtest window.

        Slice includes a _WARMUP_DAYS buffer before test_start so
        all indicators are properly initialised before trade execution.
        """
        t0 = time.perf_counter()

        # ── Build data slice (warmup + test period) ────────
        buf_start = self._localise(window.test_start - pd.Timedelta(days=_WARMUP_DAYS))
        test_end  = self._localise(window.test_end)

        if buf_start < self.df.index[0]:
            buf_start = self.df.index[0]

        mask = (self.df.index >= buf_start) & (self.df.index <= test_end)
        df_slice = self.df[mask].copy()

        if len(df_slice) < self.bt_config.get("warmup_periods", 200) + 10:
            return WindowResult(
                window=window,
                metrics={},
                signal_dist={},
                exit_dist={},
                elapsed_s=time.perf_counter() - t0,
                error="Insufficient data for window",
            )

        # ── Fresh engines per window ───────────────────────
        sig_engine  = SignalEngine()
        risk_engine = RiskEngine(config_path=self.risk_config_path)
        risk_engine.initial_capital = self.bt_config.get("initial_capital", 100_000)

        bt = BacktestEngine(config=self.bt_config)

        try:
            bt_result = bt.run_backtest(
                df_slice,
                sig_engine,
                risk_engine,
                # no start_date — warmup period is at beginning of df_slice
                end_date=str(window.test_end.date()),
                show_progress=False,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.warning(f"  Window {window.idx} failed: {exc}")
            return WindowResult(
                window=window,
                metrics={},
                signal_dist={},
                exit_dist={},
                elapsed_s=elapsed,
                error=str(exc),
            )

        # ── Signal distribution (on test slice only) ───────
        ts, te = self._localise_pair(window.test_start, window.test_end)
        test_mask = (self.df.index >= ts) & (self.df.index <= te)
        df_test = self.df[test_mask]
        sig_dist, n_signals = self._compute_signal_dist(df_test, sig_engine)

        # ── Exit reason distribution ───────────────────────
        trades = bt_result.get("trades", [])
        exit_dist = self._compute_exit_dist(trades)

        elapsed = time.perf_counter() - t0
        return WindowResult(
            window=window,
            metrics=bt_result,
            signal_dist=sig_dist,
            exit_dist=exit_dist,
            elapsed_s=elapsed,
            n_signals=n_signals,
        )

    def _compute_signal_dist(
        self, df_test: pd.DataFrame, sig_engine: SignalEngine
    ) -> Tuple[Dict[str, float], int]:
        """
        Compute signal-type distribution for the test period.
        Uses existing 'signal_type' column if available; otherwise generates.
        """
        if "signal_type" in df_test.columns:
            counts = df_test["signal_type"].value_counts(normalize=True)
        else:
            # Batch-generate signals on the test slice (correct, ~fast)
            try:
                df_with_sig = sig_engine.generate_signals_batch(df_test.copy(), start_idx=0)
                counts = df_with_sig["signal_type"].value_counts(normalize=True)
            except Exception:
                counts = pd.Series({"SKIP": 1.0})

        dist = {
            k: float(v)
            for k, v in counts.items()
        }
        n_signals = len(df_test) - int(len(df_test) * dist.get("SKIP", 1.0))
        return dist, n_signals

    @staticmethod
    def _compute_exit_dist(trades: List[Dict]) -> Dict[str, float]:
        """Exit reason distribution as fraction of total trades."""
        if not trades:
            return {}
        from collections import Counter
        c = Counter(t.get("exit_reason", "UNKNOWN") for t in trades)
        total = len(trades)
        return {k: v / total for k, v in c.items()}

    # ──────────────────────────────────────────────────────────
    # RUN ALL WINDOWS
    # ──────────────────────────────────────────────────────────

    def run_all(
        self,
        windows: Optional[List[WindowDef]] = None,
        verbose: bool = True,
    ) -> List[WindowResult]:
        """Run all (or a subset of) walk-forward windows."""
        targets = windows or self.windows
        if not targets:
            targets = self.generate_windows()

        results: List[WindowResult] = []
        for w in targets:
            if verbose:
                logger.info(f"Running {w.label}...")
            result = self.run_window(w)
            results.append(result)
            if verbose:
                m = result.metrics
                status = (
                    f"trades={m.get('total_trades', 0)}  "
                    f"ret={m.get('total_return_pct', 0):.1f}%  "
                    f"sharpe={m.get('sharpe_ratio', 0):.2f}  "
                    f"dd={m.get('max_drawdown_pct', 0):.1f}%"
                    if result.ok else f"ERROR: {result.error}"
                )
                logger.info(f"  ✓ W{w.idx}: {status}  ({result.elapsed_s:.1f}s)")

        self.results = results
        return results

    # ──────────────────────────────────────────────────────────
    # AGGREGATION
    # ──────────────────────────────────────────────────────────

    def aggregate_metrics(
        self, results: Optional[List[WindowResult]] = None
    ) -> Dict:
        """
        Compute mean, std, min, max for each backtest metric
        across all successful windows.
        """
        data = [r for r in (results or self.results) if r.ok]
        if not data:
            return {}

        metric_keys = [
            "total_return_pct", "cagr_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown_pct", "win_rate_pct", "profit_factor",
            "expectancy_r", "total_trades", "max_consecutive_losses",
        ]

        agg: Dict = {}
        for key in metric_keys:
            vals = []
            for r in data:
                v = r.metrics.get(key)
                if v is not None and np.isfinite(float(v)):
                    vals.append(float(v))
            if vals:
                arr = np.array(vals)
                agg[key] = {
                    "mean": float(np.mean(arr)),
                    "std":  float(np.std(arr)),
                    "min":  float(np.min(arr)),
                    "max":  float(np.max(arr)),
                    "values": vals,
                }

        # Buy-and-hold comparison per window
        bah_returns = []
        for r in data:
            w = r.window
            ts, te = self._localise_pair(w.test_start, w.test_end)
            mask = (self.df.index >= ts) & (self.df.index <= te)
            prices = self.df[mask].get("close", self.df[mask].get("Close", pd.Series()))
            if len(prices) > 1:
                bah_ret = (float(prices.iloc[-1]) / float(prices.iloc[0]) - 1) * 100
                bah_returns.append(bah_ret)

        if bah_returns:
            agg["buy_hold_return_pct"] = {
                "mean": float(np.mean(bah_returns)),
                "std":  float(np.std(bah_returns)),
                "min":  float(np.min(bah_returns)),
                "max":  float(np.max(bah_returns)),
                "values": bah_returns,
            }

        return agg

    # ──────────────────────────────────────────────────────────
    # STABILITY ANALYSIS
    # ──────────────────────────────────────────────────────────

    def stability_analysis(
        self, results: Optional[List[WindowResult]] = None
    ) -> Dict:
        """
        Robustness / stability report.

        Returns per-metric CV (coefficient of variation) and an
        overall Consistency Score in [0, 1].

        Consistency Score
        -----------------
        score = Σ w_i × max(0, 1 - CV_i)

        A score of 1.0 means zero variance (perfectly consistent).
        A score above 0.6 is considered "robust".
        """
        data = [r for r in (results or self.results) if r.ok]
        if not data:
            return {}

        stability: Dict = {}

        for key, weight in self._STABILITY_WEIGHTS.items():
            vals = [float(r.metrics[key]) for r in data
                    if key in r.metrics and np.isfinite(r.metrics.get(key, float('nan')))]
            if not vals:
                continue
            arr = np.array(vals)
            mean = np.mean(arr)
            std  = np.std(arr)
            cv   = std / abs(mean) if abs(mean) > 1e-9 else float('inf')
            consistency = max(0.0, 1.0 - cv)

            stability[key] = {
                "values":      vals,
                "mean":        float(mean),
                "std":         float(std),
                "cv":          float(cv),
                "consistency": float(consistency),
                "weight":      weight,
            }

        # Overall consistency score
        if stability:
            total_w = sum(v["weight"] for v in stability.values())
            score = sum(
                v["consistency"] * v["weight"]
                for v in stability.values()
            ) / total_w
            stability["overall_consistency_score"] = float(score)

            # Verdict
            if score >= 0.70:
                verdict = "ROBUST — system performs consistently across regimes"
            elif score >= 0.50:
                verdict = "MODERATE — acceptable variance, monitor closely"
            elif score >= 0.30:
                verdict = "MARGINAL — high variance, consider re-optimisation"
            else:
                verdict = "FRAGILE — strong over-fit signal, do not deploy"

            stability["verdict"] = verdict

        # ── Signal distribution stability ──────────────────
        all_types = set()
        for r in data:
            all_types.update(r.signal_dist.keys())

        sig_stability: Dict = {}
        for stype in all_types:
            vals = [r.signal_dist.get(stype, 0.0) for r in data]
            arr = np.array(vals)
            mean = np.mean(arr)
            std  = np.std(arr)
            cv   = std / mean if mean > 1e-6 else 0.0
            sig_stability[stype] = {
                "mean_pct": float(mean * 100),
                "std_pct":  float(std * 100),
                "cv":       float(cv),
            }
        stability["signal_distribution"] = sig_stability

        # ── Exit reason stability ──────────────────────────
        all_exits = set()
        for r in data:
            all_exits.update(r.exit_dist.keys())

        exit_stability: Dict = {}
        for reason in all_exits:
            vals = [r.exit_dist.get(reason, 0.0) for r in data]
            arr = np.array(vals)
            mean = np.mean(arr)
            exit_stability[reason] = {
                "mean_pct": float(mean * 100),
                "std_pct":  float(np.std(arr) * 100),
            }
        stability["exit_distribution"] = exit_stability

        # ── Profitability consistency ──────────────────────
        n_profitable = sum(1 for r in data if r.metrics.get("total_return_pct", 0) > 0)
        stability["profitable_windows"] = n_profitable
        stability["total_windows"]      = len(data)
        stability["profitability_rate"]  = n_profitable / len(data) if data else 0.0

        return stability

    # ──────────────────────────────────────────────────────────
    # FULL REPORT
    # ──────────────────────────────────────────────────────────

    def build_report(
        self, results: Optional[List[WindowResult]] = None
    ) -> Dict:
        """Assemble the complete WFO report dict."""
        data = results or self.results
        agg  = self.aggregate_metrics(data)
        stab = self.stability_analysis(data)

        # Per-window summary rows
        window_rows = []
        for r in data:
            m = r.metrics
            window_rows.append({
                "window_idx":      r.window.idx,
                "test_period":     r.window.test_label,
                "total_return_pct":   round(float(m.get("total_return_pct", 0)),   2),
                "cagr_pct":           round(float(m.get("cagr_pct", 0)),           2),
                "sharpe_ratio":       round(float(m.get("sharpe_ratio", 0)),        4),
                "sortino_ratio":      round(float(m.get("sortino_ratio", 0)),       4),
                "max_drawdown_pct":   round(float(m.get("max_drawdown_pct", 0)),   2),
                "win_rate_pct":       round(float(m.get("win_rate_pct", 0)),       2),
                "profit_factor":      round(float(m.get("profit_factor", 0) if np.isfinite(m.get("profit_factor", 0)) else 0), 4),
                "expectancy_r":       round(float(m.get("expectancy_r", 0)),       4),
                "total_trades":       int(m.get("total_trades", 0)),
                "elapsed_s":          round(r.elapsed_s, 1),
                "ok":                 r.ok,
                "error":              r.error,
            })

        return {
            "config": {
                "train_months": self.train_months,
                "test_months":  self.test_months,
                "step_months":  self.step_months,
                "n_windows":    self.n_windows,
                "first_test_start": str(self.first_test_start.date()),
            },
            "windows":          window_rows,
            "aggregated":       agg,
            "stability":        stab,
            "n_windows_run":    len(data),
            "n_windows_ok":     sum(1 for r in data if r.ok),
        }

    # ──────────────────────────────────────────────────────────
    # SAVE
    # ──────────────────────────────────────────────────────────

    def save_report(self, report: Dict, path: str) -> None:
        """Save report as YAML (human-readable) and optionally CSV."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # YAML — strip equity curves (too large) before saving
        safe = _strip_large_keys(report, drop_keys={"equity_curve", "trades", "values"})
        with open(out_path, "w") as f:
            yaml.dump(safe, f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)

        # CSV for spreadsheet analysis
        csv_path = out_path.with_suffix(".csv")
        import csv
        rows = report.get("windows", [])
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"WFO report saved → {out_path}  (+ {csv_path.name})")


# ══════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════

def _strip_large_keys(obj, drop_keys: set):
    """Recursively remove keys from nested dicts (for YAML serialisation)."""
    if isinstance(obj, dict):
        return {
            k: _strip_large_keys(v, drop_keys)
            for k, v in obj.items()
            if k not in drop_keys
        }
    if isinstance(obj, list):
        return [_strip_large_keys(v, drop_keys) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
