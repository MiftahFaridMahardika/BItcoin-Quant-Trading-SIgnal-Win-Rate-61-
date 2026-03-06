"""
Monte Carlo Risk Simulation Engine
====================================

Simulates 10,000 trade sequences to estimate:
  - Probability of ruin  (max drawdown > 50 %)
  - Confidence intervals  for final equity
  - Distribution of max drawdowns
  - Optimal leverage      (Kelly vs Practical)
  - Target-hit probabilities  (2×, 3×, 5× capital)

Architecture
------------
  TradeStats   — summary statistics extracted from a backtest trade log
  SimResult    — raw output of one Monte Carlo run
  LeverageResult — summary of one leverage level in the sweep
  MonteCarloEngine — orchestrates simulation, analysis, and reporting
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeStats:
    """
    Summary statistics describing the historical trade distribution.

    All R values are in units of 1R  (1 × base risk amount).
    For example avg_win_r = 2.5 means the average winner netted 2.5 × the
    initial stop-loss distance.

    If raw R-series are provided (win_rs / loss_rs), bootstrap sampling is
    used; otherwise a parametric distribution is fitted.
    """

    win_rate: float           # fraction [0, 1]
    avg_win_r: float          # mean win in R units
    avg_loss_r: float         # mean |loss| in R units  (positive)
    std_win_r: float = 0.80   # std-dev of win distribution
    std_loss_r: float = 0.25  # std-dev of loss distribution
    n_trades: int = 0         # number of historical trades

    # Raw R-series for bootstrap (optional — set by from_trades)
    win_rs: Optional[np.ndarray] = field(default=None, repr=False)
    loss_rs: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Derived properties ────────────────────────────────────

    @property
    def loss_rate(self) -> float:
        return 1.0 - self.win_rate

    @property
    def expectancy_r(self) -> float:
        """Expected R per trade (positive = edge)."""
        return self.win_rate * self.avg_win_r - self.loss_rate * self.avg_loss_r

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        denom = self.loss_rate * self.avg_loss_r
        if denom <= 0:
            return float("inf")
        return (self.win_rate * self.avg_win_r) / denom

    # ── Sampling ──────────────────────────────────────────────

    def sample_wins(self, rng: np.random.Generator, shape) -> np.ndarray:
        """
        Sample win R-multiples.
        Bootstrap when sufficient real samples exist; otherwise lognormal.
        """
        if self.win_rs is not None and len(self.win_rs) >= 10:
            idx = rng.integers(0, len(self.win_rs), size=shape)
            return self.win_rs[idx]

        # Lognormal: always positive, right-skewed (realistic for wins)
        mu, sigma = _lognormal_params(self.avg_win_r, self.std_win_r)
        return rng.lognormal(mu, sigma, size=shape)

    def sample_losses(self, rng: np.random.Generator, shape) -> np.ndarray:
        """
        Sample loss R-multiples (positive values representing magnitude).
        Bootstrap when sufficient samples; otherwise truncated normal.
        """
        if self.loss_rs is not None and len(self.loss_rs) >= 10:
            idx = rng.integers(0, len(self.loss_rs), size=shape)
            return self.loss_rs[idx]

        # Half-normal (reflect normal so values are positive)
        raw = rng.normal(self.avg_loss_r, self.std_loss_r, size=shape)
        return np.abs(raw).clip(0.1)

    # ── Constructors ──────────────────────────────────────────

    @classmethod
    def from_trades(cls, trades: List[Dict]) -> "TradeStats":
        """
        Build TradeStats from a list of trade dicts.
        Each dict must have a 'pnl_r' key (signed R-multiple).
        """
        if not trades:
            raise ValueError("Empty trade list — cannot build TradeStats.")

        pnl_rs = np.array([t.get("pnl_r", 0.0) for t in trades], dtype=float)
        win_rs  = pnl_rs[pnl_rs > 0]
        loss_rs = np.abs(pnl_rs[pnl_rs < 0])

        win_rate   = len(win_rs) / len(pnl_rs)
        avg_win_r  = float(win_rs.mean())  if len(win_rs)  > 0 else 1.5
        avg_loss_r = float(loss_rs.mean()) if len(loss_rs) > 0 else 1.0
        std_win_r  = float(win_rs.std())   if len(win_rs)  > 1 else 0.50
        std_loss_r = float(loss_rs.std())  if len(loss_rs) > 1 else 0.25

        return cls(
            win_rate=win_rate,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            std_win_r=max(std_win_r, 0.05),
            std_loss_r=max(std_loss_r, 0.05),
            n_trades=len(trades),
            win_rs=win_rs   if len(win_rs)  >= 10 else None,
            loss_rs=loss_rs if len(loss_rs) >= 10 else None,
        )

    @classmethod
    def from_defaults(
        cls,
        win_rate: float = 0.35,
        avg_win_r: float = 2.5,
        avg_loss_r: float = 1.0,
        std_win_r: float = 0.80,
        std_loss_r: float = 0.25,
    ) -> "TradeStats":
        """Construct from manually specified parameters (no real trade data)."""
        return cls(
            win_rate=win_rate,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            std_win_r=std_win_r,
            std_loss_r=std_loss_r,
        )


@dataclass
class SimResult:
    """Raw output of a single Monte Carlo simulation run."""

    equity_paths: np.ndarray    # (n_sims, n_trades + 1) — absolute equity
    max_drawdowns: np.ndarray   # (n_sims,) — max DD fraction  [0, 1]
    final_equities: np.ndarray  # (n_sims,) — final capital
    leverage: float
    initial_capital: float
    risk_pct: float             # base risk per trade (fraction of equity)
    n_sims: int
    n_trades: int
    elapsed_s: float = 0.0

    @property
    def final_returns(self) -> np.ndarray:
        """Total return (fraction) for each simulation path."""
        return self.final_equities / self.initial_capital - 1.0

    @property
    def ruin_mask(self) -> np.ndarray:
        """Boolean mask: True for paths that crossed the 50 % ruin threshold."""
        return self.max_drawdowns > 0.50

    def percentile_path(self, q: float) -> np.ndarray:
        """q-th percentile equity across all paths at every time step."""
        return np.percentile(self.equity_paths, q * 100, axis=0)


@dataclass
class LeverageResult:
    """Risk/reward metrics for a single leverage level."""

    leverage: float
    mean_cagr_pct: float        # annualised return (mean across sims)
    median_cagr_pct: float
    sharpe: float               # annualised Sharpe on per-trade log returns
    mean_max_dd_pct: float
    p95_max_dd_pct: float       # 95th-percentile worst-case drawdown
    ruin_prob: float            # P(max DD > 50 %)  in [0, 1]
    target_prob_2x: float       # P(final equity ≥ 2 × initial)
    kelly_frac_of_full: float   # this leverage / full-Kelly leverage  (0–1)


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _lognormal_params(mean: float, std: float) -> Tuple[float, float]:
    """
    Return (mu, sigma) for np.random.lognormal such that
    E[X] = mean  and  Std[X] = std.
    """
    mean = max(mean, 1e-6)
    std  = max(std,  1e-6)
    sigma2 = float(np.log(1.0 + (std / mean) ** 2))
    mu     = float(np.log(mean) - sigma2 / 2.0)
    return mu, float(np.sqrt(sigma2))


def _compute_max_drawdowns(equity_paths: np.ndarray) -> np.ndarray:
    """
    Vectorised max-drawdown computation.

    equity_paths : (n_sims, n_steps)
    Returns      : (n_sims,) of max DD fractions in [0, 1]
    """
    running_max = np.maximum.accumulate(equity_paths, axis=1)
    safe_max    = np.where(running_max > 0, running_max, 1.0)
    drawdowns   = (running_max - equity_paths) / safe_max
    return drawdowns.max(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class MonteCarloEngine:
    """
    Monte Carlo Simulation for Trade Risk Analysis.

    Parameters
    ----------
    n_simulations   : number of independent paths           (default 10 000)
    n_trades        : trades per path                       (default 500)
    initial_capital : starting equity in currency units     (default 100 000)
    risk_pct        : fraction of equity risked per 1R trade (default 0.01 = 1 %)
    ruin_threshold  : drawdown fraction that defines "ruin"  (default 0.50)
    trades_per_year : assumed trading frequency for CAGR     (default 250)
    seed            : random seed for reproducibility
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        n_trades: int = 500,
        initial_capital: float = 100_000.0,
        risk_pct: float = 0.01,
        ruin_threshold: float = 0.50,
        trades_per_year: int = 250,
        seed: int = 42,
    ):
        self.n_simulations   = n_simulations
        self.n_trades        = n_trades
        self.initial_capital = initial_capital
        self.risk_pct        = risk_pct
        self.ruin_threshold  = ruin_threshold
        self.trades_per_year = trades_per_year
        self._seed           = seed

    # ── Core simulation ────────────────────────────────────────────────────

    def simulate(self, stats: TradeStats, leverage: float = 1.0) -> SimResult:
        """
        Run n_simulations of n_trades each.

        Each trade
        ----------
          outcome = WIN   with probability   stats.win_rate
          R       ~ stats.sample_wins/losses(...)
          equity_change  = R × risk_pct × leverage  (fraction of current equity)

        Equity is compounded: E_{t+1} = E_t × (1 + signed_R × effective_risk).
        Paths are clipped so equity never goes below a tiny positive value.

        Returns
        -------
        SimResult with full equity_paths matrix and derived arrays.
        """
        t0 = time.perf_counter()

        N = self.n_simulations
        T = self.n_trades
        C = self.initial_capital
        effective_risk = self.risk_pct * leverage

        rng = np.random.default_rng(self._seed)

        # ── Trade outcomes (vectorised) ──────────────────────
        wins_mask = rng.random((N, T)) < stats.win_rate          # (N, T) bool
        win_rs    = stats.sample_wins(rng, (N, T))               # (N, T)
        loss_rs   = stats.sample_losses(rng, (N, T))             # (N, T)

        # Signed R-multiple per trade
        signed_rs = np.where(wins_mask, win_rs, -loss_rs)        # (N, T)

        # Fractional equity change per trade, clipped so equity stays positive
        growth = np.clip(1.0 + signed_rs * effective_risk, 1e-6, None)

        # Compound growth via cumprod
        cum_growth = np.cumprod(growth, axis=1)                  # (N, T)

        equity_paths = np.empty((N, T + 1), dtype=np.float64)
        equity_paths[:, 0]  = C
        equity_paths[:, 1:] = C * cum_growth

        max_dds        = _compute_max_drawdowns(equity_paths)
        final_equities = equity_paths[:, -1]

        return SimResult(
            equity_paths   = equity_paths,
            max_drawdowns  = max_dds,
            final_equities = final_equities,
            leverage       = leverage,
            initial_capital= C,
            risk_pct       = self.risk_pct,
            n_sims         = N,
            n_trades       = T,
            elapsed_s      = time.perf_counter() - t0,
        )

    # ── Risk metrics ────────────────────────────────────────────────────────

    def probability_of_ruin(self, result: SimResult) -> float:
        """P(max drawdown > ruin_threshold)."""
        return float((result.max_drawdowns > self.ruin_threshold).mean())

    def probability_of_target(
        self, result: SimResult, target_multiplier: float
    ) -> float:
        """P(final equity ≥ initial_capital × target_multiplier)."""
        target = result.initial_capital * target_multiplier
        return float((result.final_equities >= target).mean())

    def confidence_intervals(
        self,
        result: SimResult,
        quantiles: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    ) -> Dict[str, float]:
        """Percentile confidence intervals for final equity."""
        ci: Dict[str, float] = {}
        for q in quantiles:
            ci[f"p{int(q * 100):02d}"] = float(
                np.percentile(result.final_equities, q * 100)
            )
        ci["mean"] = float(result.final_equities.mean())
        ci["std"]  = float(result.final_equities.std())
        return ci

    def drawdown_distribution(self, result: SimResult) -> Dict[str, float]:
        """Descriptive statistics of the max-drawdown distribution (in %)."""
        dd_pct = result.max_drawdowns * 100.0
        return {
            "mean":   float(dd_pct.mean()),
            "median": float(np.median(dd_pct)),
            "p75":    float(np.percentile(dd_pct, 75)),
            "p90":    float(np.percentile(dd_pct, 90)),
            "p95":    float(np.percentile(dd_pct, 95)),
            "p99":    float(np.percentile(dd_pct, 99)),
            "max":    float(dd_pct.max()),
        }

    # ── Kelly Criterion ─────────────────────────────────────────────────────

    def kelly_fraction(self, stats: TradeStats) -> float:
        """
        Full Kelly fraction — optimal fraction of equity to risk per trade.

        f* = (WR × avgW − LR × avgL) / avgW

        Interpretation: if f* = 0.05 and risk_pct = 0.01, the Kelly-optimal
        leverage is 5× (risk 5 % of equity per trade).
        """
        num = stats.win_rate * stats.avg_win_r - stats.loss_rate * stats.avg_loss_r
        if stats.avg_win_r <= 0:
            return 0.0
        return float(max(0.0, num / stats.avg_win_r))

    def kelly_optimal_leverage(self, stats: TradeStats) -> float:
        """
        Translate Kelly fraction to a leverage multiplier relative to risk_pct.

        kelly_leverage = kelly_fraction / risk_pct
        half_kelly     = kelly_leverage / 2
        """
        kf = self.kelly_fraction(stats)
        if self.risk_pct <= 0:
            return 0.0
        return kf / self.risk_pct

    # ── Leverage sweep ──────────────────────────────────────────────────────

    def leverage_sweep(
        self,
        stats: TradeStats,
        leverages: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0),
        n_sims_per_lev: int = 3_000,
    ) -> List[LeverageResult]:
        """
        Simulate at each leverage level and record risk/reward metrics.
        Uses n_sims_per_lev < n_simulations for speed (3 000 gives ±1.5 % CI
        on probabilities, sufficient for leverage comparison).
        """
        kelly_lev = self.kelly_optimal_leverage(stats)

        # Lightweight engine for the sweep
        sweep_engine = MonteCarloEngine(
            n_simulations  = n_sims_per_lev,
            n_trades       = self.n_trades,
            initial_capital= self.initial_capital,
            risk_pct       = self.risk_pct,
            ruin_threshold = self.ruin_threshold,
            trades_per_year= self.trades_per_year,
            seed           = self._seed,
        )

        results: List[LeverageResult] = []
        n_years = self.n_trades / self.trades_per_year

        for lev in leverages:
            r = sweep_engine.simulate(stats, leverage=lev)

            # CAGR per simulation path
            with np.errstate(divide="ignore", invalid="ignore"):
                cagr_arr = (r.final_equities / self.initial_capital) ** (1.0 / n_years) - 1.0
            cagr_arr = np.nan_to_num(cagr_arr, nan=0.0, posinf=10.0, neginf=-1.0)

            # Sharpe on per-trade log returns (annualised)
            log_ret = np.log(
                r.equity_paths[:, 1:] / r.equity_paths[:, :-1].clip(1e-9)
            )
            per_path_mean = log_ret.mean(axis=1)
            per_path_std  = log_ret.std(axis=1) + 1e-9
            sharpe_arr    = per_path_mean / per_path_std * np.sqrt(self.trades_per_year)
            sharpe        = float(np.median(sharpe_arr))

            kelly_frac = float(lev / kelly_lev) if kelly_lev > 0 else 0.0

            results.append(LeverageResult(
                leverage          = float(lev),
                mean_cagr_pct     = float(cagr_arr.mean()  * 100),
                median_cagr_pct   = float(np.median(cagr_arr) * 100),
                sharpe            = sharpe,
                mean_max_dd_pct   = float(r.max_drawdowns.mean() * 100),
                p95_max_dd_pct    = float(np.percentile(r.max_drawdowns, 95) * 100),
                ruin_prob         = float((r.max_drawdowns > self.ruin_threshold).mean()),
                target_prob_2x    = float(
                    (r.final_equities >= 2.0 * self.initial_capital).mean()
                ),
                kelly_frac_of_full= min(kelly_frac, 2.0),
            ))

        return results

    def practical_optimal_leverage(
        self,
        lev_results: List[LeverageResult],
        max_ruin_prob: float = 0.05,
        min_sharpe: float = 0.0,
    ) -> Optional[LeverageResult]:
        """
        Find the highest leverage where:
          ruin_prob ≤ max_ruin_prob  AND  Sharpe ≥ min_sharpe.

        Returns None if no leverage level satisfies both constraints.
        """
        candidates = [
            r for r in lev_results
            if r.ruin_prob <= max_ruin_prob and r.sharpe >= min_sharpe
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.sharpe)

    # ── Report assembly ─────────────────────────────────────────────────────

    def build_report(
        self,
        stats: TradeStats,
        base_result: SimResult,
        lev_results: List[LeverageResult],
    ) -> Dict:
        """
        Assemble a comprehensive report dict suitable for YAML serialisation.
        """
        ci       = self.confidence_intervals(base_result)
        dd_dist  = self.drawdown_distribution(base_result)
        kelly_lv = self.kelly_optimal_leverage(stats)
        prac     = self.practical_optimal_leverage(lev_results)

        n_years      = self.n_trades / self.trades_per_year
        final_arr    = base_result.final_equities
        cagr_arr     = (final_arr / self.initial_capital) ** (1.0 / n_years) - 1.0
        cagr_arr     = np.nan_to_num(cagr_arr)

        return {
            "trade_stats": {
                "win_rate_pct":          round(stats.win_rate   * 100, 2),
                "loss_rate_pct":         round(stats.loss_rate  * 100, 2),
                "avg_win_r":             round(stats.avg_win_r,   3),
                "avg_loss_r":            round(stats.avg_loss_r,  3),
                "expectancy_r":          round(stats.expectancy_r, 3),
                "profit_factor":         round(stats.profit_factor, 3),
                "n_historical_trades":   stats.n_trades,
                "sampling_mode":         "bootstrap" if stats.win_rs is not None else "parametric",
            },
            "simulation_config": {
                "n_simulations":         base_result.n_sims,
                "n_trades_per_sim":      base_result.n_trades,
                "base_leverage":         base_result.leverage,
                "risk_pct_per_trade":    round(base_result.risk_pct * 100, 2),
                "trades_per_year":       self.trades_per_year,
                "elapsed_s":             round(base_result.elapsed_s, 2),
            },
            "ruin_analysis": {
                "ruin_threshold_pct":    round(self.ruin_threshold * 100, 1),
                "probability_of_ruin_pct": round(self.probability_of_ruin(base_result) * 100, 2),
                "probability_target_2x_pct": round(self.probability_of_target(base_result, 2.0) * 100, 2),
                "probability_target_3x_pct": round(self.probability_of_target(base_result, 3.0) * 100, 2),
                "probability_target_5x_pct": round(self.probability_of_target(base_result, 5.0) * 100, 2),
            },
            "cagr_distribution": {
                "mean_pct":   round(float(cagr_arr.mean() * 100), 2),
                "median_pct": round(float(np.median(cagr_arr) * 100), 2),
                "p05_pct":    round(float(np.percentile(cagr_arr, 5) * 100), 2),
                "p25_pct":    round(float(np.percentile(cagr_arr, 25) * 100), 2),
                "p75_pct":    round(float(np.percentile(cagr_arr, 75) * 100), 2),
                "p95_pct":    round(float(np.percentile(cagr_arr, 95) * 100), 2),
            },
            "final_equity_confidence_intervals": {
                k: round(v, 2) for k, v in ci.items()
            },
            "max_drawdown_distribution_pct": {
                k: round(v, 2) for k, v in dd_dist.items()
            },
            "leverage_analysis": {
                "kelly_full_leverage":             round(kelly_lv, 3),
                "kelly_half_leverage":             round(kelly_lv / 2.0, 3),
                "kelly_fraction_pct":              round(self.kelly_fraction(stats) * 100, 2),
                "practical_optimal_leverage":      round(prac.leverage, 2) if prac else None,
                "practical_max_ruin_threshold_pct": 5.0,
                "practical_min_sharpe":            0.0,
            },
            "leverage_sweep": [
                {
                    "leverage":           r.leverage,
                    "mean_cagr_pct":      round(r.mean_cagr_pct,   2),
                    "median_cagr_pct":    round(r.median_cagr_pct, 2),
                    "sharpe":             round(r.sharpe,           3),
                    "mean_max_dd_pct":    round(r.mean_max_dd_pct, 2),
                    "p95_max_dd_pct":     round(r.p95_max_dd_pct,  2),
                    "ruin_prob_pct":      round(r.ruin_prob * 100,  2),
                    "target_prob_2x_pct": round(r.target_prob_2x * 100, 2),
                    "kelly_frac_of_full": round(r.kelly_frac_of_full, 2),
                }
                for r in lev_results
            ],
        }

    def save_report(self, report: Dict, path: Path) -> None:
        """Save report dict to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(report, fh, default_flow_style=False, sort_keys=False)
        logger.info("Report saved → %s", path)
