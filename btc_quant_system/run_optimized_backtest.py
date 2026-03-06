#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════
  BTC QUANT SYSTEM — OPTIMIZED BACKTEST RUNNER
  Period : 2019-01-01 → 2024-12-31
  Config : All optimization layers active
══════════════════════════════════════════════════════════════

Optimizations vs Baseline:
  ✓ TrendAwareSignalEngine   — bias-adjusted thresholds
  ✓ min_quality_score = 70   — confidence ≥ 0.70 filter
  ✓ AdaptiveSLTP             — regime-aware SL/TP levels
  ✓ PullbackEntry            — wait for better entry timing
  ✓ Partial exits            — 40% @ TP1, 30% @ TP2, 30% runner
  ✓ Tiered trailing stop     — locks profit after +0.5 ATR
  ✓ DynamicKelly sizing      — recent perf + market bias
  ✓ max_risk_per_trade = 3%  — increased from 2%
  ✓ max_drawdown = 20%       — increased from 15%
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from engines.data_pipeline import DataPipeline
from engines.feature_engine import FeatureEngine
from engines.signal_engine import SignalEngine
from engines.risk_engine import RiskEngine
from engines.execution_engine import BacktestEngine
from engines.trend_follower import TrendAwareSignalEngine

# ── ANSI ─────────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"
SEP = f"{C}{'═' * 65}{R}"

logging.basicConfig(
    level=logging.INFO,
    format=f"  {D}%(asctime)s{R} │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimized_bt")

# ── BASELINE (for comparison report) ─────────────────────────
BASELINE = {
    "total_return_pct":   29.54,
    "win_rate_pct":       30.2,
    "max_drawdown_pct":   13.84,
    "sharpe_ratio":       0.944,
    "profit_factor":      None,
    "vs_buyhold_pct":    -194.0,
    "sl_exit_pct":        89.0,
    "avg_win_r":          3.2,
    "avg_loss_r":         1.04,
    "total_trades":       116,
}

# ── OPTIMIZED CONFIG ──────────────────────────────────────────
OPTIMIZED_CONFIG = {
    # Signal quality
    "min_quality_score":  70,          # confidence ≥ 0.70
    "use_multi_timeframe": True,
    "volume_confirmation": True,

    # Entry
    "use_pullback_entry": True,
    "max_pullback_wait":  5,
    "entry_candle_confirmation": True,

    # SL/TP
    "adaptive_sl_tp":     True,
    "aggressive_trailing": True,
    "partial_exit_active": True,
    "partial_exit_at_tp1": 0.4,
    "partial_exit_at_tp2": 0.3,

    # Position Sizing
    "dynamic_kelly":      True,
    "streak_scaling":     True,
    "volatility_sizing":  True,

    # Trend Following
    "trend_mode":         True,
    "bias_adjusted_thresholds": True,

    # Risk
    "max_risk_per_trade": 0.03,        # 3%
    "max_drawdown":       0.20,        # 20%

    # Backtest infra
    "initial_capital":    100_000,
    "slippage_pct":       0.0005,
    "maker_fee_pct":      0.0002,
    "taker_fee_pct":      0.0004,
    "leverage":           1,
    "warmup_periods":     0,
    "use_adaptive_sltp":  True,
}

YEARS = list(range(2019, 2025))   # 2019-2024 inclusive


# ══════════════════════════════════════════════════════════════
# HIGH-SELECTIVITY SIGNAL ENGINE WRAPPER
# ══════════════════════════════════════════════════════════════

class HighSelectivityEngine:
    """
    Wraps TrendAwareSignalEngine and applies a stricter
    confidence gate (min_confidence=0.70) before signals
    are passed downstream.

    This implements "min_quality_score: 70" from the config.
    """

    def __init__(self, base_engine: TrendAwareSignalEngine,
                 min_confidence: float = 0.70):
        self._engine = base_engine
        self.min_confidence = min_confidence

    # Forward bias helpers for external callers
    def get_current_bias(self) -> str:
        return self._engine.get_current_bias()

    def get_current_thresholds(self) -> dict:
        return self._engine.get_current_thresholds()

    def calculate_signal_score(self, df, idx):
        return self._engine.signal_engine.calculate_signal_score(df, idx)

    def generate_signals_batch(self, df, start_idx=1):
        return self._engine.signal_engine.generate_signals_batch(df, start_idx)

    def generate_trading_signal(self, df: pd.DataFrame, idx: int) -> dict:
        sig = self._engine.generate_trading_signal(df, idx)

        # Gate: reject if confidence below threshold
        if sig.get("signal") not in ("SKIP", None):
            conf = sig.get("confidence", 0.0)
            if conf < self.min_confidence:
                return {
                    "signal":     "SKIP",
                    "score":      sig.get("score", 0),
                    "regime":     sig.get("regime", "NORMAL"),
                    "timestamp":  sig.get("timestamp"),
                    "reasons":    sig.get("reasons", [])
                                  + [f"[QualityFilter] conf={conf:.2f} < {self.min_confidence:.2f}"],
                    "market_bias": sig.get("market_bias", "NEUTRAL"),
                }

        return sig


# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    print(f"\n{SEP}")
    print(f"{C}  STEP 1 — LOAD CACHED FEATURES{R}")
    print(SEP)

    t0 = time.perf_counter()

    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    proc_path = PROJECT_ROOT / "data" / "processed" / "btcusd_4h.parquet"

    if feat_path.exists():
        logger.info(f"Loading cached features: {feat_path.name}")
        df = pd.read_parquet(feat_path)
    elif proc_path.exists():
        logger.info(f"Features cache missing — computing from {proc_path.name}")
        raw = pd.read_parquet(proc_path)
        fe = FeatureEngine()
        df = fe.compute_all_features(raw)
        df.to_parquet(feat_path)
        logger.info(f"Features saved to {feat_path.name}")
    else:
        logger.info("No cache found — loading raw CSV")
        pipeline = DataPipeline(
            config_path=str(PROJECT_ROOT / "configs" / "trading_config.yaml")
        )
        raw = pipeline.get_data(timeframe="4h", start_date="2017-01-01",
                                end_date="2024-12-31", use_cache=True)
        fe = FeatureEngine()
        df = fe.compute_all_features(raw)
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(feat_path)

    # Filter to relevant window (need 2017 pre-data for warmup indicators)
    df = df[(df.index >= "2017-01-01") & (df.index <= "2024-12-31 23:59:59")]

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Data ready: {len(df):,} candles │ "
        f"{df.index[0].date()} → {df.index[-1].date()} │ {elapsed:.2f}s"
    )
    return df


# ══════════════════════════════════════════════════════════════
# STEP 2 — BUILD ENGINES
# ══════════════════════════════════════════════════════════════

def build_engines():
    print(f"\n{SEP}")
    print(f"{C}  STEP 2 — BUILD OPTIMIZED ENGINES{R}")
    print(SEP)

    # Base signal engine
    base_signal = SignalEngine()

    # Trend-following wrapper (bias-adjusted classification)
    trend_engine = TrendAwareSignalEngine(base_signal, bias_recalc_bars=24)

    # Quality filter wrapper (min_confidence=0.70)
    optimized_engine = HighSelectivityEngine(
        trend_engine,
        min_confidence=OPTIMIZED_CONFIG["min_quality_score"] / 100.0  # 0.70
    )

    logger.info(f"SignalEngine: 6-layer weighted scoring (max={SignalEngine.MAX_SCORE})")
    logger.info(f"TrendAwareSignalEngine: bias-adjusted thresholds active")
    logger.info(f"HighSelectivityEngine: min_confidence={optimized_engine.min_confidence:.2f}")

    return optimized_engine


# ══════════════════════════════════════════════════════════════
# STEP 3 — RUN YEAR-BY-YEAR BACKTEST
# ══════════════════════════════════════════════════════════════

def run_backtest_year(
    year: int,
    df_full: pd.DataFrame,
    signal_engine,
    config: dict,
) -> dict:
    """Run a single-year backtest with all optimizations."""
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    # Check data availability
    mask = (df_full.index >= pd.Timestamp(start, tz="UTC")) & \
           (df_full.index <= pd.Timestamp(end,   tz="UTC"))
    if not mask.any():
        # Try without tz
        mask = (df_full.index >= start) & (df_full.index <= end)
    if not mask.any():
        logger.warning(f"  [{year}] No data available — skipping")
        return {}

    # Fresh risk engine per year (resets capital)
    risk_engine = RiskEngine(str(PROJECT_ROOT / "configs" / "risk_config.yaml"))
    risk_engine.max_risk_per_trade = config["max_risk_per_trade"]   # 3%
    risk_engine.max_total_drawdown = config["max_drawdown"]          # 20%

    backtest = BacktestEngine(config)

    results = backtest.run_backtest(
        df=df_full,
        signal_engine=signal_engine,
        risk_engine=risk_engine,
        start_date=start,
        end_date=end,
        show_progress=False,
    )
    return results


def run_all_years(df: pd.DataFrame, signal_engine) -> dict:
    print(f"\n{SEP}")
    print(f"{C}  STEP 3 — RUN OPTIMIZED BACKTEST 2019-2024{R}")
    print(SEP)

    all_results = {}
    total_trades = 0
    cumulative_pnl = 0.0
    win_trades_all = 0
    loss_trades_all = 0

    for year in YEARS:
        t0 = time.perf_counter()
        logger.info(f"Running {year}…")

        results = run_backtest_year(year, df, signal_engine, OPTIMIZED_CONFIG)

        if not results or "total_trades" not in results or results["total_trades"] == 0:
            logger.info(f"  {year} │ No trades executed")
            all_results[str(year)] = {
                "year": year, "status": "no_trades",
                "initial_capital": OPTIMIZED_CONFIG["initial_capital"],
                "final_equity": OPTIMIZED_CONFIG["initial_capital"],
                "total_pnl": 0.0, "total_return_pct": 0.0,
                "total_trades": 0, "win_rate_pct": 0.0,
                "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0,
            }
            continue

        elapsed = time.perf_counter() - t0
        ret = results["total_return_pct"]
        wr  = results["win_rate_pct"]
        pnl = results["total_pnl"]
        trades = results["total_trades"]
        wins   = results["winning_trades"]
        losses = results["losing_trades"]
        dd     = results["max_drawdown_pct"]
        sharpe = results["sharpe_ratio"]

        total_trades   += trades
        cumulative_pnl += pnl
        win_trades_all += wins
        loss_trades_all += losses

        col = G if pnl >= 0 else RED
        logger.info(
            f"  {year} │ Trades: {trades:>4d} │ "
            f"Win: {wr:>5.1f}% │ "
            f"PnL: {col}${pnl:>10,.2f} ({ret:>+6.2f}%){R} │ "
            f"DD: {dd:.1f}% │ Sharpe: {sharpe:.2f} │ {elapsed:.1f}s"
        )

        # Compact per-year dict for saving
        all_results[str(year)] = {
            "year": year,
            "initial_capital": results["initial_capital"],
            "final_equity": results["final_equity"],
            "total_pnl": round(pnl, 2),
            "total_return_pct": round(ret, 4),
            "total_trades": trades,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate_pct": round(wr, 2),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(results.get("sortino_ratio", 0), 4),
            "calmar_ratio": round(results.get("calmar_ratio", 0), 4),
            "max_drawdown_pct": round(dd, 4),
            "profit_factor": round(results.get("profit_factor", 0), 4),
            "avg_win_r": round(results.get("avg_win_r", 0), 4),
            "avg_loss_r": round(results.get("avg_loss_r", 0), 4),
            "sl_exit_pct": round(results.get("sl_exit_pct", 0), 2),
            "trail_exit_pct": round(results.get("trail_exit_pct", 0), 2),
            "tp3_exit_pct": round(results.get("tp3_exit_pct", 0), 2),
            "partial_tp1_pct": round(results.get("partial_tp1_pct", 0), 2),
            "partial_tp2_pct": round(results.get("partial_tp2_pct", 0), 2),
            "expectancy_r": round(results.get("expectancy_r", 0), 4),
            "exit_reasons": results.get("exit_reasons", {}),
            # Keep full trades list for detailed analysis
            "trades": results.get("trades", []),
        }

    # ── Aggregate stats ───────────────────────────────────────
    all_results["AGGREGATE"] = {
        "period": "2019-2024",
        "total_trades": total_trades,
        "win_trades": win_trades_all,
        "loss_trades": loss_trades_all,
        "overall_win_rate_pct": (
            win_trades_all / total_trades * 100 if total_trades > 0 else 0.0
        ),
        "total_pnl_6yr": round(cumulative_pnl, 2),
    }

    return all_results


# ══════════════════════════════════════════════════════════════
# STEP 4 — COMPUTE CONSOLIDATED METRICS
# ══════════════════════════════════════════════════════════════

def compute_consolidated(all_results: dict) -> dict:
    """Compute 6-year consolidated statistics from per-year results."""
    years_data = [
        v for k, v in all_results.items()
        if k != "AGGREGATE" and isinstance(v, dict) and v.get("total_trades", 0) > 0
    ]

    if not years_data:
        return {}

    # Total trades across all years
    total_trades  = sum(y["total_trades"]  for y in years_data)
    win_trades    = sum(y["winning_trades"] for y in years_data)
    loss_trades   = sum(y["losing_trades"]  for y in years_data)
    win_rate      = win_trades / total_trades * 100 if total_trades > 0 else 0.0

    # Weighted averages
    def wavg(key: str):
        weighted = sum(y[key] * y["total_trades"] for y in years_data
                       if key in y and y["total_trades"] > 0)
        denom = sum(y["total_trades"] for y in years_data if key in y and y["total_trades"] > 0)
        return weighted / denom if denom > 0 else 0.0

    avg_sharpe    = wavg("sharpe_ratio")
    avg_dd        = wavg("max_drawdown_pct")
    avg_pf        = wavg("profit_factor")
    avg_win_r     = wavg("avg_win_r")
    avg_loss_r    = wavg("avg_loss_r")
    avg_sl_pct    = wavg("sl_exit_pct")
    avg_trail_pct = wavg("trail_exit_pct")
    avg_tp3_pct   = wavg("tp3_exit_pct")

    # Total return across all years (sum of PnL / initial capital)
    total_pnl    = sum(y["total_pnl"] for y in years_data)
    init_capital = years_data[0]["initial_capital"]
    total_return = total_pnl / init_capital * 100

    # Max single-year drawdown
    max_dd = max(y["max_drawdown_pct"] for y in years_data)

    # Calculate vs buy & hold:
    # BTC: 2019-01-01 at ~$3,500, 2024-12-31 at ~$93,500 → +2571%
    BTC_BUYHOLD_6YR = 2571.0

    return {
        "period": "2019-2024 (6 years)",
        "total_return_pct":   round(total_return, 2),
        "total_pnl":          round(total_pnl, 2),
        "win_rate_pct":       round(win_rate, 2),
        "max_drawdown_pct":   round(max_dd, 2),
        "avg_sharpe_ratio":   round(avg_sharpe, 4),
        "avg_profit_factor":  round(avg_pf, 4),
        "avg_win_r":          round(avg_win_r, 4),
        "avg_loss_r":         round(avg_loss_r, 4),
        "total_trades":       total_trades,
        "sl_exit_pct":        round(avg_sl_pct, 2),
        "trail_exit_pct":     round(avg_trail_pct, 2),
        "tp3_exit_pct":       round(avg_tp3_pct, 2),
        "vs_buyhold_pct":     round(total_return - BTC_BUYHOLD_6YR, 2),
    }


# ══════════════════════════════════════════════════════════════
# STEP 5 — SAVE RESULTS
# ══════════════════════════════════════════════════════════════

def save_results(all_results: dict, consolidated: dict):
    print(f"\n{SEP}")
    print(f"{C}  STEP 5 — SAVE RESULTS{R}")
    print(SEP)

    out_dir = PROJECT_ROOT / "backtests" / "results"
    rpt_dir = PROJECT_ROOT / "backtests" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Main JSON ─────────────────────────────────────────
    json_path = out_dir / "optimized_backtest.json"

    # Strip trade lists before saving (keep compact)
    compact = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            compact[k] = {x: y for x, y in v.items() if x != "trades"}

    compact["CONSOLIDATED"] = consolidated

    with open(json_path, "w") as f:
        json.dump(compact, f, indent=2, default=str)
    logger.info(f"Saved: {json_path.relative_to(PROJECT_ROOT)}")

    # ── 2. Trades CSV ─────────────────────────────────────────
    all_trades = []
    for k, v in all_results.items():
        if k not in ("AGGREGATE", "CONSOLIDATED") and isinstance(v, dict):
            year_trades = v.get("trades", [])
            for t in year_trades:
                t["year"] = k
            all_trades.extend(year_trades)

    if all_trades:
        trades_path = out_dir / "optimized_trades.csv"
        pd.DataFrame(all_trades).to_csv(trades_path, index=False)
        logger.info(f"Saved: {trades_path.relative_to(PROJECT_ROOT)} ({len(all_trades)} trades)")

    # ── 3. Comparison markdown report ─────────────────────────
    md_path = out_dir / "comparison_report.md"
    _write_comparison_md(md_path, consolidated, all_results)
    logger.info(f"Saved: {md_path.relative_to(PROJECT_ROOT)}")

    return json_path, md_path


def _write_comparison_md(path: Path, consolidated: dict, all_results: dict):
    """Write markdown comparison report."""

    def _delta(new_val, base_val, higher_better=True, fmt=".2f"):
        if base_val is None or new_val is None:
            return "N/A", "—"
        delta = new_val - base_val
        if higher_better:
            ok = delta >= 0
        else:
            ok = delta <= 0
        symbol = "✅" if ok else "❌"
        sign   = "+" if delta >= 0 else ""
        return f"{sign}{delta:{fmt}}", symbol

    c = consolidated

    def _fmt(v, fmt=".2f"):
        if v is None:
            return "N/A"
        return f"{v:{fmt}}"

    tr_delta, tr_sym    = _delta(c.get("total_return_pct"), BASELINE["total_return_pct"])
    wr_delta, wr_sym    = _delta(c.get("win_rate_pct"), BASELINE["win_rate_pct"])
    dd_delta, dd_sym    = _delta(c.get("max_drawdown_pct"), BASELINE["max_drawdown_pct"], higher_better=False)
    sh_delta, sh_sym    = _delta(c.get("avg_sharpe_ratio"), BASELINE["sharpe_ratio"])
    pf_delta, pf_sym    = _delta(c.get("avg_profit_factor"), BASELINE["profit_factor"])
    bh_delta, bh_sym    = _delta(c.get("vs_buyhold_pct"), BASELINE["vs_buyhold_pct"])
    sl_delta, sl_sym    = _delta(c.get("sl_exit_pct"), BASELINE["sl_exit_pct"], higher_better=False)
    wr_r_delta, wr_r_sym = _delta(c.get("avg_win_r"), BASELINE["avg_win_r"])
    lr_delta, lr_sym    = _delta(c.get("avg_loss_r"), BASELINE["avg_loss_r"], higher_better=False)
    tr_t_delta, tr_t_sym = _delta(c.get("total_trades"), BASELINE["total_trades"])

    def _cell(val, fmt=".2f", unit="", sign=False):
        """Format a value as a right-aligned 11-char string."""
        if val is None:
            return "N/A".rjust(11)
        if sign:
            s = f"{val:+{fmt}}{unit}"
        else:
            s = f"{val:{fmt}}{unit}"
        return s.rjust(11)

    lines = [
        "# OPTIMIZATION COMPARISON REPORT",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> Period: 2019-2024 (6 years) | Initial Capital: $100,000 per year  ",
        "",
        "```",
        "╔══════════════════════════════════════════════════════════════════════════╗",
        "║                    OPTIMIZATION COMPARISON REPORT                       ║",
        "╠══════════════════════════════════════════════════════════════════════════╣",
        "║                                                                          ║",
        f"║  METRIC              │ BASELINE    │ OPTIMIZED   │ CHANGE      │ STATUS  ║",
        "║  ────────────────────┼─────────────┼─────────────┼─────────────┼──────── ║",
        f"║  Total Return        │ +29.54%     │{_cell(c.get('total_return_pct'),sign=True,unit='%')} │{tr_delta.rjust(11)}% │  {tr_sym}     ║",
        f"║  Win Rate            │ 30.2%       │{_cell(c.get('win_rate_pct'),unit='%')} │{wr_delta.rjust(11)}% │  {wr_sym}     ║",
        f"║  Max Drawdown        │ 13.84%      │{_cell(c.get('max_drawdown_pct'),unit='%')} │{dd_delta.rjust(11)}% │  {dd_sym}     ║",
        f"║  Sharpe Ratio        │ 0.944       │{_cell(c.get('avg_sharpe_ratio'),fmt='.3f')} │{sh_delta.rjust(11)} │  {sh_sym}     ║",
        f"║  Profit Factor       │ N/A         │{_cell(c.get('avg_profit_factor'),fmt='.3f')} │{'—'.rjust(11)} │  —      ║",
        f"║  vs Buy & Hold       │ -194%       │{_cell(c.get('vs_buyhold_pct'),sign=True,unit='%')} │{bh_delta.rjust(11)}% │  {bh_sym}     ║",
        f"║  SL Exit Rate        │ 89%         │{_cell(c.get('sl_exit_pct'),unit='%')} │{sl_delta.rjust(11)}% │  {sl_sym}     ║",
        f"║  Avg Win (R)         │ 3.2R        │{_cell(c.get('avg_win_r'),unit='R')} │{wr_r_delta.rjust(11)}R │  {wr_r_sym}     ║",
        f"║  Avg Loss (R)        │ 1.04R       │{_cell(c.get('avg_loss_r'),unit='R')} │{lr_delta.rjust(11)}R │  {lr_sym}     ║",
        f"║  Total Trades        │ 116         │{str(c.get('total_trades', 0)).rjust(11)} │{tr_t_delta.rjust(11)} │  {tr_t_sym}     ║",
        "║                                                                          ║",
        "╚══════════════════════════════════════════════════════════════════════════╝",
        "```",
        "",
        "## Year-by-Year Breakdown",
        "",
        "| Year | Period | Trades | Win Rate | PnL | Return | Max DD | Sharpe | Notes |",
        "|------|--------|--------|----------|-----|--------|--------|--------|-------|",
    ]

    YEAR_NOTES = {
        "2019": "Bull recovery",
        "2020": "COVID crash + recovery",
        "2021": "Peak bull",
        "2022": "Bear market",
        "2023": "Recovery",
        "2024": "ETF bull",
    }

    for year in ["2019", "2020", "2021", "2022", "2023", "2024"]:
        yr = all_results.get(year, {})
        if not yr or yr.get("total_trades", 0) == 0:
            lines.append(f"| {year} | {YEAR_NOTES.get(year,'')} | 0 | — | — | — | — | — | No trades |")
            continue
        pnl_str = f"${yr['total_pnl']:+,.0f}"
        ret_str = f"{yr['total_return_pct']:+.2f}%"
        dd_str  = f"{yr['max_drawdown_pct']:.2f}%"
        sh_str  = f"{yr['sharpe_ratio']:.2f}"
        wr_str  = f"{yr['win_rate_pct']:.1f}%"
        lines.append(
            f"| {year} | {YEAR_NOTES.get(year,'')} | {yr['total_trades']} "
            f"| {wr_str} | {pnl_str} | {ret_str} | {dd_str} | {sh_str} | — |"
        )

    lines += [
        "",
        "## Final Assessment",
        "",
        f"| Target | Goal | Result | Status |",
        f"|--------|------|--------|--------|",
        f"| Win Rate > 50% | > 50% | {_fmt(c.get('win_rate_pct'))}% | "
        f"{'✅' if (c.get('win_rate_pct') or 0) > 50 else '❌'} |",
        f"| Max DD < 20% | < 20% | {_fmt(c.get('max_drawdown_pct'))}% | "
        f"{'✅' if (c.get('max_drawdown_pct') or 100) < 20 else '❌'} |",
        f"| Beat Buy & Hold | > 0% vs B&H | {_fmt(c.get('vs_buyhold_pct'))}% | "
        f"{'✅' if (c.get('vs_buyhold_pct') or -9999) > 0 else '❌'} |",
        f"| Sharpe > 1.0 | > 1.0 | {_fmt(c.get('avg_sharpe_ratio'))} | "
        f"{'✅' if (c.get('avg_sharpe_ratio') or 0) > 1.0 else '❌'} |",
        f"| Profit Factor > 1.5 | > 1.5 | {_fmt(c.get('avg_profit_factor'))} | "
        f"{'✅' if (c.get('avg_profit_factor') or 0) > 1.5 else '❌'} |",
        "",
        "---",
        f"*Config: min_quality_score=70, max_risk=3%, max_dd=20%, "
        f"TrendAwareEngine, PullbackEntry, AdaptiveSLTP, PartialExits*",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ══════════════════════════════════════════════════════════════
# STEP 6 — PRINT FULL COMPARISON REPORT
# ══════════════════════════════════════════════════════════════

def print_comparison_report(consolidated: dict, all_results: dict):
    c = consolidated
    if not c:
        print(f"\n{RED}  No results to report.{R}")
        return

    def _col(new_val, base_val, higher_better=True):
        if new_val is None or base_val is None:
            return Y
        if higher_better:
            return G if new_val >= base_val else RED
        else:
            return G if new_val <= base_val else RED

    def _sym(new_val, base_val, higher_better=True):
        if new_val is None or base_val is None:
            return "—"
        if higher_better:
            return "✅" if new_val >= base_val else "❌"
        else:
            return "✅" if new_val <= base_val else "❌"

    def _row(label, base_str, new_val, base_val, higher_better=True, fmt=".2f", unit=""):
        col = _col(new_val, base_val, higher_better)
        sym = _sym(new_val, base_val, higher_better)
        new_str = f"{new_val:{fmt}}{unit}" if new_val is not None else "N/A"
        if new_val is not None and base_val is not None:
            delta = new_val - base_val
            delta_str = f"{'+' if delta >= 0 else ''}{delta:{fmt}}{unit}"
        else:
            delta_str = "—"
        return (
            f"║  {label:<20s}│ {base_str:>11s} │ "
            f"{col}{new_str:>11s}{R} │ {col}{delta_str:>11s}{R} │ {sym}    ║"
        )

    print(f"""
{C}╔{'═' * 74}╗
║{'OPTIMIZATION COMPARISON REPORT':^74s}║
║{'Baseline vs Optimized System (2019-2024)':^74s}║
╠{'═' * 74}╣{R}
║{' ' * 74}║
║  {'METRIC':<20s}│ {'BASELINE':>11s} │ {'OPTIMIZED':>11s} │ {'CHANGE':>11s} │ STATUS  ║
║  {'─' * 20}┼{'─' * 13}┼{'─' * 13}┼{'─' * 13}┼{'─' * 9}║""")

    print(_row("Total Return",    "+29.54%",
               c.get("total_return_pct"), BASELINE["total_return_pct"], fmt="+.2f", unit="%"))
    print(_row("Win Rate",        "30.2%",
               c.get("win_rate_pct"),    BASELINE["win_rate_pct"],     fmt=".2f", unit="%"))
    print(_row("Max Drawdown",    "13.84%",
               c.get("max_drawdown_pct"), BASELINE["max_drawdown_pct"],
               higher_better=False, fmt=".2f", unit="%"))
    print(_row("Sharpe Ratio",    "0.944",
               c.get("avg_sharpe_ratio"), BASELINE["sharpe_ratio"],   fmt=".3f"))
    print(_row("Profit Factor",   "N/A",
               c.get("avg_profit_factor"), None, fmt=".3f"))
    print(_row("vs Buy & Hold",   "-194%",
               c.get("vs_buyhold_pct"),  BASELINE["vs_buyhold_pct"],  fmt=".1f", unit="%"))
    print(_row("SL Exit Rate",    "89%",
               c.get("sl_exit_pct"),     BASELINE["sl_exit_pct"],
               higher_better=False, fmt=".1f", unit="%"))
    print(_row("Avg Win (R)",     "3.2R",
               c.get("avg_win_r"),       BASELINE["avg_win_r"],       fmt=".2f", unit="R"))
    print(_row("Avg Loss (R)",    "1.04R",
               c.get("avg_loss_r"),      BASELINE["avg_loss_r"],
               higher_better=False, fmt=".2f", unit="R"))
    print(_row("Total Trades",    "116",
               c.get("total_trades"),    BASELINE["total_trades"],    fmt=".0f"))

    print(f"""{C}║{' ' * 74}║
╚{'═' * 74}╝{R}""")

    # ── Year breakdown ────────────────────────────────────────
    print(f"\n{B}  YEAR-BY-YEAR BREAKDOWN{R}")
    print(f"  {'─' * 100}")
    print(f"  {'YEAR':<6}│ {'PERIOD':<24}│{'TRADES':>8}│{'WIN%':>8}│{'PNL':>13}│{'RETURN%':>9}│{'MAX DD':>8}│{'SHARPE':>8}")
    print(f"  {'─' * 100}")

    YEAR_NOTES = {
        "2019": "Bull recovery",
        "2020": "COVID crash/recovery",
        "2021": "Peak bull",
        "2022": "Bear market",
        "2023": "Recovery",
        "2024": "ETF bull",
    }

    for year in ["2019", "2020", "2021", "2022", "2023", "2024"]:
        yr = all_results.get(year, {})
        if not yr or yr.get("total_trades", 0) == 0:
            print(f"  {year:<6}│ {YEAR_NOTES.get(year,''):<24}│{'—':>8}│{'—':>8}│{'—':>13}│{'—':>9}│{'—':>8}│{'—':>8}")
            continue
        pnl    = yr["total_pnl"]
        ret    = yr["total_return_pct"]
        col    = G if pnl >= 0 else RED
        print(
            f"  {year:<6}│ {YEAR_NOTES.get(year,''):<24}│"
            f"{yr['total_trades']:>8}│"
            f"{yr['win_rate_pct']:>7.1f}%│"
            f"{col}${pnl:>11,.0f}{R}│"
            f"{col}{ret:>+8.2f}%{R}│"
            f"{yr['max_drawdown_pct']:>7.2f}%│"
            f"{yr['sharpe_ratio']:>8.3f}"
        )

    # ── Final verdict ─────────────────────────────────────────
    print(f"\n{B}  FINAL ASSESSMENT{R}")
    print(f"  {'─' * 55}")

    targets = [
        ("Win Rate > 50%",    c.get("win_rate_pct", 0) > 50,
         f"{c.get('win_rate_pct', 0):.1f}%"),
        ("Max DD < 20%",      c.get("max_drawdown_pct", 100) < 20,
         f"{c.get('max_drawdown_pct', 0):.2f}%"),
        ("Beat Buy & Hold",   c.get("vs_buyhold_pct", -9999) > 0,
         f"{c.get('vs_buyhold_pct', 0):.1f}%"),
        ("Sharpe > 1.0",      c.get("avg_sharpe_ratio", 0) > 1.0,
         f"{c.get('avg_sharpe_ratio', 0):.3f}"),
        ("Profit Factor > 1.5", c.get("avg_profit_factor", 0) > 1.5,
         f"{c.get('avg_profit_factor', 0):.3f}"),
    ]

    passed = 0
    for label, ok, val in targets:
        sym = f"{G}✅{R}" if ok else f"{RED}❌{R}"
        if ok:
            passed += 1
        print(f"  {sym} {label:<26} → {val}")

    grade_col = G if passed >= 4 else Y if passed >= 2 else RED
    print(f"\n  {grade_col}Targets Met: {passed}/{len(targets)}{R}")

    # ── Next steps if targets not met ─────────────────────────
    not_met = [label for label, ok, _ in targets if not ok]
    if not_met:
        print(f"\n{Y}  NEXT STEPS:{R}")
        next_steps = {
            "Win Rate > 50%": (
                "  → Raise min_quality_score to 80+, OR use ML classifier layer\n"
                "  → Filter to STRONG_LONG/STRONG_SHORT only signals"
            ),
            "Max DD < 20%": (
                "  → Tighten drawdown_scaling thresholds (kick in at 10% not 15%)\n"
                "  → Add per-year circuit-breaker after 3 consecutive losses"
            ),
            "Beat Buy & Hold": (
                "  → BTC CAGR 2019-2024 was ~76%/yr — very hard to beat levered\n"
                "  → Consider adding leverage in STRONG_BULL regime (1.5×-2×)"
            ),
            "Sharpe > 1.0": (
                "  → Reduce position sizing variance (use fixed-fraction in NORMAL)\n"
                "  → Filter out low-Sharpe years with regime overlay"
            ),
            "Profit Factor > 1.5": (
                "  → Improve R:R with wider TP3 (8× ATR in STRONG_BULL)\n"
                "  → Reduce SL tightness: use 2.0× ATR SL in trending markets"
            ),
        }
        for item in not_met:
            if item in next_steps:
                print(f"\n{Y}  [{item}]{R}")
                print(next_steps[item])


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t_total = time.perf_counter()

    print(f"""
{C}╔{'═' * 65}╗
║{'OPTIMIZED BACKTEST RUNNER — BTC QUANT SYSTEM':^65s}║
║{'Period: 2019-01-01 → 2024-12-31  │  All Optimizations':^65s}║
╚{'═' * 65}╝{R}""")

    # Step 1: Load data
    df = load_data()

    # Step 2: Build engines
    signal_engine = build_engines()

    # Step 3: Run year-by-year
    all_results = run_all_years(df, signal_engine)

    # Step 4: Consolidated metrics
    consolidated = compute_consolidated(all_results)
    all_results["CONSOLIDATED"] = consolidated

    # Step 5: Save results
    json_path, md_path = save_results(all_results, consolidated)

    # Step 6: Print comparison
    print_comparison_report(consolidated, all_results)

    # Summary
    elapsed = time.perf_counter() - t_total
    print(f"""
{G}{'═' * 65}
  ✓ Optimized Backtest Complete  │  Total time: {elapsed:.1f}s
  Results saved to:
    • {json_path.relative_to(PROJECT_ROOT)}
    • {md_path.relative_to(PROJECT_ROOT)}
{'═' * 65}{R}
""")


if __name__ == "__main__":
    main()
