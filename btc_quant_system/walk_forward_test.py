#!/usr/bin/env python3
"""
Walk-Forward Optimization — End-to-End Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Load featured 4H data
2. Define 6 OOS windows  (2019 → 2024, 2-year train / 1-year test)
3. Run backtest per window
4. Aggregate performance metrics
5. Stability & robustness analysis
6. Save report → backtests/reports/wfo_report.yaml + .csv

Interpretation Guide
--------------------
  Consistency Score ≥ 0.70  → ROBUST
  Consistency Score 0.50-0.70 → MODERATE
  Consistency Score < 0.50  → FRAGILE (likely overfit)
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from utils.logger import setup_all_loggers
from utils.helpers import format_number, format_duration
from engines.walk_forward import WalkForwardEngine, WindowResult
from engines.signal_engine import SignalEngine

# ── ANSI ─────────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"

W   = 76
SEP = f"{C}{'═' * W}{R}"
SUB = f"  {'─' * (W - 2)}"

# ── Walk-Forward Configuration ────────────────────────────────
WFO_CONFIG = {
    "train_months":     24,   # 2 years of context per window
    "test_months":      12,   # 1 year OOS per window → 6 × 1Y = 2019-2024
    "step_months":      12,   # non-overlapping test windows
    "n_windows":         6,
    "first_test_start": "2019-01-01",
}

# ── Backtest Configuration ─────────────────────────────────────
BT_CONFIG = {
    "initial_capital": 100_000,
    "slippage_pct":    0.0005,   # 0.05%
    "maker_fee_pct":   0.0002,   # 0.02%
    "taker_fee_pct":   0.0004,   # 0.04%
    "warmup_periods":  200,
    "leverage":        1,
}

RISK_CONFIG_PATH = str(PROJECT_ROOT / "configs" / "risk_config.yaml")
REPORT_PATH      = PROJECT_ROOT / "backtests" / "reports" / "wfo_report.yaml"


# ═══════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════

def _col(v: float, good_positive: bool = True) -> str:
    """Return ANSI colour for a numeric value."""
    if np.isnan(v) or not np.isfinite(v):
        return D
    if good_positive:
        return G if v > 0 else (RED if v < 0 else D)
    else:                         # lower is better (e.g. drawdown, CV)
        return G if v < 0.3 else (Y if v < 0.6 else RED)


def _pbar(fraction: float, width: int = 18) -> str:
    f = max(0.0, min(1.0, fraction))
    filled = int(f * width)
    return f"{G}{'█' * filled}{D}{'░' * (width - filled)}{R}"


def _consistency_badge(score: float) -> str:
    if score >= 0.70:
        return f"{G}ROBUST{R}"
    if score >= 0.50:
        return f"{Y}MODERATE{R}"
    if score >= 0.30:
        return f"{Y}MARGINAL{R}"
    return f"{RED}FRAGILE{R}"


# ═══════════════════════════════════════════════════════════════
# STEP PRINTERS
# ═══════════════════════════════════════════════════════════════

def print_window_schedule(engine: WalkForwardEngine) -> None:
    print(f"\n{SEP}")
    print(f"{C}  WINDOW SCHEDULE{R}")
    print(SEP)
    print(f"\n  {B}{'#':>3s}  {'Train Period':^27s}  {'Test Period':^23s}  {'Status'}{R}")
    print(SUB)
    for w in engine.windows:
        print(
            f"  {w.idx:3d}  "
            f"{w.train_start.strftime('%Y-%m-%d')} → {w.train_end.strftime('%Y-%m-%d')}  "
            f"{w.test_start.strftime('%Y-%m-%d')} → {w.test_end.strftime('%Y-%m-%d')}  "
            f"{D}pending{R}"
        )
    print()


def print_window_result(result: WindowResult, bah_ret: float) -> None:
    """Print a single window result inline."""
    w  = result.window
    m  = result.metrics

    if not result.ok:
        print(f"  {RED}W{w.idx:02d}  ERROR: {result.error}{R}")
        return

    ret  = m.get("total_return_pct", 0)
    sh   = m.get("sharpe_ratio", 0)
    dd   = m.get("max_drawdown_pct", 0)
    wr   = m.get("win_rate_pct", 0)
    pf   = m.get("profit_factor", 0)
    if not np.isfinite(pf):
        pf = 0.0
    tr   = m.get("total_trades", 0)

    ret_col = G if ret > 0 else RED
    bah_col = G if bah_ret > 0 else RED
    alpha   = ret - bah_ret
    alpha_col = G if alpha > 0 else RED

    print(
        f"  {B}W{w.idx:02d}{R}  "
        f"{w.test_start.strftime('%Y-%m')} → {w.test_end.strftime('%Y-%m')}  "
        f"ret={ret_col}{ret:+7.1f}%{R}  "
        f"BaH={bah_col}{bah_ret:+7.1f}%{R}  "
        f"α={alpha_col}{alpha:+6.1f}%{R}  "
        f"sh={sh:.2f}  dd={dd:.1f}%  "
        f"wr={wr:.0f}%  pf={pf:.2f}  "
        f"n={tr}  ({result.elapsed_s:.1f}s)"
    )


def print_per_window_table(results: list, bah_rets: list) -> None:
    print(f"\n{SEP}")
    print(f"{C}  PER-WINDOW PERFORMANCE{R}")
    print(SEP)
    print(f"\n  {B}{'W':>2s}  {'Test Period':^23s}  {'Return':>8s}  {'BaH':>8s}  {'Alpha':>7s}  "
          f"{'Sharpe':>7s}  {'MaxDD':>7s}  {'WR':>6s}  {'PF':>5s}  {'Trades':>6s}{R}")
    print(SUB)

    for r, bah in zip(results, bah_rets):
        if not r.ok:
            print(f"  {r.window.idx:2d}  {r.window.test_label:<23s}  {RED}ERROR{R}")
            continue
        m  = r.metrics
        ret = m.get("total_return_pct", 0)
        sh  = m.get("sharpe_ratio", 0)
        dd  = m.get("max_drawdown_pct", 0)
        wr  = m.get("win_rate_pct", 0)
        pf  = m.get("profit_factor", 0)
        if not np.isfinite(pf):
            pf = 0.0
        tr  = m.get("total_trades", 0)
        alpha = ret - bah

        print(
            f"  {r.window.idx:2d}  {r.window.test_label:<23s}  "
            f"{_col(ret)}{ret:+8.2f}%{R}  "
            f"{_col(bah)}{bah:+8.2f}%{R}  "
            f"{_col(alpha)}{alpha:+7.2f}%{R}  "
            f"{_col(sh)}{sh:7.3f}{R}  "
            f"{_col(-dd, good_positive=False)}{dd:7.2f}%{R}  "
            f"{_col(wr-50)}{wr:5.1f}%{R}  "
            f"{_col(pf-1)}{pf:5.2f}{R}  "
            f"{tr:6d}"
        )

    print()


def print_aggregated_metrics(agg: dict) -> None:
    print(f"\n{SEP}")
    print(f"{C}  AGGREGATED METRICS (across all windows){R}")
    print(SEP)

    keys = [
        ("total_return_pct",   "Return (%)",         True,  "+.1f"),
        ("cagr_pct",           "CAGR (%)",           True,  "+.2f"),
        ("sharpe_ratio",       "Sharpe Ratio",       True,  ".3f"),
        ("sortino_ratio",      "Sortino Ratio",      True,  ".3f"),
        ("calmar_ratio",       "Calmar Ratio",       True,  ".3f"),
        ("max_drawdown_pct",   "Max Drawdown (%)",   False, ".2f"),
        ("win_rate_pct",       "Win Rate (%)",       True,  ".1f"),
        ("profit_factor",      "Profit Factor",      True,  ".3f"),
        ("expectancy_r",       "Expectancy (R)",     True,  ".3f"),
        ("total_trades",       "Trades / Window",    True,  ".0f"),
        ("buy_hold_return_pct","Buy & Hold (%)",     True,  "+.1f"),
    ]

    print(f"\n  {B}{'Metric':<24s}  {'Mean':>9s}  {'Std':>8s}  {'Min':>9s}  {'Max':>9s}  Distribution{R}")
    print(SUB)

    for key, label, good_pos, fmt in keys:
        if key not in agg:
            continue
        a = agg[key]
        mean = a["mean"]
        std  = a["std"]
        mn   = a["min"]
        mx   = a["max"]
        rng  = mx - mn if mx != mn else 1
        bar  = _pbar((mean - mn) / rng if rng > 0 else 0.5)

        col = _col(mean, good_pos)
        fmt_str = f"{{:{fmt}}}"
        print(
            f"  {label:<24s}  "
            f"{col}{fmt_str.format(mean):>9s}{R}  "
            f"{D}{std:>8.2f}{R}  "
            f"{fmt_str.format(mn):>9s}  "
            f"{fmt_str.format(mx):>9s}  "
            f"{bar}"
        )

    print()


def print_stability_analysis(stab: dict) -> None:
    print(f"\n{SEP}")
    print(f"{C}  STABILITY & ROBUSTNESS ANALYSIS{R}")
    print(SEP)

    # Overall score
    score   = stab.get("overall_consistency_score", 0)
    verdict = stab.get("verdict", "—")
    badge   = _consistency_badge(score)
    prof_ok = stab.get("profitable_windows", 0)
    prof_n  = stab.get("total_windows", 1)

    print(f"\n  {B}Overall Consistency Score :{R}  {score:.3f} / 1.000  [{badge}]")
    print(f"  {B}Verdict                  :{R}  {verdict}")
    print(f"  {B}Profitable Windows       :{R}  {prof_ok} / {prof_n}  "
          f"({prof_ok/prof_n*100:.0f}%)")

    # Per-metric breakdown
    metric_labels = {
        "cagr_pct":         "CAGR",
        "sharpe_ratio":     "Sharpe",
        "win_rate_pct":     "Win Rate",
        "max_drawdown_pct": "Max DD",
        "profit_factor":    "Profit Factor",
    }

    print(f"\n  {B}{'Metric':<18s}  {'Mean':>8s}  {'CV':>6s}  {'Consistency':>12s}  {'Weight':>6s}  Bar{R}")
    print(SUB)

    for key, label in metric_labels.items():
        if key not in stab:
            continue
        s   = stab[key]
        cv  = s["cv"]
        con = s["consistency"]
        wt  = s["weight"]
        col = _col(con - 0.5, good_positive=True)
        print(
            f"  {label:<18s}  "
            f"{s['mean']:>8.3f}  "
            f"{cv:>6.3f}  "
            f"  {col}{con:>10.3f}{R}  "
            f"{wt:>6.2f}  "
            f"{_pbar(con)}"
        )

    # Signal distribution stability
    sig_stab = stab.get("signal_distribution", {})
    if sig_stab:
        print(f"\n  {B}Signal Distribution (mean % across windows):{R}")
        print(f"  {D}{'Signal Type':<16s}  {'Mean':>6s}  {'Std':>6s}  {'CV':>6s}{R}")
        for stype in ["STRONG_LONG", "LONG", "SKIP", "SHORT", "STRONG_SHORT"]:
            if stype not in sig_stab:
                continue
            s = sig_stab[stype]
            col = G if stype in ("STRONG_LONG", "LONG") else (RED if stype in ("SHORT", "STRONG_SHORT") else D)
            print(f"  {col}{stype:<16s}{R}  {s['mean_pct']:6.2f}%  {s['std_pct']:6.2f}%  {s['cv']:6.3f}")

    # Exit reason distribution
    exit_stab = stab.get("exit_distribution", {})
    if exit_stab:
        print(f"\n  {B}Exit Distribution (mean % across windows):{R}")
        for reason, s in sorted(exit_stab.items(), key=lambda x: -x[1]["mean_pct"]):
            col = RED if reason == "STOP_LOSS" else G
            print(f"  {col}{reason:<16s}{R}  {s['mean_pct']:6.2f}%  ±{s['std_pct']:5.2f}%")

    print()


def print_recommendations(stab: dict, agg: dict) -> None:
    print(f"\n{SEP}")
    print(f"{C}  RECOMMENDATIONS{R}")
    print(SEP)

    score = stab.get("overall_consistency_score", 0)
    recs  = []

    # Performance checks
    mean_sharpe = agg.get("sharpe_ratio", {}).get("mean", 0)
    mean_dd     = agg.get("max_drawdown_pct", {}).get("mean", 0)
    mean_wr     = agg.get("win_rate_pct", {}).get("mean", 0)
    mean_pf     = agg.get("profit_factor", {}).get("mean", 0)
    prof_rate   = stab.get("profitability_rate", 0)
    cv_sharpe   = stab.get("sharpe_ratio", {}).get("cv", 1)
    cv_ret      = stab.get("cagr_pct", {}).get("cv", 1)

    # ── Consistency ───────────────────────────────────────
    if score >= 0.70:
        recs.append((G, "✓", "System is ROBUST — consistent across all market regimes."))
    elif score >= 0.50:
        recs.append((Y, "→", "Moderate consistency — acceptable for deployment with active monitoring."))
    else:
        recs.append((RED, "✗", "Low consistency — investigate parameter sensitivity or reduce complexity."))

    # ── Sharpe ────────────────────────────────────────────
    if mean_sharpe >= 1.5:
        recs.append((G, "✓", f"Sharpe ratio avg={mean_sharpe:.2f} exceeds 1.5 target."))
    elif mean_sharpe >= 0.8:
        recs.append((Y, "→", f"Sharpe ratio avg={mean_sharpe:.2f} acceptable but below 1.5 target."))
    else:
        recs.append((RED, "✗", f"Sharpe ratio avg={mean_sharpe:.2f} is too low — refine signal filters."))

    # ── Drawdown ──────────────────────────────────────────
    if mean_dd <= 10:
        recs.append((G, "✓", f"Average max drawdown {mean_dd:.1f}% is well-controlled."))
    elif mean_dd <= 20:
        recs.append((Y, "→", f"Average max drawdown {mean_dd:.1f}% — monitor peak-to-trough risk."))
    else:
        recs.append((RED, "✗", f"Average max drawdown {mean_dd:.1f}% is too high — tighten stop-loss rules."))

    # ── Profitability ─────────────────────────────────────
    if prof_rate >= 0.83:
        recs.append((G, "✓", f"Profitable in {prof_rate*100:.0f}% of windows — excellent."))
    elif prof_rate >= 0.67:
        recs.append((Y, "→", f"Profitable in {prof_rate*100:.0f}% of windows — investigate underperforming periods."))
    else:
        recs.append((RED, "✗", f"Only profitable in {prof_rate*100:.0f}% of windows — serious concern."))

    # ── CV of returns ─────────────────────────────────────
    if cv_ret <= 0.5:
        recs.append((G, "✓", f"Return CV={cv_ret:.2f} — stable regime-to-regime performance."))
    elif cv_ret <= 1.0:
        recs.append((Y, "→", f"Return CV={cv_ret:.2f} — moderate variance, expected for crypto."))
    else:
        recs.append((RED, "→", f"Return CV={cv_ret:.2f} — very high variance, regime-dependent system."))

    # ── Profit Factor ─────────────────────────────────────
    if mean_pf >= 1.5:
        recs.append((G, "✓", f"Profit Factor avg={mean_pf:.2f} exceeds 1.5 target."))
    elif mean_pf >= 1.1:
        recs.append((Y, "→", f"Profit Factor avg={mean_pf:.2f} positive but marginal."))
    else:
        recs.append((RED, "✗", f"Profit Factor avg={mean_pf:.2f} — system losing on average."))

    print()
    for col, icon, text in recs:
        print(f"  {col}{icon}{R}  {text}")
    print()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"""
{C}{'═' * W}
  WALK-FORWARD OPTIMIZATION
  {WFO_CONFIG['n_windows']} windows │ {WFO_CONFIG['train_months']}M train │ {WFO_CONFIG['test_months']}M test │ {WFO_CONFIG['step_months']}M step
  Period: {WFO_CONFIG['first_test_start']} → 2024-12-31
{'═' * W}{R}
""")

    setup_all_loggers(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")
    total_t0 = time.perf_counter()

    # ══════════════════════════════════════════════════════════
    # STEP 1: Load featured data
    # ══════════════════════════════════════════════════════════
    print(f"{B}{C}[STEP 1/6]{R} Loading featured 4H data...")
    print(f"{'─' * W}")

    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if not feat_path.exists():
        print(f"{RED}  Feature cache not found — run test_features.py first.{R}")
        return

    df = pd.read_parquet(feat_path)
    print(f"  ✓ {format_number(len(df), 0)} candles  │  {len(df.columns)} columns")
    print(f"  ✓ Range: {df.index[0].date()}  →  {df.index[-1].date()}")

    # Pre-generate signal columns once — WFO windows read from cached columns
    print(f"  → Pre-generating signal distribution columns (one pass)...", end=" ", flush=True)
    _sig_t0 = time.perf_counter()
    df = SignalEngine().generate_signals_batch(df, start_idx=200)
    print(f"done ({time.perf_counter() - _sig_t0:.1f}s)  "
          f"│  signal_type coverage: {(df['signal_type'] != 'SKIP').mean():.1%} actionable")

    # ══════════════════════════════════════════════════════════
    # STEP 2: Setup engine + windows
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 2/6]{R} Setting up walk-forward engine...")
    print(f"{'─' * W}")

    engine = WalkForwardEngine(
        df=df,
        bt_config=BT_CONFIG,
        risk_config_path=RISK_CONFIG_PATH,
        **WFO_CONFIG,
    )
    windows = engine.generate_windows()
    print(f"  ✓ Generated {len(windows)} windows")
    print(f"  ✓ Warmup buffer: 70 days prepended to each test slice")
    print_window_schedule(engine)

    # ══════════════════════════════════════════════════════════
    # STEP 3: Run all windows
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 3/6]{R} Running walk-forward backtests...")
    print(f"{'─' * W}")
    print()

    # Compute buy-and-hold returns for alpha comparison (tz-safe)
    bah_returns = []
    for w in windows:
        ts, te = engine._localise_pair(w.test_start, w.test_end)
        mask = (df.index >= ts) & (df.index <= te)
        p = df[mask].get("close", df[mask].get("Close", pd.Series()))
        bah_ret = (float(p.iloc[-1]) / float(p.iloc[0]) - 1) * 100 if len(p) > 1 else 0.0
        bah_returns.append(bah_ret)

    results = []
    for i, (w, bah) in enumerate(zip(windows, bah_returns)):
        t0 = time.perf_counter()
        result = engine.run_window(w)
        results.append(result)
        print_window_result(result, bah)

    engine.results = results
    run_time = sum(r.elapsed_s for r in results)
    n_ok = sum(1 for r in results if r.ok)

    print(f"\n  ✓ {n_ok}/{len(windows)} windows completed  "
          f"│  Total run time: {format_duration(run_time)}")

    # ══════════════════════════════════════════════════════════
    # STEP 4: Per-window table
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 4/6]{R} Performance table...")
    print(f"{'─' * W}")

    print_per_window_table(results, bah_returns)

    # ══════════════════════════════════════════════════════════
    # STEP 5: Aggregation + stability
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 5/6]{R} Aggregation & stability analysis...")
    print(f"{'─' * W}")

    agg  = engine.aggregate_metrics()
    stab = engine.stability_analysis()

    print_aggregated_metrics(agg)
    print_stability_analysis(stab)
    print_recommendations(stab, agg)

    # ══════════════════════════════════════════════════════════
    # STEP 6: Save report
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 6/6]{R} Saving WFO report...")
    print(f"{'─' * W}")

    report = engine.build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine.save_report(report, str(REPORT_PATH))
    print(f"  ✓ Saved → {REPORT_PATH}")
    print(f"  ✓ Saved → {REPORT_PATH.with_suffix('.csv')}")

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    total_time = time.perf_counter() - total_t0

    score       = stab.get("overall_consistency_score", 0)
    verdict     = stab.get("verdict", "—")
    mean_return = agg.get("total_return_pct", {}).get("mean", 0)
    mean_sharpe = agg.get("sharpe_ratio", {}).get("mean", 0)
    mean_dd     = agg.get("max_drawdown_pct", {}).get("mean", 0)
    mean_bah    = agg.get("buy_hold_return_pct", {}).get("mean", 0)
    mean_alpha  = mean_return - mean_bah
    total_trades= sum(r.metrics.get("total_trades", 0) for r in results if r.ok)

    alpha_col = G if mean_alpha > 0 else RED
    badge     = _consistency_badge(score)

    print(f"""
{C}{'═' * W}
  FINAL SUMMARY — Walk-Forward Optimization
{'═' * W}{R}
  Windows run         : {n_ok} / {len(windows)}  ({WFO_CONFIG['train_months']}M train / {WFO_CONFIG['test_months']}M test)
  Period covered      : {WFO_CONFIG['first_test_start']} → 2024-12-31
  Total trades        : {total_trades:,}

  {B}Avg Return         :{R}  {_col(mean_return)}{mean_return:+.2f}%{R}
  {B}Buy & Hold avg     :{R}  {_col(mean_bah)}{mean_bah:+.2f}%{R}
  {B}Alpha (avg)        :{R}  {alpha_col}{mean_alpha:+.2f}%{R}
  {B}Avg Sharpe         :{R}  {_col(mean_sharpe)}{mean_sharpe:.3f}{R}
  {B}Avg Max Drawdown   :{R}  {_col(-mean_dd)}{mean_dd:.2f}%{R}

  {B}Consistency Score  :{R}  {score:.3f}  [{badge}]
  {B}Verdict            :{R}  {verdict}

  Total time          : {format_duration(total_time)}
  Report saved        : {REPORT_PATH}
{G}{'═' * W}
  ✓ Walk-forward optimization complete
{'═' * W}{R}
""")


if __name__ == "__main__":
    main()
