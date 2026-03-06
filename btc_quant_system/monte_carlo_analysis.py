#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  BTC QUANT SYSTEM — Monte Carlo Risk Analysis
  Runs 10 000 simulations of 500-trade sequences and produces:
    • Equity fan chart  (5 / 25 / 50 / 75 / 95th-percentile paths)
    • Final equity distribution histogram
    • Max-drawdown distribution histogram
    • Leverage sweep   (CAGR, Sharpe, Ruin Probability vs leverage)
    • YAML report      → backtests/reports/monte_carlo/mc_report.yaml
    • PNG charts       → backtests/reports/monte_carlo/
═══════════════════════════════════════════════════════════════════════════════

Usage
-----
  # Use synthetic defaults (based on WFO aggregate results):
  python3 monte_carlo_analysis.py

  # Override key trade statistics:
  python3 monte_carlo_analysis.py --win_rate 0.38 --avg_win_r 2.8 --avg_loss_r 1.0

  # Load real backtest trades (JSON list with 'pnl_r' field):
  python3 monte_carlo_analysis.py --trades_file backtests/trades/trades_2023.json

  # Custom simulation size:
  python3 monte_carlo_analysis.py --sims 5000 --n_trades 250

  # Skip chart generation (faster, useful in headless environments):
  python3 monte_carlo_analysis.py --no_charts
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from engines.monte_carlo import (
    MonteCarloEngine,
    SimResult,
    TradeStats,
    LeverageResult,
)

# ── ANSI colour codes ─────────────────────────────────────────────────────────
C   = "\033[36m"    # cyan
G   = "\033[32m"    # green
Y   = "\033[33m"    # yellow
RED = "\033[31m"    # red
B   = "\033[1m"     # bold
D   = "\033[2m"     # dim
R   = "\033[0m"     # reset

W   = 72            # terminal width
SEP = f"{C}{'═' * W}{R}"
SUB = f"  {'─' * (W - 2)}"

OUTPUT_DIR = PROJECT_ROOT / "backtests" / "reports" / "monte_carlo"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="  %(message)s",
)
logger = logging.getLogger("mc_analysis")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo risk analysis for BTC Quant System.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trades_file",  type=str, default=None,
                   help="JSON file with trade list (each trade needs 'pnl_r' key)")
    p.add_argument("--win_rate",     type=float, default=None,
                   help="Win rate override [0–1]")
    p.add_argument("--avg_win_r",    type=float, default=None,
                   help="Average win R-multiple override")
    p.add_argument("--avg_loss_r",   type=float, default=None,
                   help="Average loss R-multiple override  (positive)")
    p.add_argument("--capital",      type=float, default=100_000,
                   help="Initial capital")
    p.add_argument("--risk_pct",     type=float, default=1.0,
                   help="Base risk per trade as %% of equity  (e.g. 1 = 1 %%)")
    p.add_argument("--sims",         type=int,   default=10_000,
                   help="Number of Monte Carlo simulations")
    p.add_argument("--n_trades",     type=int,   default=500,
                   help="Trades per simulation path")
    p.add_argument("--no_charts",    action="store_true",
                   help="Skip matplotlib chart generation")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

# WFO aggregate defaults (based on 6-window walk-forward results)
_WFO_DEFAULTS = dict(
    win_rate   = 0.32,   # average across 6 OOS windows
    avg_win_r  = 2.80,   # TP3 partial exits + runners
    avg_loss_r = 1.00,   # stop-loss exits at 1R
    std_win_r  = 1.00,   # high variance in wins
    std_loss_r = 0.25,   # tighter variance in losses
)


def _load_trade_stats(args: argparse.Namespace) -> TradeStats:
    """Load trade stats from JSON file, CLI args, or WFO defaults."""

    if args.trades_file:
        path = Path(args.trades_file)
        if not path.exists():
            print(f"{RED}  [ERROR] trades_file not found: {path}{R}")
            sys.exit(1)
        with open(path) as fh:
            trades = json.load(fh)
        print(f"{D}  Loaded {len(trades)} historical trades from {path}{R}")
        stats = TradeStats.from_trades(trades)
        # Allow CLI overrides on top of real data
        if args.win_rate   is not None: stats.win_rate   = args.win_rate
        if args.avg_win_r  is not None: stats.avg_win_r  = args.avg_win_r
        if args.avg_loss_r is not None: stats.avg_loss_r = args.avg_loss_r
        return stats

    # CLI parameter overrides
    kw = dict(_WFO_DEFAULTS)
    if args.win_rate   is not None: kw["win_rate"]   = args.win_rate
    if args.avg_win_r  is not None: kw["avg_win_r"]  = args.avg_win_r
    if args.avg_loss_r is not None: kw["avg_loss_r"] = args.avg_loss_r

    return TradeStats.from_defaults(**kw)


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _pct_col(v: float, invert: bool = False) -> str:
    """ANSI colour for a percentage value."""
    pos = v > 0
    if invert:
        pos = not pos
    return G if pos else (RED if v < 0 else D)


def _equity_str(v: float, initial: float) -> str:
    ret = (v / initial - 1) * 100
    col = G if ret >= 0 else RED
    return f"${v:>12,.0f}  ({col}{ret:+.1f}%{R})"


def print_header():
    print(f"""
{C}╔{'═' * (W)}╗
║{'MONTE CARLO RISK ANALYSIS — BTC QUANT SYSTEM':^{W}}║
╚{'═' * (W)}╝{R}""")


def print_trade_stats(stats: TradeStats):
    pf_str = (f"{stats.profit_factor:.3f}" if np.isfinite(stats.profit_factor)
              else "∞")
    ex_col = G if stats.expectancy_r > 0 else RED
    pf_col = G if stats.profit_factor > 1 else RED

    print(f"""
{SEP}
  {B}TRADE STATISTICS{R}
{SUB}
  Win Rate         {G}{stats.win_rate*100:.1f}%{R}   /   Loss Rate  {RED}{stats.loss_rate*100:.1f}%{R}
  Avg Win (R)      {stats.avg_win_r:.3f}         Avg Loss (R) {stats.avg_loss_r:.3f}
  Expectancy (R)   {ex_col}{stats.expectancy_r:+.3f}{R}
  Profit Factor    {pf_col}{pf_str}{R}
  Sampling Mode    {D}{'bootstrap' if stats.win_rs is not None else 'parametric (lognormal / half-normal)'}{R}
  Historical Trades {stats.n_trades if stats.n_trades else 'N/A (synthetic)'}""")


def print_simulation_info(engine: MonteCarloEngine, result: SimResult):
    n_years = engine.n_trades / engine.trades_per_year
    print(f"""
{SEP}
  {B}SIMULATION CONFIGURATION{R}
{SUB}
  Paths            {result.n_sims:,}
  Trades / path    {result.n_trades:,}  ({n_years:.1f} years at {engine.trades_per_year} trades/yr)
  Base Leverage    {result.leverage}×
  Risk / trade     {result.risk_pct * 100:.2f}% of equity  (= {result.risk_pct * result.initial_capital:,.0f} at initial capital)
  Initial Capital  ${result.initial_capital:,.0f}
  Ruin Threshold   {engine.ruin_threshold*100:.0f}% drawdown
  Elapsed          {result.elapsed_s:.1f}s""")


def print_ruin_and_targets(engine: MonteCarloEngine, result: SimResult):
    ruin    = engine.probability_of_ruin(result)
    p2x     = engine.probability_of_target(result, 2.0)
    p3x     = engine.probability_of_target(result, 3.0)
    p5x     = engine.probability_of_target(result, 5.0)
    ruin_c  = RED if ruin > 0.10 else (Y if ruin > 0.05 else G)

    print(f"""
{SEP}
  {B}RUIN & TARGET PROBABILITIES  (1× leverage, 1% risk/trade){R}
{SUB}
  P(Ruin — DD > 50%)        {ruin_c}{B}{ruin*100:>6.2f}%{R}
  P(2× capital)             {G if p2x > 0.50 else D}{p2x*100:>6.2f}%{R}
  P(3× capital)             {G if p3x > 0.25 else D}{p3x*100:>6.2f}%{R}
  P(5× capital)             {G if p5x > 0.10 else D}{p5x*100:>6.2f}%{R}""")


def print_confidence_intervals(engine: MonteCarloEngine, result: SimResult):
    ci  = engine.confidence_intervals(result)
    C0  = result.initial_capital
    print(f"""
{SEP}
  {B}FINAL EQUITY — CONFIDENCE INTERVALS  (after {result.n_trades} trades){R}
{SUB}
  {'Percentile':<14} {'Final Equity':>18}  Return
  {'──────────':<14} {'────────────':>18}  ──────""")
    for label, key in [
        ("5th  (worst)",  "p05"),
        ("10th",          "p10"),
        ("25th",          "p25"),
        ("50th (median)", "p50"),
        ("75th",          "p75"),
        ("90th",          "p90"),
        ("95th  (best)",  "p95"),
    ]:
        v   = ci.get(key, ci.get(key.replace("p0", "p"), 0))
        ret = (v / C0 - 1) * 100
        col = G if ret >= 0 else RED
        print(f"  {label:<14} ${v:>15,.0f}  {col}{ret:+.1f}%{R}")
    print(f"  {'Mean':<14} ${ci['mean']:>15,.0f}  "
          f"{_pct_col(ci['mean']/C0-1)}{(ci['mean']/C0-1)*100:+.1f}%{R}")


def print_drawdown_distribution(engine: MonteCarloEngine, result: SimResult):
    dd = engine.drawdown_distribution(result)
    ruin_pct = engine.ruin_threshold * 100

    print(f"""
{SEP}
  {B}MAX DRAWDOWN DISTRIBUTION{R}
{SUB}
  Mean Max DD         {Y}{dd['mean']:.1f}%{R}
  Median Max DD       {Y}{dd['median']:.1f}%{R}
  75th pct Max DD     {Y}{dd['p75']:.1f}%{R}
  90th pct Max DD     {RED if dd['p90'] > ruin_pct else Y}{dd['p90']:.1f}%{R}
  95th pct Max DD     {RED if dd['p95'] > ruin_pct else Y}{dd['p95']:.1f}%{R}
  99th pct Max DD     {RED if dd['p99'] > ruin_pct else Y}{dd['p99']:.1f}%{R}
  Worst-case Max DD   {RED}{dd['max']:.1f}%{R}""")


def print_leverage_analysis(
    engine: MonteCarloEngine,
    stats: TradeStats,
    lev_results: list,
):
    kelly_lv   = engine.kelly_optimal_leverage(stats)
    kelly_frac = engine.kelly_fraction(stats)
    prac       = engine.practical_optimal_leverage(lev_results)

    prac_str = (f"{prac.leverage:.1f}×" if prac else "None (all violate constraints)")
    prac_col = G if prac else RED

    print(f"""
{SEP}
  {B}LEVERAGE ANALYSIS — KELLY vs PRACTICAL OPTIMAL{R}
{SUB}
  Kelly Full Leverage   {Y}{kelly_lv:.2f}×{R}  (Kelly fraction = {kelly_frac*100:.2f}% of equity/trade)
  Kelly Half Leverage   {G}{kelly_lv/2:.2f}×{R}  ← recommended safer target
  Practical Optimal     {prac_col}{prac_str}{R}  (max ruin ≤ 5%, Sharpe > 0)

  {B}Leverage Sweep{R}
  {'Lev':>5} {'CAGR%':>8} {'Sharpe':>8} {'Mean DD%':>10} {'P95 DD%':>9} {'Ruin%':>7} {'P(2×)':>7}
  {'─'*5:>5} {'─'*8:>8} {'─'*8:>8} {'─'*10:>10} {'─'*9:>9} {'─'*7:>7} {'─'*7:>7}""")

    for r in lev_results:
        ruin_c  = RED if r.ruin_prob > 0.10 else (Y if r.ruin_prob > 0.05 else G)
        cagr_c  = G   if r.mean_cagr_pct > 0 else RED
        sh_c    = G   if r.sharpe > 0.5 else (Y if r.sharpe > 0 else RED)
        marker  = ""
        if prac and abs(r.leverage - prac.leverage) < 0.01:
            marker = f" {G}← practical opt{R}"
        elif abs(r.leverage - kelly_lv / 2) < 0.01:
            marker = f" {Y}← half-Kelly{R}"
        print(
            f"  {r.leverage:>4.1f}× "
            f"{cagr_c}{r.mean_cagr_pct:>7.1f}%{R} "
            f"{sh_c}{r.sharpe:>8.3f}{R} "
            f"{Y}{r.mean_max_dd_pct:>9.1f}%{R} "
            f"{RED if r.p95_max_dd_pct > 50 else Y}{r.p95_max_dd_pct:>8.1f}%{R} "
            f"{ruin_c}{r.ruin_prob*100:>6.1f}%{R} "
            f"{r.target_prob_2x*100:>6.1f}%"
            f"{marker}"
        )


def print_recommendations(
    engine: MonteCarloEngine,
    stats: TradeStats,
    result: SimResult,
    lev_results: list,
):
    ruin      = engine.probability_of_ruin(result)
    kelly_lv  = engine.kelly_optimal_leverage(stats)
    prac      = engine.practical_optimal_leverage(lev_results)
    dd_median = engine.drawdown_distribution(result)["median"]
    ex        = stats.expectancy_r

    rec_lev   = prac.leverage if prac else 1.0
    rec_risk  = engine.risk_pct * rec_lev * 100

    # Viability assessment
    if ex <= 0:
        viability = f"{RED}NEGATIVE EDGE — strategy has no edge; do NOT trade live.{R}"
    elif ruin > 0.20:
        viability = f"{RED}HIGH RISK — ruin probability {ruin*100:.1f}% is unacceptable.{R}"
    elif ruin > 0.05:
        viability = f"{Y}MODERATE RISK — reduce leverage or tighten stop losses.{R}"
    else:
        viability = f"{G}VIABLE — strategy shows positive edge with acceptable risk.{R}"

    # Position sizing recommendation
    if rec_lev < 1.0:
        ps_rec = (f"Use {rec_lev:.1f}× leverage (conservative), "
                  f"risk {rec_risk:.2f}% per trade.  1× is also viable (ruin < 1%).")
    elif rec_lev == 1.0:
        ps_rec = f"Use 1× leverage, risk {engine.risk_pct*100:.1f}% per trade."
    elif rec_lev <= kelly_lv / 2:
        ps_rec = (f"Use {rec_lev:.1f}× leverage (within half-Kelly), "
                  f"risk {rec_risk:.2f}% per trade.")
    else:
        ps_rec = (f"Caution: {rec_lev:.1f}× exceeds half-Kelly ({kelly_lv/2:.1f}×). "
                  f"Consider staying at {kelly_lv/2:.1f}×.")

    print(f"""
{SEP}
  {B}RECOMMENDATIONS{R}
{SUB}
  Viability         {viability}

  Expected Outcome  {_pct_col(ex)}Expectancy {ex:+.3f}R / trade{R}
  Typical Max DD    ~{dd_median:.1f}%  (median across {result.n_sims:,} simulations)

  Position Sizing   {ps_rec}
  Kelly Guidance    Full Kelly = {kelly_lv:.2f}× | Half-Kelly = {kelly_lv/2:.2f}× (recommended cap)

  Key Action Items""")

    actions = []

    if ex <= 0:
        actions.append(f"{RED}Stop live trading — improve signal quality or tighten filters.{R}")
    elif ex < 0.1:
        actions.append(f"{Y}Marginal edge ({ex:.3f}R). Test signal score filter ≥ 9 (currently 7).{R}")

    if ruin > 0.05:
        actions.append(f"{Y}Ruin prob {ruin*100:.1f}% > 5%: reduce base risk_pct or use 1× leverage.{R}")

    if dd_median > 30:
        actions.append(f"{Y}Median max DD {dd_median:.1f}% is high — add drawdown-based position scaling.{R}")

    if kelly_lv < 1.0:
        actions.append(f"{RED}Kelly leverage {kelly_lv:.2f}× < 1: trade smaller than 1% risk per trade.{R}")
    elif kelly_lv > 5.0:
        actions.append(f"{G}Kelly supports meaningful leverage; half-Kelly ({kelly_lv/2:.2f}×) is practical.{R}")

    if not actions:
        actions.append(f"{G}Risk parameters look healthy. Continue monitoring with rolling WFO.{R}")

    for i, a in enumerate(actions, 1):
        print(f"    {i}. {a}")

    print()
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# CHART GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_charts(
    engine: MonteCarloEngine,
    stats: TradeStats,
    result: SimResult,
    lev_results: list,
    output_dir: Path,
) -> list:
    """
    Generate 4 publication-quality charts and save as PNG.
    Returns list of saved file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.patches import Patch
    except ImportError:
        print(f"  {Y}[WARN] matplotlib not installed — skipping charts.{R}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # ── Shared style ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#e6edf3",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "text.color":        "#e6edf3",
        "grid.color":        "#21262d",
        "grid.linewidth":    0.6,
        "axes.grid":         True,
        "font.family":       "monospace",
        "font.size":         9,
    })

    trade_xs = np.arange(result.n_trades + 1)
    C0       = result.initial_capital

    # ── FIGURE 1: Equity Fan Chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Monte Carlo — Equity Paths  ({result.n_sims:,} simulations, "
        f"{result.n_trades} trades/path, 1× leverage)",
        fontsize=11, color="#e6edf3", y=0.98,
    )

    # Percentile bands
    band_specs = [
        (5,  95, "#1f6feb", 0.15, "5–95th pct"),
        (10, 90, "#388bfd", 0.20, "10–90th pct"),
        (25, 75, "#58a6ff", 0.30, "25–75th pct"),
    ]
    for lo, hi, col, alpha, label in band_specs:
        lo_path = result.percentile_path(lo / 100)
        hi_path = result.percentile_path(hi / 100)
        ax.fill_between(trade_xs, lo_path, hi_path, alpha=alpha, color=col, label=label)

    # Individual paths (50 random)
    rng_vis  = np.random.default_rng(0)
    idx_vis  = rng_vis.choice(result.n_sims, size=50, replace=False)
    for i in idx_vis:
        ax.plot(trade_xs, result.equity_paths[i], color="#58a6ff", alpha=0.04, linewidth=0.4)

    # Median and mean paths
    ax.plot(trade_xs, result.percentile_path(0.50),
            color="#f0e68c", linewidth=1.8, label="Median", zorder=5)
    ax.plot(trade_xs, result.equity_paths.mean(axis=0),
            color="#ff7b72", linewidth=1.4, linestyle="--", label="Mean", zorder=5)

    ax.axhline(C0, color="#8b949e", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axhline(C0 * 0.50, color="#f85149", linewidth=1.0, linestyle="--",
               alpha=0.8, label="Ruin threshold (−50%)")

    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Portfolio Equity ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.3)
    plt.tight_layout()
    p = output_dir / "01_equity_fan.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── FIGURE 2: Final Equity Distribution ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Final Equity Distribution  ({result.n_trades} trades, 1× leverage)",
        fontsize=11, color="#e6edf3",
    )

    final = result.final_equities
    bins  = np.linspace(max(0, final.min()), min(final.max(), C0 * 6), 100)

    ax.hist(final, bins=bins, color="#388bfd", alpha=0.7, edgecolor="none", label="Final equity")

    # Confidence interval markers
    ci = engine.confidence_intervals(result)
    ci_colors = {
        "p05": ("#f85149", "5th pct"),
        "p25": ("#d29922", "25th pct"),
        "p50": ("#f0e68c", "Median"),
        "p75": ("#3fb950", "75th pct"),
        "p95": ("#58a6ff", "95th pct"),
    }
    ymax = ax.get_ylim()[1]
    for key, (col, lab) in ci_colors.items():
        v = ci.get(key, 0)
        ax.axvline(v, color=col, linewidth=1.5, linestyle="--", alpha=0.9, label=lab)

    ax.axvline(C0, color="#8b949e", linewidth=1.0, linestyle=":", alpha=0.7, label="Initial capital")
    ax.axvline(C0 * 0.50, color="#f85149", linewidth=1.2, alpha=0.6, label="Ruin (50%)")

    ax.set_xlabel("Final Portfolio Equity ($)")
    ax.set_ylabel("Frequency (simulations)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8, framealpha=0.3)
    plt.tight_layout()
    p = output_dir / "02_final_equity_dist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── FIGURE 3: Max Drawdown Distribution ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Max Drawdown Distribution  ({result.n_sims:,} simulations, 1× leverage)",
        fontsize=11, color="#e6edf3",
    )

    dd_pct = result.max_drawdowns * 100
    ax.hist(dd_pct, bins=80, color="#d29922", alpha=0.75, edgecolor="none")

    ruin_line = engine.ruin_threshold * 100
    ax.axvline(ruin_line, color="#f85149", linewidth=1.8, linestyle="--",
               label=f"Ruin threshold ({ruin_line:.0f}%)")

    for pct, col, lab in [
        (50, "#8b949e", "p50"),
        (90, "#d29922", "p90"),
        (95, "#f85149", "p95"),
    ]:
        v = np.percentile(dd_pct, pct)
        ax.axvline(v, color=col, linewidth=1.2, linestyle=":",
                   label=f"{lab} = {v:.1f}%")

    ruin_sims = (dd_pct > ruin_line).sum()
    ax.text(
        0.97, 0.95,
        f"Ruin probability\n{ruin_sims / result.n_sims * 100:.2f}%",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10, color="#f85149",
        bbox=dict(facecolor="#161b22", edgecolor="#f85149", alpha=0.8),
    )

    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Frequency (simulations)")
    ax.legend(fontsize=8, framealpha=0.3)
    plt.tight_layout()
    p = output_dir / "03_max_drawdown_dist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── FIGURE 4: Leverage Sweep ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Leverage Analysis — Risk/Reward Trade-off",
                 fontsize=11, color="#e6edf3", y=1.01)

    levs       = [r.leverage       for r in lev_results]
    cagrs      = [r.mean_cagr_pct  for r in lev_results]
    sharpes    = [r.sharpe          for r in lev_results]
    ruin_probs = [r.ruin_prob * 100 for r in lev_results]
    p95_dds    = [r.p95_max_dd_pct  for r in lev_results]

    kelly_lv   = engine.kelly_optimal_leverage(stats)
    prac       = engine.practical_optimal_leverage(lev_results)

    # Panel A: CAGR vs leverage
    ax = axes[0]
    ax.plot(levs, cagrs, color="#388bfd", linewidth=2, marker="o", markersize=4)
    ax.fill_between(levs, 0, cagrs, alpha=0.15,
                    color="#388bfd" if max(cagrs) > 0 else "#f85149")
    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    if kelly_lv <= max(levs):
        ax.axvline(kelly_lv, color="#f0e68c", linewidth=1.0, linestyle=":",
                   alpha=0.8, label=f"Full Kelly ({kelly_lv:.1f}×)")
        ax.axvline(kelly_lv / 2, color="#3fb950", linewidth=1.0, linestyle=":",
                   alpha=0.8, label=f"Half Kelly ({kelly_lv/2:.1f}×)")
    if prac:
        ax.axvline(prac.leverage, color="#f85149", linewidth=1.4, linestyle="--",
                   alpha=0.9, label=f"Practical opt ({prac.leverage:.1f}×)")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Mean CAGR (%)")
    ax.set_title("CAGR vs Leverage")
    ax.legend(fontsize=7, framealpha=0.3)

    # Panel B: Sharpe vs leverage
    ax = axes[1]
    ax.plot(levs, sharpes, color="#3fb950", linewidth=2, marker="o", markersize=4)
    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color="#3fb950", linewidth=0.8, linestyle=":", alpha=0.6,
               label="Sharpe = 1.0")
    if kelly_lv <= max(levs):
        ax.axvline(kelly_lv / 2, color="#3fb950", linewidth=1.0, linestyle=":",
                   alpha=0.7, label=f"Half Kelly ({kelly_lv/2:.1f}×)")
    if prac:
        ax.axvline(prac.leverage, color="#f85149", linewidth=1.4, linestyle="--",
                   alpha=0.9, label=f"Practical opt ({prac.leverage:.1f}×)")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Median Sharpe Ratio")
    ax.set_title("Sharpe vs Leverage")
    ax.legend(fontsize=7, framealpha=0.3)

    # Panel C: Ruin prob + P95 DD vs leverage
    ax = axes[2]
    ax2 = ax.twinx()
    l1, = ax.plot(levs, ruin_probs, color="#f85149", linewidth=2,
                  marker="o", markersize=4, label="Ruin prob (%)")
    l2, = ax2.plot(levs, p95_dds, color="#d29922", linewidth=2,
                   marker="s", markersize=4, linestyle="--", label="P95 Max DD (%)")
    ax.axhline(5.0, color="#f85149", linewidth=0.8, linestyle=":", alpha=0.7,
               label="5% ruin threshold")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Ruin Probability (%)", color="#f85149")
    ax2.set_ylabel("P95 Max Drawdown (%)", color="#d29922")
    ax.set_title("Risk vs Leverage")
    ax2.tick_params(axis="y", colors="#d29922")
    ax.tick_params(axis="y", colors="#f85149")
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7, framealpha=0.3)

    plt.tight_layout()
    p = output_dir / "04_leverage_sweep.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = _parse_args()

    print_header()

    # ── Step 1: Trade statistics ──────────────────────────────────────────────
    print(f"\n  {D}Step 1/5  Loading trade statistics…{R}")
    stats = _load_trade_stats(args)
    print_trade_stats(stats)

    # ── Step 2: Base simulation (1× leverage) ─────────────────────────────────
    print(f"\n  {D}Step 2/5  Running {args.sims:,} Monte Carlo simulations…{R}", flush=True)

    engine = MonteCarloEngine(
        n_simulations  = args.sims,
        n_trades       = args.n_trades,
        initial_capital= args.capital,
        risk_pct       = args.risk_pct / 100.0,
        ruin_threshold = 0.50,
        trades_per_year= 250,
        seed           = 42,
    )

    t0     = time.perf_counter()
    result = engine.simulate(stats, leverage=1.0)
    print(f"  {G}Done — {result.n_sims:,} paths × {result.n_trades} trades in {result.elapsed_s:.1f}s{R}")

    print_simulation_info(engine, result)
    print_ruin_and_targets(engine, result)
    print_confidence_intervals(engine, result)
    print_drawdown_distribution(engine, result)

    # ── Step 3: Leverage sweep ─────────────────────────────────────────────────
    print(f"\n  {D}Step 3/5  Leverage sweep (0.5× → 10×, 3 000 sims each)…{R}", flush=True)
    t1 = time.perf_counter()
    lev_results = engine.leverage_sweep(
        stats,
        leverages      = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0),
        n_sims_per_lev = 3_000,
    )
    print(f"  {G}Done in {time.perf_counter()-t1:.1f}s{R}")

    print_leverage_analysis(engine, stats, lev_results)
    print_recommendations(engine, stats, result, lev_results)

    # ── Step 4: Save YAML report ───────────────────────────────────────────────
    print(f"  {D}Step 4/5  Building and saving YAML report…{R}", flush=True)
    report     = engine.build_report(stats, result, lev_results)
    report_dir = OUTPUT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "mc_report.yaml"
    engine.save_report(report, report_path)
    print(f"  {G}Report → {report_path}{R}")

    # ── Step 5: Charts ─────────────────────────────────────────────────────────
    if args.no_charts:
        print(f"\n  {D}Step 5/5  Charts skipped (--no_charts).{R}")
    else:
        print(f"\n  {D}Step 5/5  Generating charts…{R}", flush=True)
        saved = generate_charts(engine, stats, result, lev_results, report_dir)
        if saved:
            for p in saved:
                print(f"  {G}Chart  → {p}{R}")
        else:
            print(f"  {Y}No charts generated.{R}")

    total = time.perf_counter() - t0
    print(f"\n  {B}{G}Monte Carlo analysis complete  ({total:.1f}s total){R}\n")


if __name__ == "__main__":
    main()
