#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  BACKTEST ANALYSIS & VISUALIZATION
  Full performance breakdown + 6 charts
═══════════════════════════════════════════════════════════════
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "axes.grid": True,
    "grid.color": "#222",
    "grid.alpha": 0.5,
    "text.color": "#ccc",
    "xtick.color": "#999",
    "ytick.color": "#999",
    "font.size": 10,
    "font.family": "sans-serif",
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#333",
    "legend.fontsize": 9,
})
CYAN  = "#00d4ff"
GREEN = "#00ff88"
RED   = "#ff4444"
GOLD  = "#ffd700"
GRAY  = "#888888"
PURPLE = "#bb86fc"

# ── ANSI ─────────────────────────────────────────────────────
A_C   = "\033[36m"
A_G   = "\033[32m"
A_Y   = "\033[33m"
A_RED = "\033[31m"
A_B   = "\033[1m"
A_D   = "\033[2m"
A_R   = "\033[0m"
SEP   = f"{A_C}{'═' * 60}{A_R}"
SUB   = f"  {'─' * 56}"

# ── Paths ────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
REPORTS_DIR = PROJECT_ROOT / "backtests" / "reports"
CHARTS_DIR  = REPORTS_DIR / "charts"


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load backtest results, trades CSV, equity curve."""
    # JSON results
    json_path = RESULTS_DIR / "backtest_2023_2024.json"
    with open(json_path) as f:
        results = json.load(f)

    # Trades
    trades_df = pd.read_csv(RESULTS_DIR / "trades.csv")

    # Equity curve
    eq_df = pd.read_csv(RESULTS_DIR / "equity_curve.csv")
    eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
    eq_df = eq_df.set_index("timestamp")

    # BTC price for Buy & Hold
    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    btc_df = pd.read_parquet(feat_path)
    btc_test = btc_df[
        (btc_df.index >= "2023-01-01") & (btc_df.index <= "2024-12-31")
    ][["Close"]].copy()

    return results, trades_df, eq_df, btc_test


# ═══════════════════════════════════════════════════════════════
# PART 1 — DETAILED METRICS
# ═══════════════════════════════════════════════════════════════

def part1_detailed_metrics(results, trades_df, eq_df, btc_test):
    """Print comprehensive metrics to console."""
    print(f"\n{SEP}")
    print(f"{A_C}  PART 1 — DETAILED PERFORMANCE METRICS{A_R}")
    print(SEP)

    trades = trades_df.to_dict("records")

    # ── 1. Performance vs Targets ────────────────────────
    print(f"\n  {A_B}1. PERFORMANCE VS TARGETS{A_R}")
    print(SUB)

    metrics = [
        ("Total Return",  results.get("total_return_pct", 0),  0,   "%",  True),
        ("CAGR",           results.get("cagr_pct", 0),          20,  "%",  True),
        ("Win Rate",       results.get("win_rate_pct", 0),      55,  "%",  True),
        ("Sharpe Ratio",   results.get("sharpe_ratio", 0),      1.5, "",   True),
        ("Sortino Ratio",  results.get("sortino_ratio", 0),     2.0, "",   True),
        ("Calmar Ratio",   results.get("calmar_ratio", 0),      1.0, "",   True),
        ("Max Drawdown",   results.get("max_drawdown_pct", 0),  15,  "%",  False),
        ("Profit Factor",  results.get("profit_factor", 0),     1.5, "",   True),
        ("Expectancy",     results.get("expectancy_r", 0),      0.3, "R",  True),
    ]

    print(f"\n  {'Metric':<18s} {'Value':>10s} {'Target':>10s} {'Status':>8s}")
    print(f"  {'─'*50}")
    for name, val, target, unit, higher in metrics:
        ok = val >= target if higher else val <= target
        col = A_G if ok else A_RED
        mark = "✅" if ok else "❌"
        print(
            f"  {name:<18s} "
            f"{col}{val:>9.2f}{unit}{A_R}  "
            f"{'>' if higher else '<'}{target:.1f}{unit}  "
            f"{mark}"
        )

    passed = sum(1 for _, v, t, _, h in metrics if (v >= t if h else v <= t))
    print(f"\n  Scorecard: {A_B}{passed}/{len(metrics)}{A_R} targets met")

    # ── 2. Trade Analysis ────────────────────────────────
    print(f"\n  {A_B}2. TRADE ANALYSIS{A_R}")
    print(SUB)

    sorted_by_pnl = sorted(trades, key=lambda t: t["pnl"], reverse=True)

    print(f"\n  {A_B}Best 5 Trades:{A_R}")
    for t in sorted_by_pnl[:5]:
        d = t.get("direction", "?")
        print(
            f"    {t['id']:<15s} {d:<5s} "
            f"{A_G}${t['pnl']:>9,.2f}  {t.get('pnl_r',0):>5.2f}R{A_R}  "
            f"{t.get('exit_reason','?')}"
        )

    print(f"\n  {A_B}Worst 5 Trades:{A_R}")
    for t in sorted_by_pnl[-5:]:
        d = t.get("direction", "?")
        print(
            f"    {t['id']:<15s} {d:<5s} "
            f"{A_RED}${t['pnl']:>9,.2f}  {t.get('pnl_r',0):>5.2f}R{A_R}  "
            f"{t.get('exit_reason','?')}"
        )

    # Streaks
    max_win, max_loss, cur_w, cur_l = 0, 0, 0, 0
    for t in trades:
        if t["pnl"] > 0:
            cur_w += 1; cur_l = 0
            max_win = max(max_win, cur_w)
        else:
            cur_l += 1; cur_w = 0
            max_loss = max(max_loss, cur_l)

    print(f"\n  Longest winning streak: {A_G}{max_win}{A_R}")
    print(f"  Longest losing streak:  {A_RED}{max_loss}{A_R}")

    # Holding period
    if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
        trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_dt"]  = pd.to_datetime(trades_df["exit_time"])
        trades_df["holding_hours"] = (
            trades_df["exit_dt"] - trades_df["entry_dt"]
        ).dt.total_seconds() / 3600
        valid = trades_df["holding_hours"].dropna()
        if len(valid) > 0:
            avg_h = valid.mean()
            print(f"  Avg holding period:     {avg_h:.1f} hours ({avg_h/24:.1f} days)")

    # Trades per month
    if "entry_dt" in trades_df.columns:
        trades_df["month"] = trades_df["entry_dt"].dt.to_period("M")
        monthly = trades_df.groupby("month").size()
        print(f"\n  {A_B}Trades per Month:{A_R}")
        for m, cnt in monthly.items():
            print(f"    {str(m):<10s}: {cnt:>3d} trades")

    # ── 3. Signal Analysis ───────────────────────────────
    print(f"\n  {A_B}3. SIGNAL ANALYSIS{A_R}")
    print(SUB)

    # Direction distribution
    if "direction" in trades_df.columns:
        for d in ["LONG", "SHORT"]:
            sub = trades_df[trades_df["direction"] == d]
            if len(sub) == 0:
                continue
            wr = (sub["pnl"] > 0).mean() * 100
            avg_pnl = sub["pnl"].mean()
            col = A_G if wr >= 50 else A_Y
            print(
                f"  {d:<14s}: {len(sub):>3d} trades, "
                f"{col}WR={wr:.1f}%{A_R}, "
                f"Avg PnL=${avg_pnl:,.2f}"
            )

    # Exit reason analysis
    if "exit_reason" in trades_df.columns:
        print(f"\n  {A_B}Exit Reason Breakdown:{A_R}")
        for reason, grp in trades_df.groupby("exit_reason"):
            wr = (grp["pnl"] > 0).mean() * 100
            tot = grp["pnl"].sum()
            col = A_G if tot > 0 else A_RED
            print(
                f"    {str(reason):<14s}: {len(grp):>3d} trades, "
                f"WR={wr:.1f}%, "
                f"{col}Total=${tot:>10,.2f}{A_R}"
            )

    # ── 4. Time Analysis ─────────────────────────────────
    print(f"\n  {A_B}4. TIME ANALYSIS{A_R}")
    print(SUB)

    if "entry_dt" in trades_df.columns:
        trades_df["hour"] = trades_df["entry_dt"].dt.hour
        trades_df["dow"]  = trades_df["entry_dt"].dt.day_name()

        # By hour
        print(f"\n  {A_B}Performance by Hour (UTC):{A_R}")
        for h, grp in trades_df.groupby("hour"):
            if len(grp) == 0: continue
            avg = grp["pnl"].mean()
            col = A_G if avg > 0 else A_RED
            bar = "█" * max(1, int(abs(avg) / 200))
            print(f"    {h:02d}:00  {len(grp):>3d} trades  {col}Avg=${avg:>8,.2f}  {bar}{A_R}")

        # By day of week
        print(f"\n  {A_B}Performance by Day:{A_R}")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        for day in day_order:
            grp = trades_df[trades_df["dow"] == day]
            if len(grp) == 0: continue
            avg = grp["pnl"].mean()
            col = A_G if avg > 0 else A_RED
            print(f"    {day:<10s}  {len(grp):>3d} trades  {col}Avg=${avg:>8,.2f}{A_R}")

    return passed, len(metrics)


# ═══════════════════════════════════════════════════════════════
# PART 2 — CHARTS
# ═══════════════════════════════════════════════════════════════

def part2_charts(results, trades_df, eq_df, btc_test):
    """Generate 6 PNG charts."""
    print(f"\n{SEP}")
    print(f"{A_C}  PART 2 — GENERATING CHARTS{A_R}")
    print(SEP)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    # ── 1. Equity Curve + Drawdown + Buy & Hold ──────────

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    height_ratios=[3, 1],
                                    sharex=True)
    fig.suptitle("Equity Curve vs Buy & Hold  │  2023-2024", color=CYAN,
                 fontsize=14, fontweight="bold")

    # Strategy equity
    ax1.plot(eq_df.index, eq_df["equity"], color=CYAN, linewidth=1.5,
             label="Strategy", zorder=3)

    # Buy & Hold
    if len(btc_test) > 0:
        init_cap = 100_000
        btc_close = btc_test["Close"]
        bh_equity = init_cap * btc_close / btc_close.iloc[0]
        ax1.plot(bh_equity.index, bh_equity.values, color=GOLD, linewidth=1.2,
                 alpha=0.8, linestyle="--", label="Buy & Hold")

    ax1.axhline(100_000, color=GRAY, linestyle=":", alpha=0.4, linewidth=0.8)
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Drawdown
    eq_vals = eq_df["equity"]
    rolling_max = eq_vals.expanding().max()
    drawdown = (eq_vals - rolling_max) / rolling_max * 100
    ax2.fill_between(eq_df.index, drawdown, 0, color=RED, alpha=0.4)
    ax2.plot(eq_df.index, drawdown, color=RED, linewidth=0.8)
    ax2.axhline(-15, color=GOLD, linestyle="--", alpha=0.6, linewidth=0.8,
                label="Max DD Limit (-15%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("")
    ax2.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    path = CHARTS_DIR / "equity_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(("equity_curve.png", "Equity + DD + B&H"))
    print(f"  ✓ {path.name}")

    # ── 2. Monthly Returns Heatmap ───────────────────────

    returns = eq_df["equity"].pct_change().dropna()
    returns.index = pd.to_datetime(returns.index)
    monthly_ret = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    months = monthly_ret.index
    years = sorted(set(months.year))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    grid = np.full((len(years), 12), np.nan)
    for dt, val in monthly_ret.items():
        yi = years.index(dt.year)
        mi = dt.month - 1
        grid[yi, mi] = val

    fig, ax = plt.subplots(figsize=(12, 3))
    fig.suptitle("Monthly Returns Heatmap (%)", color=CYAN,
                 fontsize=13, fontweight="bold")

    vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 1)
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years])

    for yi in range(len(years)):
        for mi in range(12):
            val = grid[yi, mi]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(mi, yi, f"{val:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Return %")
    fig.tight_layout()
    path = CHARTS_DIR / "monthly_returns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(("monthly_returns.png", "Monthly heatmap"))
    print(f"  ✓ {path.name}")

    # ── 3. Trade PnL Distribution ────────────────────────

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Trade PnL Distribution", color=CYAN,
                 fontsize=13, fontweight="bold")

    pnl_vals = trades_df["pnl"].values
    wins = pnl_vals[pnl_vals > 0]
    losses = pnl_vals[pnl_vals <= 0]

    bins = np.linspace(pnl_vals.min() - 500, pnl_vals.max() + 500, 25)
    ax.hist(wins, bins=bins, color=GREEN, alpha=0.7,
            label=f"Wins ({len(wins)})", edgecolor="#004d40")
    ax.hist(losses, bins=bins, color=RED, alpha=0.7,
            label=f"Losses ({len(losses)})", edgecolor="#4a0000")

    ax.axvline(0, color="white", linestyle="--", alpha=0.5)
    ax.axvline(pnl_vals.mean(), color=GOLD, linestyle="--", alpha=0.7,
               label=f"Mean: ${pnl_vals.mean():,.0f}")
    ax.set_xlabel("PnL ($)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.tight_layout()
    path = CHARTS_DIR / "trade_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(("trade_distribution.png", "PnL histogram"))
    print(f"  ✓ {path.name}")

    # ── 4. Drawdown Analysis ─────────────────────────────

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle("Drawdown Analysis", color=CYAN,
                 fontsize=13, fontweight="bold")

    ax.fill_between(eq_df.index, drawdown, 0, color=RED, alpha=0.3)
    ax.plot(eq_df.index, drawdown, color=RED, linewidth=1)

    # Highlight max DD
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax.annotate(
        f"Max DD: {max_dd_val:.1f}%",
        xy=(max_dd_idx, max_dd_val),
        xytext=(max_dd_idx + pd.Timedelta(days=30), max_dd_val + 2),
        arrowprops=dict(arrowstyle="->", color=GOLD),
        color=GOLD, fontsize=10, fontweight="bold",
    )
    ax.axhline(-15, color=GOLD, linestyle="--", alpha=0.6, linewidth=1,
               label="DD Limit (-15%)")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")

    fig.tight_layout()
    path = CHARTS_DIR / "drawdown_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(("drawdown_analysis.png", "DD timeline"))
    print(f"  ✓ {path.name}")

    # ── 5. Signal Performance ────────────────────────────

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Performance by Direction", color=CYAN,
                 fontsize=13, fontweight="bold")

    dirs = ["LONG", "SHORT"]
    dir_colors = [GREEN, RED]

    # Win rate by direction
    wr_vals = []
    cnt_vals = []
    for d in dirs:
        sub = trades_df[trades_df["direction"] == d]
        wr_vals.append((sub["pnl"] > 0).mean() * 100 if len(sub) > 0 else 0)
        cnt_vals.append(len(sub))

    bars = ax1.bar(dirs, wr_vals, color=dir_colors, alpha=0.7, edgecolor="#333")
    ax1.axhline(50, color=GOLD, linestyle="--", alpha=0.5, label="50%")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Win Rate by Direction", color="#ccc", fontsize=11)
    for bar, wr, cnt in zip(bars, wr_vals, cnt_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{wr:.1f}%\n({cnt})", ha="center", fontsize=10, color="#ccc")
    ax1.legend()

    # Avg PnL by direction
    avg_pnl_vals = []
    for d in dirs:
        sub = trades_df[trades_df["direction"] == d]
        avg_pnl_vals.append(sub["pnl"].mean() if len(sub) > 0 else 0)

    colors = [GREEN if v > 0 else RED for v in avg_pnl_vals]
    bars = ax2.bar(dirs, avg_pnl_vals, color=colors, alpha=0.7, edgecolor="#333")
    ax2.axhline(0, color=GRAY, linestyle="-", alpha=0.3)
    ax2.set_ylabel("Avg PnL ($)")
    ax2.set_title("Avg PnL by Direction", color="#ccc", fontsize=11)
    for bar, val in zip(bars, avg_pnl_vals):
        y_off = 50 if val >= 0 else -150
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_off,
                 f"${val:,.0f}", ha="center", fontsize=10, color="#ccc")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.tight_layout()
    path = CHARTS_DIR / "signal_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(("signal_performance.png", "Direction breakdown"))
    print(f"  ✓ {path.name}")

    # ── 6. Rolling Metrics ───────────────────────────────

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Rolling Performance Metrics  │  30-Trade Window",
                 color=CYAN, fontsize=13, fontweight="bold")

    # Compute rolling metrics from trade sequence
    pnl_series = trades_df["pnl"].values
    pnl_r_series = trades_df["pnl_r"].values
    window = min(20, len(pnl_series) - 1)

    if window > 0 and len(trades_df) > window:
        # Parse entry times
        trade_dates = pd.to_datetime(trades_df["entry_time"])

        # Rolling win rate
        rolling_wr = pd.Series(pnl_series > 0, dtype=float).rolling(window).mean() * 100
        ax1.plot(trade_dates, rolling_wr, color=CYAN, linewidth=1.5,
                 label=f"Rolling WR ({window}-trade)")
        ax1.axhline(50, color=GOLD, linestyle="--", alpha=0.5, label="50%")
        ax1.set_ylabel("Win Rate (%)")
        ax1.legend(loc="upper right")
        ax1.set_ylim(0, 100)

        # Cumulative PnL
        cum_pnl = np.cumsum(pnl_series)
        ax2.plot(trade_dates, cum_pnl, color=GREEN if cum_pnl[-1] > 0 else RED,
                 linewidth=1.5, label="Cumulative PnL")
        ax2.fill_between(trade_dates, cum_pnl, 0,
                         where=cum_pnl >= 0, color=GREEN, alpha=0.1)
        ax2.fill_between(trade_dates, cum_pnl, 0,
                         where=cum_pnl < 0, color=RED, alpha=0.1)
        ax2.axhline(0, color=GRAY, linestyle="-", alpha=0.3)
        ax2.set_ylabel("Cumulative PnL ($)")
        ax2.set_xlabel("")
        ax2.legend(loc="upper left")
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.tight_layout()
    path = CHARTS_DIR / "rolling_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(("rolling_metrics.png", "Rolling WR + Cum PnL"))
    print(f"  ✓ {path.name}")

    print(f"\n  {A_G}✓ All 6 charts saved to {CHARTS_DIR.relative_to(PROJECT_ROOT)}/{A_R}")
    return generated


# ═══════════════════════════════════════════════════════════════
# PART 3 — BUY & HOLD COMPARISON
# ═══════════════════════════════════════════════════════════════

def part3_buy_hold(results, eq_df, btc_test):
    """Compare strategy vs buy & hold."""
    print(f"\n{SEP}")
    print(f"{A_C}  PART 3 — STRATEGY vs BUY & HOLD{A_R}")
    print(SEP)

    init_cap = 100_000
    btc_close = btc_test["Close"]
    bh_equity = init_cap * btc_close / btc_close.iloc[0]

    # BH metrics
    bh_final = bh_equity.iloc[-1]
    bh_return = (bh_final - init_cap) / init_cap * 100
    bh_returns = btc_close.pct_change().dropna()

    bh_roll_max = bh_equity.expanding().max()
    bh_drawdown = ((bh_equity - bh_roll_max) / bh_roll_max).min() * 100

    bh_excess = bh_returns - (0.05 / 252)
    bh_sharpe = float(np.sqrt(252) * bh_excess.mean() / bh_excess.std()) if bh_excess.std() > 0 else 0

    days = (btc_close.index[-1] - btc_close.index[0]).days
    years = days / 365.25
    bh_cagr = ((bh_final / init_cap) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Strategy metrics
    strat_ret = results.get("total_return_pct", 0)
    strat_cagr = results.get("cagr_pct", 0)
    strat_dd = results.get("max_drawdown_pct", 0)
    strat_sharpe = results.get("sharpe_ratio", 0)

    comparison = {
        "bh_return": bh_return,
        "bh_cagr": bh_cagr,
        "bh_dd": abs(bh_drawdown),
        "bh_sharpe": bh_sharpe,
        "bh_final": bh_final,
    }

    print(f"""
  {'Metric':<18s} {'Strategy':>12s} {'Buy & Hold':>12s} {'Difference':>12s}
  {'─'*56}""")

    rows = [
        ("Total Return",  strat_ret,     bh_return,     "%"),
        ("CAGR",           strat_cagr,    bh_cagr,       "%"),
        ("Max Drawdown",   strat_dd,      abs(bh_drawdown), "%"),
        ("Sharpe Ratio",   strat_sharpe,  bh_sharpe,     ""),
    ]

    for name, strat, bh, unit in rows:
        diff = strat - bh
        col = A_G if diff > 0 else A_RED
        if name == "Max Drawdown":
            col = A_G if diff < 0 else A_RED  # Lower is better
        print(
            f"  {name:<18s} "
            f"{strat:>11.2f}{unit}  "
            f"{bh:>11.2f}{unit}  "
            f"{col}{diff:>+11.2f}{unit}{A_R}"
        )

    print(f"""
  {A_B}BTC Price:{A_R}  ${btc_close.iloc[0]:,.2f} → ${btc_close.iloc[-1]:,.2f}  ({bh_return:+.1f}%)
  {A_B}B&H Equity:{A_R} ${init_cap:,.2f} → ${bh_final:,.2f}
""")

    return comparison


# ═══════════════════════════════════════════════════════════════
# PART 4 — SAVE REPORTS
# ═══════════════════════════════════════════════════════════════

def part4_save_reports(results, trades_df, comparison, charts, score):
    """Save markdown reports."""
    print(f"\n{SEP}")
    print(f"{A_C}  PART 4 — SAVING REPORTS{A_R}")
    print(SEP)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Performance Report ───────────────────────────────

    passed, total = score

    report = f"""# BTC Backtest Performance Report
## Period: 2023-01-01 → 2024-12-31 (Out-of-Sample)

### Scorecard: {passed}/{total} targets met

### Capital
| Metric | Value |
|--------|-------|
| Initial Capital | ${results['initial_capital']:,.2f} |
| Final Equity | ${results['final_equity']:,.2f} |
| Total P&L | ${results['total_pnl']:,.2f} |
| Total Return | {results['total_return_pct']:.2f}% |
| CAGR | {results.get('cagr_pct', 0):.2f}% |

### Trade Statistics
| Metric | Value |
|--------|-------|
| Total Trades | {results['total_trades']} |
| Winning | {results['winning_trades']} |
| Losing | {results['losing_trades']} |
| Win Rate | {results['win_rate_pct']:.1f}% |
| Avg Win | ${results['avg_win_usd']:,.2f} ({results['avg_win_r']:.2f}R) |
| Avg Loss | ${results['avg_loss_usd']:,.2f} ({results['avg_loss_r']:.2f}R) |
| Max Consec Losses | {results.get('max_consecutive_losses', 0)} |

### Risk Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Sharpe Ratio | {results['sharpe_ratio']:.2f} | >1.5 | {'✅' if results['sharpe_ratio'] >= 1.5 else '❌'} |
| Sortino Ratio | {results['sortino_ratio']:.2f} | >2.0 | {'✅' if results['sortino_ratio'] >= 2.0 else '❌'} |
| Calmar Ratio | {results['calmar_ratio']:.2f} | >1.0 | {'✅' if results['calmar_ratio'] >= 1.0 else '❌'} |
| Max Drawdown | {results['max_drawdown_pct']:.1f}% | <15% | {'✅' if results['max_drawdown_pct'] <= 15 else '❌'} |
| Profit Factor | {results['profit_factor']:.2f} | >1.5 | {'✅' if results['profit_factor'] >= 1.5 else '❌'} |
| Expectancy | {results['expectancy_r']:.2f}R | >0.3R | {'✅' if results['expectancy_r'] >= 0.3 else '❌'} |

### Strategy vs Buy & Hold
| Metric | Strategy | Buy & Hold | Diff |
|--------|----------|------------|------|
| Total Return | {results['total_return_pct']:.2f}% | {comparison['bh_return']:.2f}% | {results['total_return_pct'] - comparison['bh_return']:+.2f}% |
| CAGR | {results.get('cagr_pct',0):.2f}% | {comparison['bh_cagr']:.2f}% | {results.get('cagr_pct',0) - comparison['bh_cagr']:+.2f}% |
| Max Drawdown | {results['max_drawdown_pct']:.1f}% | {comparison['bh_dd']:.1f}% | {results['max_drawdown_pct'] - comparison['bh_dd']:+.1f}% |
| Sharpe | {results['sharpe_ratio']:.2f} | {comparison['bh_sharpe']:.2f} | {results['sharpe_ratio'] - comparison['bh_sharpe']:+.2f} |

### Charts
"""
    for fname, desc in charts:
        report += f"- **{desc}**: `charts/{fname}`\n"

    path = REPORTS_DIR / "performance_report.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"  ✓ {path.relative_to(PROJECT_ROOT)}")

    # ── Trade Analysis Report ────────────────────────────

    trades = trades_df.to_dict("records")
    sorted_trades = sorted(trades, key=lambda t: t["pnl"], reverse=True)

    trade_report = """# Trade Analysis Report
## Period: 2023-2024

### Best 10 Trades
| ID | Dir | Entry | Exit | PnL | R | Reason |
|----|-----|-------|------|-----|---|--------|
"""
    for t in sorted_trades[:10]:
        d = t.get("direction", "?")
        r = t.get("exit_reason", "?")
        trade_report += (
            f"| {t['id']} | {d} | ${t['entry_price']:,.2f} | "
            f"${t['exit_price']:,.2f} | ${t['pnl']:,.2f} | "
            f"{t.get('pnl_r',0):.2f}R | {r} |\n"
        )

    trade_report += """
### Worst 10 Trades
| ID | Dir | Entry | Exit | PnL | R | Reason |
|----|-----|-------|------|-----|---|--------|
"""
    for t in sorted_trades[-10:]:
        d = t.get("direction", "?")
        r = t.get("exit_reason", "?")
        trade_report += (
            f"| {t['id']} | {d} | ${t['entry_price']:,.2f} | "
            f"${t['exit_price']:,.2f} | ${t['pnl']:,.2f} | "
            f"{t.get('pnl_r',0):.2f}R | {r} |\n"
        )

    # All trades
    trade_report += """
### All Trades
| ID | Dir | Entry | Exit | PnL | R | Reason |
|----|-----|-------|------|-----|---|--------|
"""
    for t in trades:
        d = t.get("direction", "?")
        r = t.get("exit_reason", "?")
        trade_report += (
            f"| {t['id']} | {d} | ${t['entry_price']:,.2f} | "
            f"${t['exit_price']:,.2f} | ${t['pnl']:,.2f} | "
            f"{t.get('pnl_r',0):.2f}R | {r} |\n"
        )

    path = REPORTS_DIR / "trade_analysis.md"
    with open(path, "w") as f:
        f.write(trade_report)
    print(f"  ✓ {path.relative_to(PROJECT_ROOT)}")

    print(f"\n  {A_G}✓ All reports saved to {REPORTS_DIR.relative_to(PROJECT_ROOT)}/{A_R}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"""
{A_C}╔{'═' * 58}╗
║{'BACKTEST ANALYSIS & VISUALIZATION':^58s}║
║{'BTC 2023-2024  │  Out-of-Sample':^58s}║
╚{'═' * 58}╝{A_R}
""")
    t0 = time.perf_counter()

    # Load
    results, trades_df, eq_df, btc_test = load_data()
    print(f"  Loaded: {len(trades_df)} trades, {len(eq_df)} equity points")

    # Part 1: Detailed metrics
    score = part1_detailed_metrics(results, trades_df, eq_df, btc_test)

    # Part 2: Charts
    charts = part2_charts(results, trades_df, eq_df, btc_test)

    # Part 3: Buy & Hold comparison
    comparison = part3_buy_hold(results, eq_df, btc_test)

    # Part 4: Save reports
    part4_save_reports(results, trades_df, comparison, charts, score)

    elapsed = time.perf_counter() - t0
    print(f"""
{A_G}{'═' * 60}
  ✓ Analysis Complete in {elapsed:.2f}s
  Charts: {CHARTS_DIR.relative_to(PROJECT_ROOT)}/
  Reports: {REPORTS_DIR.relative_to(PROJECT_ROOT)}/
{'═' * 60}{A_R}
""")


if __name__ == "__main__":
    main()
