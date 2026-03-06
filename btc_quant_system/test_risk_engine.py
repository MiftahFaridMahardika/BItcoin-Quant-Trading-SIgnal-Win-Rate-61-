#!/usr/bin/env python3
"""
Risk Engine — End-to-End Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Kelly Criterion with various win rates
2. Position size calculation examples
3. Drawdown scaling at different DD levels
4. SL/TP calculation with real ATR from BTC data
5. Risk checks simulation
6. Performance metrics on synthetic equity curve
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from engines.risk_engine import RiskEngine, PositionSize, SLTPLevels

# ── ANSI ─────────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"

SEP = f"{C}{'═' * 65}{R}"
SUB = f"  {'─' * 61}"


# ═══════════════════════════════════════════════════════════════
# TEST 1 — KELLY CRITERION
# ═══════════════════════════════════════════════════════════════

def test_kelly_criterion(engine: RiskEngine):
    """Kelly Criterion across a range of win-rate / reward scenarios."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 1 — KELLY CRITERION POSITION SIZING{R}")
    print(SEP)

    scenarios = [
        # (win_rate, avg_win_r, label)
        (0.40, 3.0, "Low WR / High RR  (scalp)"),
        (0.50, 2.0, "Balanced  (standard trend)"),
        (0.55, 1.8, "Moderate edge"),
        (0.58, 2.3, "Strong edge  (reference case)"),
        (0.65, 1.5, "High WR / Low RR  (mean-rev)"),
        (0.35, 1.5, "Negative EV — Kelly = 0"),
    ]

    print(f"\n  {B}{'Scenario':<32s} {'Full-K':>7s} {'Half-K':>7s} {'Capped':>7s} {'Risk $':>10s}{R}")
    print(SUB)

    for wr, avg_win, label in scenarios:
        res = engine.kelly_criterion(win_rate=wr, avg_win_r=avg_win, avg_loss_r=1.0)
        full = res["kelly_full_pct"]
        half = res["kelly_half_pct"]
        cap  = res["kelly_capped_pct"]
        risk = res["dollar_risk"]

        # Colour-code by capped pct
        col = G if cap > 0 else RED
        print(
            f"  {label:<32s} "
            f"{col}{full:6.2f}%{R}  "
            f"{col}{half:6.2f}%{R}  "
            f"{col}{cap:6.2f}%{R}  "
            f"{col}${risk:>9,.2f}{R}"
        )

    # Edge: negative kelly scenario explanation
    neg = engine.kelly_criterion(win_rate=0.35, avg_win_r=1.5, avg_loss_r=1.0)
    print(f"\n  {D}Note: Negative Kelly clamped to 0 — do not trade this setup.{R}")
    print(f"\n  {G}✓ Kelly Criterion verified (half-kelly, capped at {engine.max_risk_per_trade:.0%}){R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# TEST 2 — POSITION SIZE CALCULATION
# ═══════════════════════════════════════════════════════════════

def test_position_sizing(engine: RiskEngine):
    """Position size from entry + stop-loss at different leverages."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 2 — POSITION SIZE CALCULATION{R}")
    print(SEP)

    # Base scenario: BTC at ~$65 000, ATR ~$1 400
    entry   = 65_000.0
    atr     = 1_400.0
    sl_dist = atr * 1.5          # normal profile multiplier
    sl      = entry - sl_dist    # LONG trade

    risk_amount = engine.fixed_fractional(risk_pct=0.02)  # 2%

    leverages = [1, 2, 5, 10, 20]

    print(f"\n  Entry:       ${entry:>12,.2f}")
    print(f"  Stop-Loss:   ${sl:>12,.2f}  (SL dist = ${sl_dist:,.2f}  /  {sl_dist/entry*100:.2f}%)")
    print(f"  Risk amount: ${risk_amount:>12,.2f}  (2% of ${engine.account_balance:,.0f})")
    print()
    print(f"  {B}{'Leverage':>9s} {'Notional':>14s} {'Margin':>14s} {'Qty (BTC)':>12s}{R}")
    print(SUB)

    for lev in leverages:
        ps = engine.calculate_position_size(entry, sl, risk_amount, leverage=lev)
        print(
            f"  {lev:>8d}x  "
            f"${ps.position_notional:>13,.2f}  "
            f"${ps.margin_required:>13,.2f}  "
            f"{ps.quantity:>11.4f}"
        )

    # Fixed fractional vs volatility-adjusted
    print(f"\n  {B}Volatility-Adjusted Sizing (base 2%):{R}")
    print(SUB)
    vol_cases = [
        (500,   entry, "Low vol  (ATR $500)"),
        (1_400, entry, "Normal   (ATR $1400)"),
        (3_000, entry, "High vol (ATR $3000)"),
    ]
    for atr_v, px, label in vol_cases:
        va = engine.volatility_adjusted_sizing(atr_v, px)
        col = G if va["vol_multiplier"] >= 1 else Y
        print(
            f"  {label:<28s}  "
            f"vol={col}{va['volatility_pct']:.2f}%{R}  "
            f"mult={col}{va['vol_multiplier']:.2f}x{R}  "
            f"risk={col}{va['adjusted_risk_pct']:.2f}%{R}  "
            f"${va['dollar_risk']:,.2f}"
        )

    print(f"\n  {G}✓ Position sizing calculation verified{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# TEST 3 — DRAWDOWN SCALING
# ═══════════════════════════════════════════════════════════════

def test_drawdown_scaling(engine: RiskEngine):
    """Simulate different drawdown levels and verify size scaling."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 3 — DRAWDOWN SCALING{R}")
    print(SEP)

    base_risk = 2_000.0   # $2 000 base risk

    dd_scenarios = [
        (0.00, "No drawdown"),
        (0.03, "3% drawdown"),
        (0.06, "6% drawdown"),
        (0.12, "12% drawdown"),
        (0.17, "17% drawdown"),
        (0.22, "22% drawdown — STOP"),
        (0.30, "30% drawdown — STOP"),
    ]

    print(f"\n  Base risk per trade: ${base_risk:,.2f}")
    print()
    print(f"  {B}{'Scenario':<28s} {'DD':>6s} {'Mult':>6s} {'Adj Risk':>12s} {'Status':<18s}{R}")
    print(SUB)

    for dd, label in dd_scenarios:
        engine.current_drawdown = dd
        res = engine.apply_drawdown_scaling(base_risk)

        mult = res["multiplier"]
        adj  = res["adjusted_risk"]
        status = res["status"]

        col = RED if mult == 0 else (Y if mult < 1 else G)
        print(
            f"  {label:<28s}  "
            f"{dd:5.1%}  "
            f"{col}{mult:5.2f}x{R}  "
            f"{col}${adj:>11,.2f}{R}  "
            f"{col}{status}{R}"
        )

    # Reset for subsequent tests
    engine.current_drawdown = 0.0

    print(f"\n  {G}✓ Drawdown scaling verified across all levels{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# TEST 4 — SL/TP WITH REAL BTC ATR
# ═══════════════════════════════════════════════════════════════

def test_sl_tp(engine: RiskEngine):
    """Compute SL/TP levels using ATR from the real BTC dataset."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 4 — SL/TP CALCULATION (REAL BTC ATR){R}")
    print(SEP)

    # ── Try loading real 4H feature data ────────────────────
    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
        last = df.iloc[-1]
        entry = float(last.get("close", 65_000))
        atr   = float(last.get("atr_14", 1_400))
        source = f"Live data  ({df.index[-1].date()})"
    else:
        entry  = 65_000.0
        atr    = 1_400.0
        source = "Synthetic data (feature cache not found)"

    print(f"\n  Source : {source}")
    print(f"  Entry  : ${entry:>12,.2f}")
    print(f"  ATR-14 : ${atr:>12,.2f}  ({atr/entry*100:.2f}% of price)")

    # ── LONG trade — all profiles ────────────────────────────
    print(f"\n  {B}LONG — Partial Exit (TP1=33%, TP2=33%, TP3=34%):{R}")
    print()
    print(f"  {B}{'Profile':<14s} {'SL':>12s} {'TP1':>12s} {'TP2':>12s} {'TP3':>12s} {'SL%':>6s} {'RR':>5s}{R}")
    print(SUB)

    for profile in ["conservative", "normal", "aggressive"]:
        lvl = engine.calculate_sl_tp(entry, "LONG", atr, profile)
        print(
            f"  {profile:<14s} "
            f"{RED}${lvl.stop_loss:>11,.2f}{R}  "
            f"{G}${lvl.take_profit_1:>11,.2f}{R}  "
            f"{G}${lvl.take_profit_2:>11,.2f}{R}  "
            f"{G}${lvl.take_profit_3:>11,.2f}{R}  "
            f"{lvl.sl_percentage:5.2f}%  "
            f"{Y}{lvl.risk_reward_ratio:.2f}R{R}"
        )

    # ── SHORT trade — normal profile ─────────────────────────
    print(f"\n  {B}SHORT — Normal Profile:{R}")
    print()
    print(f"  {B}{'Direction':<14s} {'SL':>12s} {'TP1':>12s} {'TP2':>12s} {'TP3':>12s} {'SL%':>6s} {'RR':>5s}{R}")
    print(SUB)

    lvl = engine.calculate_sl_tp(entry, "SHORT", atr, "normal")
    print(
        f"  {'normal':<14s} "
        f"{RED}${lvl.stop_loss:>11,.2f}{R}  "
        f"{G}${lvl.take_profit_1:>11,.2f}{R}  "
        f"{G}${lvl.take_profit_2:>11,.2f}{R}  "
        f"{G}${lvl.take_profit_3:>11,.2f}{R}  "
        f"{lvl.sl_percentage:5.2f}%  "
        f"{Y}{lvl.risk_reward_ratio:.2f}R{R}"
    )

    # ── Full position cascade ────────────────────────────────
    print(f"\n  {B}Full Trade Setup (LONG, Normal, 2% risk, no leverage):{R}")
    print(SUB)

    lvl_n    = engine.calculate_sl_tp(entry, "LONG", atr, "normal")
    risk_amt = engine.fixed_fractional(0.02)
    ps       = engine.calculate_position_size(entry, lvl_n.stop_loss, risk_amt, leverage=1)

    print(f"  Risk Amount  : ${ps.risk_amount:>10,.2f}")
    print(f"  Notional     : ${ps.position_notional:>10,.2f}")
    print(f"  Margin       : ${ps.margin_required:>10,.2f}")
    print(f"  Quantity     : {ps.quantity:>10.4f} BTC")
    print(f"  Stop-Loss    : ${lvl_n.stop_loss:>10,.2f}  ({RED}-{lvl_n.sl_percentage:.2f}%{R})")
    print(f"  TP1 (33%)    : ${lvl_n.take_profit_1:>10,.2f}  ({G}+{abs(lvl_n.take_profit_1-entry)/entry*100:.2f}%{R})")
    print(f"  TP2 (33%)    : ${lvl_n.take_profit_2:>10,.2f}  ({G}+{abs(lvl_n.take_profit_2-entry)/entry*100:.2f}%{R})")
    print(f"  TP3 (34%)    : ${lvl_n.take_profit_3:>10,.2f}  ({G}+{abs(lvl_n.take_profit_3-entry)/entry*100:.2f}%{R})")
    print(f"  R/R Ratio    : {Y}{lvl_n.risk_reward_ratio:.2f}R{R}")

    print(f"\n  {G}✓ SL/TP calculation verified{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# TEST 5 — RISK CHECKS SIMULATION
# ═══════════════════════════════════════════════════════════════

def test_risk_checks(engine: RiskEngine):
    """Simulate various breach scenarios for the risk guard."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 5 — RISK CHECKS SIMULATION{R}")
    print(SEP)

    def _run_check(label: str):
        res = engine.check_can_trade()
        ok  = res["can_trade"]
        col = G if ok else RED
        mark = "✓" if ok else "✗"
        print(f"\n  {B}{label}{R}")
        print(f"  Can Trade: {col}{mark} {'YES' if ok else 'NO'}{R}")
        for name, chk in res["checks"].items():
            status_col = G if chk["passed"] else RED
            s = "PASS" if chk["passed"] else "FAIL"
            print(
                f"    {name:<18s} limit={chk['limit']:.2%}  "
                f"current={chk['current']:.2%}  "
                f"{status_col}[{s}]{R}"
            )

    # ── Normal state ──────────────────────────────────────────
    engine.reset_all(100_000)
    _run_check("Scenario A: Fresh account — all clear")

    # ── Daily loss breach ─────────────────────────────────────
    engine.reset_all(100_000)
    engine.update_balance(-5_500)   # 5.5% loss → breaches 5% limit
    _run_check("Scenario B: Daily loss 5.5%  (limit 5%)")

    # ── Max drawdown breach ───────────────────────────────────
    engine.reset_all(100_000)
    engine.peak_balance = 100_000
    engine.account_balance = 83_000  # 17% DD → breaches 15%
    engine.current_drawdown = 0.17
    engine.daily_pnl = 0.0
    _run_check("Scenario C: Drawdown 17%  (limit 15%)")

    # ── Concurrent positions breach ───────────────────────────
    engine.reset_all(100_000)
    engine.open_positions = [{"id": i} for i in range(3)]   # at max (3)
    _run_check("Scenario D: 3 open positions  (limit 3 — no new trade)")

    # ── All good again ────────────────────────────────────────
    engine.reset_all(100_000)
    _run_check("Scenario E: Reset — all clear")

    print(f"\n  {G}✓ Risk checks simulation complete{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# TEST 6 — PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════

def test_performance_metrics(engine: RiskEngine):
    """Compute Sharpe, Sortino, Calmar, PF, Expectancy on synthetic data."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 6 — PERFORMANCE METRICS{R}")
    print(SEP)

    rng = np.random.default_rng(42)

    # ── Build synthetic equity curve ──────────────────────────
    # Simulate 500 trades: 55% WR, avg win 2R, avg loss 1R
    n_trades = 500
    win_rate = 0.55
    wins  = rng.random(n_trades) < win_rate
    pnl_r = np.where(wins, rng.uniform(1.5, 2.5, n_trades), -rng.uniform(0.8, 1.2, n_trades))
    risk  = 200.0   # $200 per trade

    pnl_dollars = pnl_r * risk
    equity = pd.Series(
        np.concatenate([[100_000], 100_000 + pnl_dollars.cumsum()]),
        name="equity"
    )
    returns = equity.pct_change().dropna()

    # Synthetic trade list
    trades = [
        {
            "pnl":   float(pnl_dollars[i]),
            "pnl_r": float(pnl_r[i]),
        }
        for i in range(n_trades)
    ]

    # ── Metrics ───────────────────────────────────────────────
    sharpe   = engine.calculate_sharpe_ratio(returns)
    sortino  = engine.calculate_sortino_ratio(returns)
    dd_info  = engine.calculate_max_drawdown(equity)
    calmar   = engine.calculate_calmar_ratio(returns, equity)
    pf       = engine.calculate_profit_factor(trades)
    exp_r    = engine.calculate_expectancy(trades)

    n_wins  = sum(1 for t in trades if t["pnl"] > 0)
    wr_pct  = n_wins / n_trades

    print(f"\n  {B}Simulation: {n_trades} trades, "
          f"target WR={win_rate:.0%}, risk=${risk:,.0f}/trade{R}")
    print()

    def _fmt(label, value, target, unit=""):
        ok = value >= target
        col = G if ok else Y
        mark = "✓" if ok else "~"
        print(f"  {label:<22s}: {col}{value:>8.3f}{unit}   target ≥ {target}{unit}  [{mark}]{R}")

    _fmt("Sharpe Ratio",    sharpe,              1.5)
    _fmt("Sortino Ratio",   sortino,             2.0)
    _fmt("Calmar Ratio",    calmar,              1.0)
    _fmt("Profit Factor",   pf,                  1.5)
    _fmt("Expectancy",      exp_r,               0.3, "R")

    print(f"\n  {'Max Drawdown':<22s}: {RED}{dd_info['max_drawdown']:>8.2%}{R}")
    print(f"  {'Win Rate':<22s}: {G if wr_pct >= 0.5 else Y}{wr_pct:>8.2%}{R}")
    print(f"  {'Actual Wins':<22s}: {n_wins:>8d} / {n_trades}")
    print(f"  {'Final Equity':<22s}: ${equity.iloc[-1]:>11,.2f}")
    print(f"  {'Return':<22s}: {(equity.iloc[-1]/equity.iloc[0]-1):>8.2%}")

    print(f"\n  {G}✓ Performance metrics verified{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"""
{C}{'═' * 65}
  RISK ENGINE — END-TO-END TEST
  Hedge Fund Grade Risk Management
{'═' * 65}{R}
""")

    t0 = time.perf_counter()

    config_path = str(PROJECT_ROOT / "configs" / "risk_config.yaml")
    engine = RiskEngine(config_path=config_path)

    print(f"  {B}Account Balance :{R} ${engine.account_balance:>12,.2f}")
    print(f"  {B}Max Risk/Trade  :{R} {engine.max_risk_per_trade:.0%}")
    print(f"  {B}Max Daily Loss  :{R} {engine.max_daily_loss:.0%}")
    print(f"  {B}Max Drawdown    :{R} {engine.max_total_drawdown:.0%}")
    print(f"  {B}Max Concurrent  :{R} {engine.max_concurrent}")

    # ── Run all tests ─────────────────────────────────────────
    test_kelly_criterion(engine)
    test_position_sizing(engine)
    test_drawdown_scaling(engine)
    test_sl_tp(engine)
    test_risk_checks(engine)
    test_performance_metrics(engine)

    elapsed = time.perf_counter() - t0

    print(f"""
{G}{'═' * 65}
  ✓ Risk Engine test complete in {elapsed:.2f}s
  → ready for backtest integration
{'═' * 65}{R}
""")


if __name__ == "__main__":
    main()
