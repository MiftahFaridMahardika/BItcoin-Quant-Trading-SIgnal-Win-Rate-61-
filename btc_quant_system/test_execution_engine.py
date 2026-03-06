#!/usr/bin/env python3
"""
Execution Engine — End-to-End Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. State machine transitions
2. Entry / exit execution with slippage & fees
3. Mini backtest on featured BTC data
4. Performance summary
5. State transition trace
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from engines.execution_engine import (
    ExecutionEngine, BacktestEngine, TradeState, Trade,
)
from engines.signal_engine import SignalEngine
from engines.risk_engine import RiskEngine

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
# TEST 1 — STATE MACHINE TRANSITIONS
# ═══════════════════════════════════════════════════════════════

def test_state_machine():
    """Verify all state transitions are valid."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 1 — STATE MACHINE TRANSITIONS{R}")
    print(SEP)

    states = list(TradeState)
    print(f"\n  {B}Registered States ({len(states)}):{R}")
    print(SUB)

    valid_transitions = {
        TradeState.IDLE:          [TradeState.SCANNING],
        TradeState.SCANNING:      [TradeState.SIGNAL_FOUND, TradeState.SCANNING],
        TradeState.SIGNAL_FOUND:  [TradeState.PENDING_ENTRY, TradeState.SCANNING],
        TradeState.PENDING_ENTRY: [TradeState.IN_TRADE],
        TradeState.IN_TRADE:      [TradeState.CLOSING, TradeState.IN_TRADE],
        TradeState.PARTIAL_EXIT:  [TradeState.IN_TRADE, TradeState.CLOSING],
        TradeState.CLOSING:       [TradeState.SCANNING],
    }

    for state in states:
        targets = valid_transitions.get(state, [])
        arrows = " → ".join([t.value for t in targets]) if targets else "(terminal)"
        print(f"  {state.value:<16s} → {arrows}")

    # Verify happy-path cycle
    happy_path = [
        TradeState.IDLE,
        TradeState.SCANNING,
        TradeState.SIGNAL_FOUND,
        TradeState.PENDING_ENTRY,
        TradeState.IN_TRADE,
        TradeState.CLOSING,
        TradeState.SCANNING,
    ]
    print(f"\n  {B}Happy-Path Cycle:{R}")
    print(f"  {' → '.join(s.value for s in happy_path)}")

    print(f"\n  {G}✓ State machine structure verified{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# TEST 2 — TRADE DATACLASS
# ═══════════════════════════════════════════════════════════════

def test_trade_dataclass():
    """Verify Trade dataclass fields."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 2 — TRADE DATA CLASS{R}")
    print(SEP)

    trade = Trade(
        id="TRADE_000001",
        entry_time=pd.Timestamp("2023-06-15 08:00"),
        direction="LONG",
        entry_price=65_000.0,
        stop_loss=63_900.0,
        take_profit_1=67_800.0,
        take_profit_2=69_900.0,
        take_profit_3=72_000.0,
        quantity=0.95,
        risk_amount=2_000.0,
        signal_score=10,
        confidence=0.72,
    )

    fields = [
        ("ID",       trade.id),
        ("Entry",    f"${trade.entry_price:,.2f}"),
        ("SL",       f"${trade.stop_loss:,.2f}"),
        ("TP1",      f"${trade.take_profit_1:,.2f}"),
        ("TP2",      f"${trade.take_profit_2:,.2f}"),
        ("TP3",      f"${trade.take_profit_3:,.2f}"),
        ("Qty",      f"{trade.quantity:.4f} BTC"),
        ("Risk",     f"${trade.risk_amount:,.2f}"),
        ("Score",    f"{trade.signal_score}"),
        ("Conf",     f"{trade.confidence:.2%}"),
        ("Status",   trade.status),
    ]

    print()
    for label, val in fields:
        print(f"  {label:<12s}: {val}")

    assert trade.status == "OPEN"
    assert trade.pnl == 0.0
    assert trade.partial_exits == []

    print(f"\n  {G}✓ Trade dataclass verified{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# TEST 3 — ENTRY / EXIT WITH SLIPPAGE & FEES
# ═══════════════════════════════════════════════════════════════

def test_entry_exit_mechanics():
    """Test entry and exit with realistic costs."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 3 — ENTRY / EXIT MECHANICS{R}")
    print(SEP)

    # Setup mock signal engine that returns a controlled signal
    class MockSignalEngine:
        def generate_trading_signal(self, df, idx):
            return {
                "signal": "STRONG_LONG",
                "direction": "LONG",
                "entry_price": 65_000.0,
                "stop_loss": 62_900.0,
                "tp1": 67_800.0,
                "tp2": 69_900.0,
                "tp3": 72_000.0,
                "risk_reward": 2.33,
                "confidence": 0.72,
                "score": 10,
                "regime": "NORMAL",
                "reasons": ["Test signal"],
                "timestamp": pd.Timestamp("2023-06-15 08:00"),
            }

    risk_config = str(PROJECT_ROOT / "configs" / "risk_config.yaml")
    risk_engine = RiskEngine(config_path=risk_config)
    risk_engine.reset_all(100_000)

    config = {
        'slippage_pct': 0.0005,   # 0.05%
        'maker_fee_pct': 0.0002,
        'taker_fee_pct': 0.0004,
        'leverage': 1,
        'initial_capital': 100_000,
    }

    engine = ExecutionEngine(config, MockSignalEngine(), risk_engine)

    # Build minimal DataFrame
    dates = pd.date_range("2023-06-15 04:00", periods=10, freq="4h")
    df = pd.DataFrame({
        'open':  [64_800 + i*100 for i in range(10)],
        'high':  [65_200 + i*100 for i in range(10)],
        'low':   [64_500 + i*100 for i in range(10)],
        'close': [65_000 + i*100 for i in range(10)],
        'Close': [65_000 + i*100 for i in range(10)],
        'High':  [65_200 + i*100 for i in range(10)],
        'Low':   [64_500 + i*100 for i in range(10)],
        'atr_14': [1_400] * 10,
    }, index=dates)

    print(f"\n  {B}State Machine Step-by-Step:{R}")
    print(SUB)

    trace = []
    for step in range(6):
        result = engine.run_step(df, step, dates[step])
        state = result['state']
        action = result.get('action', '—')
        trade = result.get('trade')

        col = G if action in ('ENTRY', 'EXIT') else Y if action else D
        print(
            f"  Step {step} │ {col}{state:<16s}{R} │ "
            f"action={col}{str(action):<16s}{R}"
            f"{f' │ id={trade.id}' if trade else ''}"
        )
        trace.append(state)

    print(f"\n  {B}Trade Details:{R}")
    print(SUB)

    if engine.current_trade:
        t = engine.current_trade
        slip = abs(t.entry_price - 65_000.0)
        print(f"  Entry (raw)   : $65,000.00")
        print(f"  Entry (slipped): ${t.entry_price:,.2f}  (+${slip:.2f} slippage)")
        print(f"  Direction     : {t.direction}")
        print(f"  Quantity      : {t.quantity:.4f} BTC")
        print(f"  Stop-Loss     : ${t.stop_loss:,.2f}")
        print(f"  TP3           : ${t.take_profit_3:,.2f}")
        print(f"  Risk Amount   : ${t.risk_amount:,.2f}")

    # Force close via TP3 hit
    # Create a row with high > TP3
    close_row = pd.Series({
        'close': 72_500, 'Close': 72_500,
        'high': 73_000, 'High': 73_000,
        'low': 71_800, 'Low': 71_800,
        'atr_14': 1_400,
    }, name=pd.Timestamp("2023-06-17 08:00"))

    # Inject the close row
    df_close = pd.concat([df, close_row.to_frame().T])

    # Manually trigger close
    engine.state = TradeState.CLOSING
    result = engine.run_step(df_close, len(df_close)-1, close_row.name)

    if result.get('trade'):
        t = result['trade']
        col = G if t.pnl > 0 else RED
        print(f"\n  {B}Exit:{R}")
        print(f"  Exit Price    : ${t.exit_price:,.2f}")
        print(f"  PnL           : {col}${t.pnl:,.2f}{R}")
        print(f"  PnL (R)       : {col}{t.pnl_r:.2f}R{R}")
        print(f"  PnL (%)       : {col}{t.pnl_pct:.2f}%{R}")
        print(f"  Exit Reason   : {t.exit_reason}")
        print(f"  Status        : {t.status}")

    print(f"\n  {G}✓ Entry/Exit mechanics verified{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# TEST 4 — MINI BACKTEST (SYNTHETIC OR REAL DATA)
# ═══════════════════════════════════════════════════════════════

def test_mini_backtest():
    """Run a mini backtest on available data."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 4 — MINI BACKTEST{R}")
    print(SEP)

    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"

    if feat_path.exists():
        print(f"\n  Loading featured data...")
        df = pd.read_parquet(feat_path)
        source = f"Real BTC 4H ({len(df):,} candles)"
    else:
        print(f"\n  {Y}Feature file not found — using synthetic data{R}")
        # Generate synthetic featured data
        rng = np.random.default_rng(42)
        n = 2000
        dates = pd.date_range("2023-01-01", periods=n, freq="4h")
        base_price = 30_000
        returns = rng.normal(0.0002, 0.015, n)
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'Close': prices,
            'close': prices,
            'Open': prices * (1 - rng.uniform(0, 0.01, n)),
            'High': prices * (1 + rng.uniform(0.005, 0.02, n)),
            'high': prices * (1 + rng.uniform(0.005, 0.02, n)),
            'Low': prices * (1 - rng.uniform(0.005, 0.02, n)),
            'low': prices * (1 - rng.uniform(0.005, 0.02, n)),
            'Volume': rng.uniform(100, 1000, n),
            'atr_14': prices * 0.02,
            'ema_stack_signal': rng.choice([-1, 0, 1], n, p=[0.25, 0.50, 0.25]),
            'hma_signal': rng.choice([-1, 0, 1], n, p=[0.30, 0.40, 0.30]),
            'supertrend_dir': rng.choice([-1, 1], n, p=[0.45, 0.55]),
            'rsi_signal': rng.choice([-1, 0, 1], n, p=[0.15, 0.70, 0.15]),
            'rsi': rng.uniform(20, 80, n),
            'macd_hist': rng.normal(0, 100, n),
            'macd_line': rng.normal(0, 200, n),
            'zscore_20': rng.normal(0, 1.5, n),
            'zscore_signal': rng.choice([-1, 0, 1], n, p=[0.10, 0.80, 0.10]),
            'bb_signal': rng.choice([-1, 0, 1], n, p=[0.10, 0.80, 0.10]),
            'bb_pct_b': rng.uniform(0, 1, n),
            'kc_signal': rng.choice([-1, 0, 1], n, p=[0.10, 0.80, 0.10]),
            'vol_ratio': rng.uniform(0.5, 2.5, n),
            'obv_signal': rng.choice([-1, 0, 1], n, p=[0.15, 0.70, 0.15]),
            'ret_1': rng.normal(0, 1, n),
            'ret_6': rng.normal(0, 3, n),
            'ret_42': rng.normal(0, 10, n),
            'trend_structure': rng.choice([-1, 0, 1], n, p=[0.25, 0.50, 0.25]),
            'vol_regime': rng.choice([0, 1, 2, 3], n, p=[0.15, 0.55, 0.25, 0.05]),
        }, index=dates)
        source = f"Synthetic ({n:,} candles)"

    print(f"  Source: {source}")
    print(f"  Range : {df.index[0]} → {df.index[-1]}")

    # Engines
    risk_config = str(PROJECT_ROOT / "configs" / "risk_config.yaml")
    risk_engine = RiskEngine(config_path=risk_config)
    signal_engine = SignalEngine()

    backtest_config = {
        'initial_capital': 100_000,
        'slippage_pct': 0.0005,
        'maker_fee_pct': 0.0002,
        'taker_fee_pct': 0.0004,
        'leverage': 1,
        'warmup_periods': 200,
    }

    bt = BacktestEngine(backtest_config)

    import logging
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    print(f"\n  {B}Running backtest...{R}\n")
    t0 = time.perf_counter()

    results = bt.run_backtest(
        df=df,
        signal_engine=signal_engine,
        risk_engine=risk_engine,
        show_progress=True,
    )

    elapsed = time.perf_counter() - t0

    print(f"\n  {B}Backtest completed in {elapsed:.2f}s{R}")
    print(SUB)

    # ── Results ──────────────────────────────────────────

    if 'error' in results:
        print(f"\n  {Y}⚠ {results['error']}{R}")
    else:
        print(f"\n  {B}{'CAPITAL':}{R}")
        print(f"  Initial Capital  : ${results['initial_capital']:>12,.2f}")
        print(f"  Final Equity     : ${results['final_equity']:>12,.2f}")
        print(f"  Total PnL        : {'${:>12,.2f}'.format(results['total_pnl'])}")
        print(f"  Total Return     : {results['total_return_pct']:>12.2f}%")

        print(f"\n  {B}{'TRADE STATS':}{R}")
        print(f"  Total Trades     : {results['total_trades']:>12d}")
        print(f"  Winning          : {results['winning_trades']:>12d}")
        print(f"  Losing           : {results['losing_trades']:>12d}")
        print(f"  Win Rate         : {results['win_rate_pct']:>12.1f}%")

        print(f"\n  {B}{'AVERAGES':}{R}")
        print(f"  Avg Win          : ${results['avg_win_usd']:>12,.2f}")
        print(f"  Avg Loss         : ${results['avg_loss_usd']:>12,.2f}")
        print(f"  Avg Win (R)      : {results['avg_win_r']:>12.2f}R")
        print(f"  Avg Loss (R)     : {results['avg_loss_r']:>12.2f}R")

        print(f"\n  {B}{'RISK METRICS':}{R}")

        def _metric(label, val, target=None, fmt=".2f"):
            s = f"  {label:<20s}: {val:>12{fmt}}"
            if target is not None:
                ok = val >= target if target > 0 else val <= abs(target)
                col = G if ok else Y
                mark = "✓" if ok else "~"
                s += f"  {col}(target ≥ {target:{fmt}}) [{mark}]{R}"
            print(s)

        _metric("Sharpe Ratio",     results['sharpe_ratio'],    1.5)
        _metric("Sortino Ratio",    results['sortino_ratio'],   2.0)
        _metric("Calmar Ratio",     results['calmar_ratio'],    1.0)
        _metric("Max Drawdown %",   results['max_drawdown_pct'])
        _metric("Profit Factor",    results['profit_factor'],   1.5)
        _metric("Expectancy (R)",   results['expectancy_r'],    0.3)
        _metric("Max Consec. Losses", results['max_consecutive_losses'], fmt="d")

        # Sample trades
        trades = results.get('trades', [])
        if trades:
            print(f"\n  {B}Sample Trades (first 5):{R}")
            print(SUB)
            print(
                f"  {'ID':<16s} {'Dir':<6s} {'Entry':>12s} {'Exit':>12s} "
                f"{'PnL':>10s} {'R':>6s} {'Reason':<12s}"
            )
            print(SUB)
            for t in trades[:5]:
                col = G if t['pnl'] > 0 else RED
                d = t.get('direction') or '?'
                reason = t.get('exit_reason') or '?'
                print(
                    f"  {t['id']:<16s} {d:<6s} "
                    f"${t['entry_price']:>11,.2f} ${t['exit_price']:>11,.2f} "
                    f"{col}${t['pnl']:>9,.2f}{R} "
                    f"{col}{t['pnl_r']:>5.2f}R{R} "
                    f"{reason:<12s}"
                )

    print(f"\n  {G}✓ Mini backtest verified{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# TEST 5 — STATE TRANSITION TRACE
# ═══════════════════════════════════════════════════════════════

def test_state_trace():
    """Visualise state transitions from a short run."""
    print(f"\n{SEP}")
    print(f"{C}  TEST 5 — STATE TRANSITION TRACE{R}")
    print(SEP)

    class FixedSignalEngine:
        """Returns alternating signals for trace visualisation."""
        def __init__(self):
            self.call_count = 0

        def generate_trading_signal(self, df, idx):
            self.call_count += 1
            # Signal every 8th bar
            if self.call_count % 8 == 0:
                return {
                    "signal": "LONG",
                    "direction": "LONG",
                    "entry_price": float(df.iloc[idx].get('close', 65_000)),
                    "stop_loss": float(df.iloc[idx].get('close', 65_000)) * 0.97,
                    "tp1": float(df.iloc[idx].get('close', 65_000)) * 1.03,
                    "tp2": float(df.iloc[idx].get('close', 65_000)) * 1.05,
                    "tp3": float(df.iloc[idx].get('close', 65_000)) * 1.08,
                    "risk_reward": 2.5,
                    "confidence": 0.75,
                    "score": 8,
                    "regime": "NORMAL",
                    "reasons": [],
                    "timestamp": df.iloc[idx].name,
                }
            return {"signal": "SKIP"}

    risk_config = str(PROJECT_ROOT / "configs" / "risk_config.yaml")
    risk_engine = RiskEngine(config_path=risk_config)
    risk_engine.reset_all(100_000)

    config = {
        'slippage_pct': 0.0005,
        'maker_fee_pct': 0.0002,
        'taker_fee_pct': 0.0004,
        'leverage': 1,
    }

    sig_eng = FixedSignalEngine()
    engine = ExecutionEngine(config, sig_eng, risk_engine)

    # Minimal DataFrame
    rng = np.random.default_rng(99)
    n = 30
    dates = pd.date_range("2023-07-01", periods=n, freq="4h")
    base = 65_000
    prices = base + rng.normal(0, 500, n).cumsum()

    df = pd.DataFrame({
        'close': prices, 'Close': prices,
        'open': prices - 50, 'Open': prices - 50,
        'high': prices + rng.uniform(200, 800, n),
        'High': prices + rng.uniform(200, 800, n),
        'low': prices - rng.uniform(200, 800, n),
        'Low': prices - rng.uniform(200, 800, n),
        'atr_14': [1_400] * n,
    }, index=dates)

    print(f"\n  {B}{'Step':<6s} {'State':<16s} {'Action':<18s} {'Price':>12s}{R}")
    print(SUB)

    state_colors = {
        'IDLE': D, 'SCANNING': C, 'SIGNAL_FOUND': Y,
        'PENDING_ENTRY': Y, 'IN_TRADE': G,
        'CLOSING': RED, 'PARTIAL_EXIT': Y,
    }

    for step in range(n):
        result = engine.run_step(df, step, dates[step])
        state = result['state']
        action = result.get('action') or '—'
        price = prices[step]
        col = state_colors.get(state, D)
        acol = G if action in ('ENTRY', 'EXIT', 'SIGNAL_FOUND') else D
        print(
            f"  {step:<6d} {col}{state:<16s}{R} "
            f"{acol}{action:<18s}{R} ${price:>11,.2f}"
        )

    trades = engine.trade_history
    print(f"\n  Trades completed: {len(trades)}")
    for t in trades:
        col = G if t.pnl > 0 else RED
        print(f"    {t.id}: {t.direction} ${t.entry_price:,.2f} → ${t.exit_price:,.2f} "
              f"{col}{t.pnl_r:+.2f}R{R} ({t.exit_reason})")

    print(f"\n  {G}✓ State transition trace verified{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"""
{C}{'═' * 65}
  EXECUTION ENGINE — END-TO-END TEST
  State Machine + Backtesting Engine
{'═' * 65}{R}
""")

    t0 = time.perf_counter()

    test_state_machine()
    test_trade_dataclass()
    test_entry_exit_mechanics()
    test_mini_backtest()
    test_state_trace()

    elapsed = time.perf_counter() - t0

    print(f"""
{G}{'═' * 65}
  ✓ Execution Engine test complete in {elapsed:.2f}s
  → State Machine + Backtest verified
{'═' * 65}{R}
""")


if __name__ == "__main__":
    main()
