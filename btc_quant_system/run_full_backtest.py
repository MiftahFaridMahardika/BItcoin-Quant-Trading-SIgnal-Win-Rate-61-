#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  BTC QUANT SYSTEM — FULL BACKTEST RUNNER
  Period: 2023-2024 (Out-of-Sample)
═══════════════════════════════════════════════════════════════

Workflow:
  1. Load data (1-min CSV → 4H resample)
  2. Compute features (50+ indicators)
  3. Generate signals (6-layer scoring)
  4. Run backtest (State Machine execution)
  5. Save results (JSON, CSV)
  6. Print formatted summary
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
from engines.execution_engine import ExecutionEngine, BacktestEngine

# ── ANSI ─────────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"

SEP = f"{C}{'═' * 60}{R}"


# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format=f"  {D}%(asctime)s{R} │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")


def print_header():
    print(f"""
{C}╔{'═' * 58}╗
║{'FULL BACKTEST — BTC QUANT SYSTEM':^58s}║
║{'Out-of-Sample: 2023-01-01 → 2024-12-31':^58s}║
╚{'═' * 58}╝{R}
""")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════

def step1_load_data() -> pd.DataFrame:
    """Load 1-min data, resample to 4H, period 2015-2024."""
    print(f"\n{SEP}")
    print(f"{C}  STEP 1 — LOAD DATA{R}")
    print(SEP)

    t0 = time.perf_counter()

    config_path = str(PROJECT_ROOT / "configs" / "trading_config.yaml")
    pipeline = DataPipeline(config_path=config_path)

    # Check processed cache first
    processed_path = PROJECT_ROOT / "data" / "processed" / "btcusd_4h.parquet"
    if processed_path.exists():
        logger.info(f"Loading cached 4H data from {processed_path.name}")
        df = pd.read_parquet(processed_path)
    else:
        logger.info("Loading raw 1-min CSV...")
        df = pipeline.get_data(
            timeframe="4h",
            start_date="2015-01-01",
            end_date="2024-12-31",
            use_cache=True,
        )

    # Filter 2015-2024
    df = df[df.index >= "2015-01-01"]
    df = df[df.index <= "2024-12-31 23:59:59"]

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Loaded: {len(df):,} candles │ "
        f"{df.index[0].date()} → {df.index[-1].date()} │ "
        f"{elapsed:.2f}s"
    )

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2 — PREPARE DATA (FEATURES)
# ═══════════════════════════════════════════════════════════════

def step2_prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 50+ technical indicators."""
    print(f"\n{SEP}")
    print(f"{C}  STEP 2 — COMPUTE FEATURES (50+ indicators){R}")
    print(SEP)

    t0 = time.perf_counter()

    # Check feature cache
    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if feat_path.exists():
        logger.info(f"Loading cached features from {feat_path.name}")
        df_feat = pd.read_parquet(feat_path)

        # Filter same range as raw data
        df_feat = df_feat[df_feat.index >= "2015-01-01"]
        df_feat = df_feat[df_feat.index <= "2024-12-31 23:59:59"]

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Features loaded: {len(df_feat):,} rows × "
            f"{len(df_feat.columns)} cols │ {elapsed:.2f}s"
        )
        return df_feat

    # Compute from scratch
    logger.info("Computing features from scratch...")
    engine = FeatureEngine()
    df_feat = engine.compute_all_features(df)

    # Save cache
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(feat_path)
    logger.info(f"Features saved to {feat_path.name}")

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Features computed: {len(df_feat):,} rows × "
        f"{len(df_feat.columns)} cols │ {elapsed:.2f}s"
    )
    return df_feat


# ═══════════════════════════════════════════════════════════════
# STEP 3 — GENERATE SIGNALS
# ═══════════════════════════════════════════════════════════════

def step3_generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate signals using 6-layer weighted scoring."""
    print(f"\n{SEP}")
    print(f"{C}  STEP 3 — GENERATE SIGNALS{R}")
    print(SEP)

    t0 = time.perf_counter()

    signal_engine = SignalEngine()
    df = signal_engine.generate_signals_batch(df, start_idx=200)

    elapsed = time.perf_counter() - t0

    # Distribution
    dist = df["signal_type"].value_counts()
    total = len(df)
    actionable = total - dist.get("SKIP", 0)

    logger.info(f"Signals generated in {elapsed:.2f}s")
    logger.info(f"Actionable: {actionable:,} / {total:,} ({actionable/total*100:.1f}%)")

    for sig_type in ["STRONG_LONG", "LONG", "SHORT", "STRONG_SHORT"]:
        cnt = dist.get(sig_type, 0)
        if cnt > 0:
            logger.info(f"  {sig_type:<14s}: {cnt:>6,} ({cnt/total*100:.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 4 — RUN BACKTEST
# ═══════════════════════════════════════════════════════════════

def step4_run_backtest(df: pd.DataFrame) -> dict:
    """Run backtest on 2023-2024 out-of-sample period."""
    print(f"\n{SEP}")
    print(f"{C}  STEP 4 — RUN BACKTEST (2023-2024 Out-of-Sample){R}")
    print(SEP)

    t0 = time.perf_counter()

    # Config
    backtest_config = {
        'initial_capital': 100_000,
        'slippage_pct': 0.0005,      # 0.05%
        'maker_fee_pct': 0.0002,     # 0.02%
        'taker_fee_pct': 0.0004,     # 0.04%
        'leverage': 1,
        'warmup_periods': 0,          # Data already has features from 2015
    }

    # Engines
    risk_config_path = str(PROJECT_ROOT / "configs" / "risk_config.yaml")
    risk_engine = RiskEngine(config_path=risk_config_path)
    signal_engine = SignalEngine()

    bt = BacktestEngine(backtest_config)

    logger.info(f"Capital: ${backtest_config['initial_capital']:,.0f}")
    logger.info(f"Risk/trade: 2% (Half Kelly, capped)")
    logger.info(f"Max DD: 15% │ Slippage: 0.05% │ Fees: 0.04% taker")

    # Filter test period
    df_test = df[(df.index >= "2023-01-01") & (df.index <= "2024-12-31")]
    logger.info(f"Test period: {df_test.index[0].date()} → {df_test.index[-1].date()}")
    logger.info(f"Candles: {len(df_test):,}")
    logger.info("")

    results = bt.run_backtest(
        df=df_test,
        signal_engine=signal_engine,
        risk_engine=risk_engine,
        show_progress=True,
    )

    elapsed = time.perf_counter() - t0
    logger.info(f"Backtest completed in {elapsed:.2f}s")

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 5 — SAVE RESULTS
# ═══════════════════════════════════════════════════════════════

def step5_save_results(results: dict):
    """Save results to JSON, trades.csv, equity_curve.csv."""
    print(f"\n{SEP}")
    print(f"{C}  STEP 5 — SAVE RESULTS{R}")
    print(SEP)

    output_dir = PROJECT_ROOT / "backtests" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON (full results)
    json_path = output_dir / "backtest_2023_2024.json"
    results_json = results.copy()

    # Convert equity curve for JSON serialization
    eq = results_json.pop('equity_curve', {})
    results_json['equity_curve_summary'] = {
        'points': len(eq),
        'start_equity': list(eq.values())[0] if eq else 0,
        'end_equity': list(eq.values())[-1] if eq else 0,
    }

    # Trades already dicts
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info(f"Saved: {json_path.name}")

    # 2. Trades CSV
    trades = results.get('trades', [])
    if trades:
        trades_path = output_dir / "trades.csv"
        df_trades = pd.DataFrame(trades)
        df_trades.to_csv(trades_path, index=False)
        logger.info(f"Saved: {trades_path.name} ({len(trades)} trades)")

    # 3. Equity Curve CSV
    eq_raw = results.get('equity_curve', {})
    if eq_raw:
        eq_path = output_dir / "equity_curve.csv"
        df_eq = pd.DataFrame({
            'timestamp': [str(k) for k in eq_raw.keys()],
            'equity': list(eq_raw.values()),
        })
        df_eq.to_csv(eq_path, index=False)
        logger.info(f"Saved: {eq_path.name} ({len(df_eq)} points)")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — PRINT FORMATTED SUMMARY
# ═══════════════════════════════════════════════════════════════

def step6_print_summary(results: dict):
    """Print hedge-fund style formatted summary."""
    print(f"\n{SEP}")
    print(f"{C}  STEP 6 — RESULTS SUMMARY{R}")
    print(SEP)

    if 'error' in results:
        print(f"\n  {RED}⚠ {results['error']}{R}")
        return

    # ── Header ───────────────────────────────────────────
    print(f"""
{C}╔{'═' * 58}╗
║{'BACKTEST RESULTS: 2023-2024':^58s}║
║{'Out-of-Sample │ No Look-Ahead Bias':^58s}║
╠{'═' * 58}╣{R}""")

    # ── Capital ──────────────────────────────────────────
    init = results['initial_capital']
    final = results['final_equity']
    pnl = results['total_pnl']
    pnl_col = G if pnl >= 0 else RED

    print(f"""║{R}
║  {B}CAPITAL{R}
║  {'─' * 40}
║  Initial Capital:  ${init:>12,.2f}
║  Final Equity:     {pnl_col}${final:>12,.2f}{R}
║  Total P&L:        {pnl_col}${pnl:>12,.2f}{R}""")

    # ── Returns ──────────────────────────────────────────
    ret = results['total_return_pct']
    cagr = results.get('cagr_pct', 0)
    ret_col = G if ret >= 0 else RED

    print(f"""║
║  {B}RETURNS{R}
║  {'─' * 40}
║  Total Return:     {ret_col}{ret:>12.2f}%{R}
║  CAGR:             {ret_col}{cagr:>12.2f}%{R}""")

    # ── Trade Statistics ─────────────────────────────────
    total_t = results['total_trades']
    wins = results['winning_trades']
    losses = results['losing_trades']
    wr = results['win_rate_pct']
    avg_w = results['avg_win_usd']
    avg_l = results['avg_loss_usd']
    avg_wr = results['avg_win_r']
    avg_lr = results['avg_loss_r']
    max_cl = results.get('max_consecutive_losses', 0)

    print(f"""║
║  {B}TRADE STATISTICS{R}
║  {'─' * 40}
║  Total Trades:     {total_t:>12d}
║  Winning:          {G}{wins:>12d}{R}
║  Losing:           {RED}{losses:>12d}{R}
║  Win Rate:         {G if wr >= 50 else Y}{wr:>12.1f}%{R}
║  Avg Win:          {G}${avg_w:>11,.2f}  ({avg_wr:.2f}R){R}
║  Avg Loss:         {RED}${avg_l:>11,.2f}  ({avg_lr:.2f}R){R}
║  Max Consec Loss:  {max_cl:>12d}""")

    # ── Risk Metrics ─────────────────────────────────────
    sharpe = results['sharpe_ratio']
    sortino = results['sortino_ratio']
    calmar = results['calmar_ratio']
    max_dd = results['max_drawdown_pct']
    pf = results['profit_factor']
    exp = results['expectancy_r']

    def _metric(label, val, target, higher_better=True, fmt=".2f"):
        if higher_better:
            ok = val >= target
        else:
            ok = val <= target
        col = G if ok else Y
        mark = "✓" if ok else "✗"
        cmp = ">" if higher_better else "<"
        return (
            f"║  {label:<20s} {col}{val:>8{fmt}}{R}    "
            f"[Target: {cmp}{target:{fmt}}] {col}[{mark}]{R}"
        )

    print(f"""║
║  {B}RISK METRICS{R}
║  {'─' * 40}""")
    print(_metric("Sharpe Ratio:",   sharpe,  1.5))
    print(_metric("Sortino Ratio:",  sortino, 2.0))
    print(_metric("Calmar Ratio:",   calmar,  1.0))
    print(_metric("Max Drawdown:",   max_dd,  15.0, higher_better=False, fmt=".1f"))
    print(_metric("Profit Factor:",  pf,      1.5))
    print(_metric("Expectancy:",     exp,     0.3))

    # ── Score Card ───────────────────────────────────────
    targets = [
        sharpe >= 1.5,
        sortino >= 2.0,
        calmar >= 1.0,
        max_dd <= 15.0,
        pf >= 1.5,
        exp >= 0.3,
    ]
    passed = sum(targets)
    total_targets = len(targets)

    grade_col = G if passed >= 5 else Y if passed >= 3 else RED

    print(f"""║
║  {B}SCORECARD{R}
║  {'─' * 40}
║  Targets Met:      {grade_col}{passed}/{total_targets}{R}
║""")

    # ── Footer ───────────────────────────────────────────
    print(f"""{C}╚{'═' * 58}╝{R}""")

    # ── Sample Trades ────────────────────────────────────
    trades = results.get('trades', [])
    if trades:
        print(f"\n  {B}Sample Trades (first 10):{R}")
        print(f"  {'─' * 95}")
        print(
            f"  {'ID':<16s} {'Dir':<6s} {'Entry':>12s} {'Exit':>12s} "
            f"{'PnL':>11s} {'R':>7s} {'Reason':<12s}"
        )
        print(f"  {'─' * 95}")
        for t in trades[:10]:
            col = G if t['pnl'] > 0 else RED
            d = t.get('direction') or '?'
            reason = t.get('exit_reason') or '?'
            print(
                f"  {t['id']:<16s} {d:<6s} "
                f"${t['entry_price']:>11,.2f} ${t['exit_price']:>11,.2f} "
                f"{col}${t['pnl']:>10,.2f}{R} "
                f"{col}{t['pnl_r']:>6.2f}R{R} "
                f"{reason:<12s}"
            )

    # ── Trade Distribution by Exit Reason ────────────────
    if trades:
        print(f"\n  {B}Exit Reason Distribution:{R}")
        print(f"  {'─' * 40}")
        reason_counts = {}
        reason_pnl = {}
        for t in trades:
            r = t.get('exit_reason') or 'UNKNOWN'
            reason_counts[r] = reason_counts.get(r, 0) + 1
            reason_pnl[r] = reason_pnl.get(r, 0) + t['pnl']

        for reason, cnt in sorted(reason_counts.items(),
                                   key=lambda x: -x[1]):
            avg_pnl = reason_pnl[reason] / cnt
            col = G if reason_pnl[reason] > 0 else RED
            print(
                f"  {reason:<14s} │ {cnt:>4d} trades │ "
                f"{col}Total: ${reason_pnl[reason]:>10,.2f}  "
                f"Avg: ${avg_pnl:>8,.2f}{R}"
            )


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print_header()
    t0 = time.perf_counter()

    # Step 1: Load data
    df = step1_load_data()

    # Step 2: Compute features
    df_feat = step2_prepare_features(df)

    # Step 3: Generate signals
    df_signals = step3_generate_signals(df_feat)

    # Step 4: Run backtest
    results = step4_run_backtest(df_signals)

    # Step 5: Save results
    step5_save_results(results)

    # Step 6: Print summary
    step6_print_summary(results)

    # Final
    elapsed = time.perf_counter() - t0
    print(f"""
{G}{'═' * 60}
  ✓ Full Backtest Pipeline Complete
  Total time: {elapsed:.2f}s
  Results saved to backtests/results/
{'═' * 60}{R}
""")


if __name__ == "__main__":
    main()
