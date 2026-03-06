"""
BTC Quant Trading System — Custom Yearly Backtest (2017-2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Trade Plan:
- POSITION: Fixed $1,000 margin with 15x leverage ($15,000 notional)
- SL: 1.333% from entry (~21.2% account risk with fees)
- TP: Trailing TP. Initial target 0.71%. Cut on reversal.
- Period: 2017-2026 (Yearly reports)
- Timeframe: 4H
"""

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple, Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from engines.data_pipeline import DataPipeline
from engines.feature_engine import FeatureEngine
from engines.signal_engine import SignalEngine
from engines.risk_engine import RiskEngine
from engines.execution_engine import ExecutionEngine, Trade, TradeState, BacktestEngine

# ══════════════════════════════════════════════════════════════
# CUSTOM EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════

class CustomExecutionEngine(ExecutionEngine):
    """
    Implements the user's specific trade plan:
    - SL at 1.333%
    - Trailing TP starts at 0.71%
    - Fixed $15,000 notional ($1,000 * 15x)
    """

    def __init__(self, config, signal_engine, risk_engine):
        super().__init__(config, signal_engine, risk_engine)
        self.sl_pct = 0.01333
        self.tp_trigger_pct = 0.0071
        self.notional = 15000.0
        self.peak_favorable_price = None

    def _execute_entry(self, row: pd.Series) -> Trade:
        signal = self.pending_signal
        direction = signal['direction']
        
        # Apply slippage
        raw_price = signal['entry_price']
        if direction == 'LONG':
            entry_price = raw_price * (1 + self.slippage_pct)
            stop_loss = entry_price * (1 - self.sl_pct)
        else:
            entry_price = raw_price * (1 - self.slippage_pct)
            stop_loss = entry_price * (1 + self.sl_pct)

        # Fixed quantity based on $15,000 notional
        quantity = self.notional / entry_price

        self.trade_count += 1
        self.peak_favorable_price = entry_price

        return Trade(
            id=f"TRADE_{self.trade_count:06d}",
            entry_time=row.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_3=0, # Not used here, we use trailing
            quantity=quantity,
            risk_amount=1000.0, # Margin used
            signal_score=signal.get('score', 0),
            confidence=signal.get('confidence', 0),
            status="OPEN",
        )

    def _check_exit(self, row: pd.Series) -> Dict:
        trade = self.current_trade
        high = float(row.get('high', row.get('High', 0)))
        low = float(row.get('low', row.get('Low', 0)))
        close = float(row.get('close', row.get('Close', 0)))

        # 1. Check Stop Loss
        if trade.direction == 'LONG':
            if low <= trade.stop_loss:
                return {'should_exit': True, 'reason': 'STOP_LOSS'}
        else:
            if high >= trade.stop_loss:
                return {'should_exit': True, 'reason': 'STOP_LOSS'}

        # 2. Update Peak Price for Trailing
        if trade.direction == 'LONG':
            if high > self.peak_favorable_price:
                self.peak_favorable_price = high
        else:
            if low < self.peak_favorable_price:
                self.peak_favorable_price = low

        # 3. Check Trailing / Reversal
        # Reversal logic: if we hit 0.71% target, then exit if price reverses by 0.3% from peak
        move_pct = 0
        if trade.direction == 'LONG':
            move_pct = (self.peak_favorable_price - trade.entry_price) / trade.entry_price
        else:
            move_pct = (trade.entry_price - self.peak_favorable_price) / trade.entry_price

        if move_pct >= self.tp_trigger_pct:
            # Reversal threshold (e.g., 0.3% move back from peak)
            if trade.direction == 'LONG':
                rev_pct = (self.peak_favorable_price - close) / self.peak_favorable_price
            else:
                rev_pct = (close - self.peak_favorable_price) / self.peak_favorable_price
            
            if rev_pct >= 0.003: # 0.3% reversal
                return {'should_exit': True, 'reason': 'TRAILING_TP'}

        return {'should_exit': False, 'reason': None}

    def _update_trailing_stop(self, row: pd.Series):
        # We handle trailing in _check_exit for this specific strategy
        pass

# ══════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def run_custom_backtest():
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("trading")
    logger.setLevel(logging.INFO)

    # 1. DATA PIPELINE
    print("Loading data...")
    pipeline = DataPipeline()
    df_4h = pipeline.get_data(timeframe="4h", use_cache=True)
    
    print("Computing features...")
    fe = FeatureEngine()
    df_featured = fe.compute_all_features(df_4h)

    # 2. SETUP ENGINES
    signal_engine = SignalEngine({"load_models": True})
    
    config = {
        "initial_capital": 10000,
        "slippage_pct": 0.0005,
        "taker_fee_pct": 0.0004,
    }

    print(f"\n{'═' * 60}")
    print(f" CUSTOM YEARLY BACKTEST (2017 - 2026)")
    print(f" Plan: Fixed $1k Margin @ 15x / SL 1.33% / Trail TP")
    print(f"{'═' * 60}\n")

    summary_stats = []

    for year in range(2017, 2027):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        year_df = df_featured[(df_featured.index >= start_date) & (df_featured.index <= end_date)]
        if len(year_df) < 200:
            continue

        risk_engine = RiskEngine(str(PROJECT_ROOT / "configs" / "risk_config.yaml"))
        risk_engine.initial_capital = config["initial_capital"]
        risk_engine.account_balance = config["initial_capital"]
        
        # Override BacktestEngine to use our CustomExecutionEngine
        class CustomBT(BacktestEngine):
            def run_backtest(self, df, sig_eng, risk_eng, start_date=None, end_date=None, show_progress=False):
                # Filter
                if start_date: df = df[df.index >= start_date]
                if end_date: df = df[df.index <= end_date]
                
                execution = CustomExecutionEngine(self.config, sig_eng, risk_eng)
                
                initial_cap = self.config.get('initial_capital', 10000)
                risk_eng.reset_all(initial_cap)
                
                equity_curve = [initial_cap]
                equity_dates = [df.index[0]]
                
                warmup = self.config.get('warmup_periods', 200)
                
                for idx in range(warmup, len(df)):
                    row = df.iloc[idx]
                    execution.run_step(df, idx, row.name)
                    equity_curve.append(risk_eng.account_balance)
                    equity_dates.append(row.name)
                
                self.results = self._calculate_results(execution.trade_history, pd.Series(equity_curve, index=equity_dates), initial_cap)
                return self.results

        bt = CustomBT(config)
        res = bt.run_backtest(df_featured, signal_engine, risk_engine, start_date=start_date, end_date=end_date)

        if "total_trades" in res and res["total_trades"] > 0:
            pnl = res["total_pnl"]
            wr = res["win_rate_pct"]
            trades = res["total_trades"]
            
            color = "\033[32m" if pnl >= 0 else "\033[31m"
            reset = "\033[0m"
            
            print(f"  {year} │ Trades: {trades:>4d} │ Win Rate: {wr:>6.2f}% │ PNL: {color}${pnl:>10,.2f}{reset} │ Bal: ${res['final_equity']:>10,.2f}")
            summary_stats.append({
                "Year": year,
                "Trades": trades,
                "Win Rate": f"{wr:.2f}%",
                "PNL": f"${pnl:,.2f}",
                "Balance": f"${res['final_equity']:,.2f}"
            })
        else:
            print(f"  {year} │ No trades.")

    print(f"\n{'═' * 60}\n")

if __name__ == "__main__":
    run_custom_backtest()
