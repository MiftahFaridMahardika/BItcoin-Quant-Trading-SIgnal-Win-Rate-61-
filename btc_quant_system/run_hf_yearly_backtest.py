"""
BTC Quant Trading System — Optimized High-Frequency Backtest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Trade Plan (Optimized for Frequency):
- POSITION: Fixed $1,000 margin per trade
- LEVERAGE: 15x ($15,000 notional)
- CONCURRENCY: Allow up to 10 concurrent trades (Diversified signals)
- RELAXED FILTERS: 
    * Confidence: 0.5 (was 0.6)
    * R:R: 1.0 (was 1.5)
    * Min Score: 3 (was 4)
- SL: 1.333% from entry
- TP: Trailing after 0.71% move
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
# MULTI-TRADE EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════

class MultiTradeEngine(ExecutionEngine):
    """
    Enhanced execution engine that supports:
    1. Multiple concurrent trades (up to max_positions)
    2. Specific user SL/TP logic
    3. Relaxed threshold filters
    """

    def __init__(self, config, signal_engine, risk_engine, max_positions=10):
        super().__init__(config, signal_engine, risk_engine)
        self.max_positions = max_positions
        self.active_trades: List[Trade] = []
        self.sl_pct = 0.01333
        self.tp_trigger_pct = 0.0071
        self.notional = 15000.0
        
        # Performance overrides (relaxed)
        self.min_confidence = 0.5 
        self.min_rr = 1.0

    def run_step(self, df: pd.DataFrame, idx: int, current_date: datetime) -> Dict:
        row = df.iloc[idx]
        
        # 1. Monitor active trades
        exited_anything = False
        remaining_trades = []
        for trade in self.active_trades:
            exit_check = self._check_custom_exit(trade, row)
            if exit_check['should_exit']:
                self._close_trade(trade, row, exit_check['reason'])
                exited_anything = True
            else:
                remaining_trades.append(trade)
        self.active_trades = remaining_trades

        # 2. Look for NEW signals if capacity allows
        if len(self.active_trades) < self.max_positions:
            signal = self.signal_engine.generate_trading_signal(df, idx)
            
            # Relaxed filters
            if signal.get('signal') != 'SKIP':
                conf = signal.get('confidence', 0)
                rr = signal.get('risk_reward', 0)
                
                if conf >= self.min_confidence and rr >= self.min_rr:
                    new_trade = self._open_custom_trade(signal, row)
                    self.active_trades.append(new_trade)
                    self.trade_history.append(new_trade)

        return {"active": len(self.active_trades), "total_history": len(self.trade_history)}

    def _open_custom_trade(self, signal, row) -> Trade:
        direction = signal['direction']
        raw_price = float(row['Close'])
        
        if direction == 'LONG':
            entry_price = raw_price * (1 + self.slippage_pct)
            stop_loss = entry_price * (1 - self.sl_pct)
        else:
            entry_price = raw_price * (1 - self.slippage_pct)
            stop_loss = entry_price * (1 + self.sl_pct)

        quantity = self.notional / entry_price
        self.trade_count += 1
        
        trade = Trade(
            id=f"T_{self.trade_count:06d}",
            entry_time=row.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            quantity=quantity,
            status="OPEN"
        )
        # Custom peak tracker for trailing
        trade.peak_favorable = entry_price 
        return trade

    def _check_custom_exit(self, trade, row) -> Dict:
        high = float(row.get('high', row.get('High', 0)))
        low = float(row.get('low', row.get('Low', 0)))
        close = float(row.get('close', row.get('Close', 0)))

        # SL Check
        if trade.direction == 'LONG':
            if low <= trade.stop_loss: return {'should_exit': True, 'reason': 'SL'}
            if high > trade.peak_favorable: trade.peak_favorable = high
        else:
            if high >= trade.stop_loss: return {'should_exit': True, 'reason': 'SL'}
            if low < trade.peak_favorable: trade.peak_favorable = low

        # Trailing TP Check
        move = 0
        if trade.direction == 'LONG':
            move = (trade.peak_favorable - trade.entry_price) / trade.entry_price
        else:
            move = (trade.entry_price - trade.peak_favorable) / trade.entry_price

        if move >= self.tp_trigger_pct:
            # Reversal logic (0.3%)
            rev = 0
            if trade.direction == 'LONG':
                rev = (trade.peak_favorable - close) / trade.peak_favorable
            else:
                rev = (close - trade.peak_favorable) / trade.peak_favorable
            
            if rev >= 0.003: return {'should_exit': True, 'reason': 'TRAIL'}

        return {'should_exit': False, 'reason': None}

    def _close_trade(self, trade, row, reason):
        exit_price = float(row['Close'])
        
        # Apply exit slippage
        if trade.direction == 'LONG':
            exit_price *= (1 - self.slippage_pct)
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            exit_price *= (1 + self.slippage_pct)
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

        fees = (self.taker_fee * 2) * trade.quantity * trade.entry_price
        net_pnl = (pnl_pct * trade.quantity * trade.entry_price) - fees
        
        trade.exit_time = row.name
        trade.exit_price = exit_price
        trade.pnl = net_pnl
        trade.pnl_pct = pnl_pct * 100
        trade.status = "CLOSED"
        trade.exit_reason = reason
        
        self.risk_engine.update_balance(net_pnl)

# ══════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════

def run_optimized_yearly_backtest():
    logging.basicConfig(level=logging.ERROR)
    
    pipeline = DataPipeline()
    df_4h = pipeline.get_data(timeframe="4h", use_cache=True)
    fe = FeatureEngine()
    df_featured = fe.compute_all_features(df_4h)
    
    # Relax thresholds in SignalEngine directly for higher frequency
    se = SignalEngine({"load_models": True})
    se.LONG_THRESHOLD = 3
    se.SHORT_THRESHOLD = -3

    config = {
        "initial_capital": 10000,
        "slippage_pct": 0.0005,
        "taker_fee_pct": 0.0004,
    }

    print(f"\n{'═' * 70}")
    print(f" HIGH-FREQUENCY YEARLY BACKTEST (2017 - 2026)")
    print(f" Plan: Multi-Trade (Max 10) / Relaxed Thresholds / 15x Lev")
    print(f"{'═' * 70}\n")

    for year in range(2017, 2027):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        year_df = df_featured[(df_featured.index >= start_date) & (df_featured.index <= end_date)]
        if len(year_df) < 50: continue

        risk_eng = RiskEngine(str(PROJECT_ROOT / "configs" / "risk_config.yaml"))
        risk_eng.account_balance = config["initial_capital"]
        
        engine = MultiTradeEngine(config, se, risk_eng, max_positions=10)
        
        warmup = 100
        for i in range(warmup, len(year_df)):
            engine.run_step(year_df, i, year_df.index[i])

        trades = [t for t in engine.trade_history if t.status == "CLOSED"]
        if trades:
            wins = len([t for t in trades if t.pnl > 0])
            wr = wins / len(trades) * 100
            pnl = sum([t.pnl for t in trades])
            
            color = "\033[32m" if pnl >= 0 else "\033[31m"
            reset = "\033[0m"
            print(f"  {year} │ Trades: {len(trades):>4d} │ WinRate: {wr:>6.2f}% │ PNL: {color}${pnl:>12,.2f}{reset} │ Bal: ${risk_eng.account_balance:>12,.2f}")
        else:
            print(f"  {year} │ No trades.")

    print(f"\n{'═' * 70}\n")

if __name__ == "__main__":
    run_optimized_yearly_backtest()
