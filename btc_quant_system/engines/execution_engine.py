"""
Execution Engine — State Machine Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handles:
1. Trade execution logic (state machine)
2. Backtesting with realistic simulation
3. Trade logging and tracking
4. Performance analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json

try:
    from engines.entry_filters import EntryFilterManager
except ImportError:
    EntryFilterManager = None

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# STATE MACHINE STATES
# ══════════════════════════════════════════════════════════════

class TradeState(Enum):
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    ENTRY_WAIT = "ENTRY_WAIT"   # waiting for pullback entry zone
    SIGNAL_FOUND = "SIGNAL_FOUND"
    PENDING_ENTRY = "PENDING_ENTRY"
    IN_TRADE = "IN_TRADE"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    CLOSING = "CLOSING"


# ══════════════════════════════════════════════════════════════
# TRADE DATA CLASS
# ══════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Trade record."""
    id: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0
    quantity: float = 0.0
    risk_amount: float = 0.0
    pnl: float = 0.0
    pnl_r: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    signal_score: int = 0
    confidence: float = 0.0
    status: str = "OPEN"
    partial_exits: List = field(default_factory=list)
    # Partial exit tracking (populated by adaptive strategy)
    remaining_qty: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    realized_pnl: float = 0.0
    breakeven_activated: bool = False


# ══════════════════════════════════════════════════════════════
# EXECUTION ENGINE (STATE MACHINE)
# ══════════════════════════════════════════════════════════════

class ExecutionEngine:
    """
    State Machine based Execution Engine.

    State transitions:
        IDLE → SCANNING → SIGNAL_FOUND → PENDING_ENTRY → IN_TRADE → CLOSING → SCANNING
                 ↑                                            ↓
                 └────────────────────────────────────────────┘
    """

    def __init__(self, config: dict,
                 signal_engine,
                 risk_engine):
        self.config = config
        self.signal_engine = signal_engine
        self.risk_engine = risk_engine

        # State
        self.state = TradeState.IDLE
        self.current_trade: Optional[Trade] = None
        self.pending_signal: Optional[Dict] = None
        self._pending_exit_reason: Optional[str] = None

        # Tracking
        self.trade_history: List[Trade] = []
        self.trade_count = 0
        self.daily_trades = 0

        # Simulation settings
        self.slippage_pct = config.get('slippage_pct', 0.0005)
        self.maker_fee = config.get('maker_fee_pct', 0.0002)
        self.taker_fee = config.get('taker_fee_pct', 0.0004)
        self.use_adaptive_sltp = config.get('use_adaptive_sltp', True)

        # ── Entry Filters ──────────────────────────────────────────
        if EntryFilterManager is not None:
            self.entry_filter: Optional[EntryFilterManager] = EntryFilterManager(config)
        else:
            self.entry_filter = None

    # ──────────────────────────────────────────────────────
    # CORE STATE MACHINE STEP
    # ──────────────────────────────────────────────────────

    def run_step(self,
                 df: pd.DataFrame,
                 idx: int,
                 current_date: datetime) -> Dict:
        """
        Run single step of state machine.

        Returns dict with keys: state, action, trade
        """
        result: Dict = {"state": self.state.value, "action": None, "trade": None}

        current_row = df.iloc[idx]
        current_price = float(current_row.get('close', current_row.get('Close', 0)))

        # ── IDLE → SCANNING ──────────────────────────────
        if self.state == TradeState.IDLE:
            self.state = TradeState.SCANNING

        # ── SCANNING: look for signals ───────────────────
        elif self.state == TradeState.SCANNING:
            # Check risk limits first
            risk_check = self.risk_engine.check_can_trade()
            if not risk_check['can_trade']:
                result['action'] = 'RISK_BLOCKED'
                return result

            # Generate signal via SignalEngine
            signal = self.signal_engine.generate_trading_signal(df, idx)

            if signal.get('signal') not in ('SKIP', None):
                self.pending_signal = signal
                self.state = TradeState.SIGNAL_FOUND
                result['action'] = 'SIGNAL_FOUND'

        # ── SIGNAL_FOUND: validate ───────────────────────
        elif self.state == TradeState.SIGNAL_FOUND:
            sig = self.pending_signal

            # Reject low R:R
            rr = sig.get('risk_reward', 0)
            if rr < 1.5:
                self.state = TradeState.SCANNING
                self.pending_signal = None
                if self.entry_filter:
                    self.entry_filter.reset_wait()
                result['action'] = 'RR_REJECTED'
                return result

            # Reject low confidence
            conf = sig.get('confidence', 0)
            if conf < 0.6:
                self.state = TradeState.SCANNING
                self.pending_signal = None
                if self.entry_filter:
                    self.entry_filter.reset_wait()
                result['action'] = 'CONFIDENCE_LOW'
                return result

            # ── Entry filter check ────────────────────────
            if self.entry_filter is not None:
                filter_result = self.entry_filter.check_all(
                    df=df, idx=idx, signal=sig, row=current_row
                )
                if not filter_result['allowed']:
                    if filter_result['is_pullback_wait']:
                        # Stay in SIGNAL_FOUND state — keep pending_signal
                        # but move to ENTRY_WAIT so we poll each bar
                        self.state = TradeState.ENTRY_WAIT
                        result['action'] = 'ENTRY_WAIT_PULLBACK'
                    else:
                        self.state = TradeState.SCANNING
                        self.pending_signal = None
                        self.entry_filter.reset_wait()
                        result['action'] = f'ENTRY_FILTERED: {filter_result["reason"]}'
                    return result
                # Approved — carry improved price if pullback gave better entry
                if filter_result.get('improved_price'):
                    self.pending_signal['entry_price'] = filter_result['improved_price']

            self.state = TradeState.PENDING_ENTRY

        # ── ENTRY_WAIT: poll for pullback zone each bar ───
        elif self.state == TradeState.ENTRY_WAIT:
            sig = self.pending_signal
            if self.entry_filter is not None:
                filter_result = self.entry_filter.check_all(
                    df=df, idx=idx, signal=sig, row=current_row
                )
                if filter_result['allowed']:
                    if filter_result.get('improved_price'):
                        self.pending_signal['entry_price'] = filter_result['improved_price']
                    self.state = TradeState.PENDING_ENTRY
                elif not filter_result['is_pullback_wait']:
                    # Timed out or filtered for another reason
                    self.state = TradeState.SCANNING
                    self.pending_signal = None
                    self.entry_filter.reset_wait()
                    result['action'] = 'ENTRY_WAIT_EXPIRED'
                # else: still waiting
                return result
            else:
                self.state = TradeState.PENDING_ENTRY

        # ── PENDING_ENTRY: execute entry ─────────────────
        elif self.state == TradeState.PENDING_ENTRY:
            trade = self._execute_entry(current_row)
            self.current_trade = trade
            if self.entry_filter:
                self.entry_filter.reset_wait()
            self.state = TradeState.IN_TRADE
            result['action'] = 'ENTRY'
            result['trade'] = trade

        # ── IN_TRADE: monitor position ───────────────────
        elif self.state == TradeState.IN_TRADE:
            exit_check = self._check_exit(current_row)

            if exit_check.get('partial_tp'):
                # Partial exit — stay in trade, update trail
                self._execute_partial_exit(current_row, exit_check['partial_tp'])
                self._update_trailing_stop(current_row)
                result['action'] = f'PARTIAL_{exit_check["partial_tp"]}'
            elif exit_check['should_exit']:
                self._pending_exit_reason = exit_check['reason']
                self.state = TradeState.CLOSING
                result['exit_trigger'] = exit_check['reason']
            else:
                self._update_trailing_stop(current_row)

        # ── CLOSING: execute exit ────────────────────────
        elif self.state == TradeState.CLOSING:
            closed_trade = self._execute_exit(
                current_row, exit_reason=self._pending_exit_reason
            )
            self.trade_history.append(closed_trade)

            # Update risk engine balance
            self.risk_engine.update_balance(closed_trade.pnl)

            # Reset state
            self.current_trade = None
            self.pending_signal = None
            self._pending_exit_reason = None
            self.state = TradeState.SCANNING

            result['action'] = 'EXIT'
            result['trade'] = closed_trade

        result['state'] = self.state.value
        return result

    def get_filter_stats(self) -> Dict:
        """Return entry filter pass/block statistics. Returns {} if filters disabled."""
        if self.entry_filter is None:
            return {}
        return self.entry_filter.summary()

    # ──────────────────────────────────────────────────────
    # ENTRY EXECUTION
    # ──────────────────────────────────────────────────────

    def _execute_entry(self, row: pd.Series) -> Trade:
        """Execute entry with slippage, fees, and adaptive SL/TP."""
        signal    = self.pending_signal
        direction = signal['direction']

        # Apply slippage
        raw_price = signal['entry_price']
        entry_price = (
            raw_price * (1 + self.slippage_pct) if direction == 'LONG'
            else raw_price * (1 - self.slippage_pct)
        )

        # ── Adaptive SL/TP ────────────────────────────────────────────
        atr = float(row.get('atr_14', entry_price * 0.02) or entry_price * 0.02)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.02

        if self.use_adaptive_sltp:
            regime  = signal.get('regime', 'NORMAL')
            vol_reg = int(row.get('vol_regime', 1) or 1)
            vol_pct = {0: 15.0, 1: 50.0, 2: 75.0, 3: 95.0}.get(vol_reg, 50.0)
            t_str   = float(signal.get('confidence', 0.5))

            levels = self.risk_engine.calculate_adaptive_sl_tp(
                entry_price=entry_price,
                direction=direction,
                atr=atr,
                regime=regime,
                volatility_percentile=vol_pct,
                trend_strength=t_str,
            )
        else:
            levels = None

        if levels is not None:
            sl  = levels.stop_loss
            tp1 = levels.take_profit_1
            tp2 = levels.take_profit_2
            tp3 = levels.take_profit_3
        else:
            sl  = signal['stop_loss']
            tp1 = signal.get('tp1', signal.get('take_profit_1', entry_price + atr * 2.0))
            tp2 = signal.get('tp2', signal.get('take_profit_2', entry_price + atr * 3.5))
            tp3 = signal.get('tp3', signal.get('take_profit_3', entry_price + atr * 5.0))

        # ── Position sizing ───────────────────────────────────────────
        kelly       = self.risk_engine.kelly_criterion(win_rate=0.58, avg_win_r=2.3)
        risk_amount = self.risk_engine.apply_drawdown_scaling(
            kelly['dollar_risk']
        )['adjusted_risk']

        position = self.risk_engine.calculate_position_size(
            entry_price=entry_price,
            stop_loss=sl,
            risk_amount=risk_amount,
            leverage=self.config.get('leverage', 1),
        )

        self.trade_count  += 1
        self.daily_trades += 1

        return Trade(
            id=f"TRADE_{self.trade_count:06d}",
            entry_time=row.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            quantity=position.quantity,
            risk_amount=risk_amount,
            signal_score=signal.get('score', 0),
            confidence=signal.get('confidence', 0),
            status="OPEN",
            remaining_qty=position.quantity,   # track for partial exits
        )

    # ──────────────────────────────────────────────────────
    # EXIT CHECKS
    # ──────────────────────────────────────────────────────

    def _check_exit(self, row: pd.Series) -> Dict:
        """
        Check exit conditions, including partial TP exits.

        Returns
        -------
        dict with keys:
          should_exit : bool  — True = close full remaining position
          partial_tp  : str | None  — 'TP1' or 'TP2' = partial exit only
          reason      : str
          price       : float
        """
        trade = self.current_trade
        high  = float(row.get('high', row.get('High', 0)))
        low   = float(row.get('low',  row.get('Low',  0)))

        if trade.direction == 'LONG':
            # ── Partial TP1 (40%) — check before SL so optimistic ──────
            if not trade.tp1_hit and high >= trade.take_profit_1:
                return {'should_exit': False, 'partial_tp': 'TP1',
                        'reason': 'TP1', 'price': trade.take_profit_1}
            # ── Partial TP2 (30%) ───────────────────────────────────────
            if trade.tp1_hit and not trade.tp2_hit and high >= trade.take_profit_2:
                return {'should_exit': False, 'partial_tp': 'TP2',
                        'reason': 'TP2', 'price': trade.take_profit_2}
            # ── Full exit: TP3 runner (30%) ─────────────────────────────
            if trade.tp2_hit and high >= trade.take_profit_3:
                return {'should_exit': True, 'partial_tp': None,
                        'reason': 'TP3', 'price': trade.take_profit_3}
            # ── Full exit: SL / trail ───────────────────────────────────
            if low <= trade.stop_loss:
                reason = 'TRAIL_STOP' if trade.breakeven_activated else 'STOP_LOSS'
                return {'should_exit': True, 'partial_tp': None,
                        'reason': reason, 'price': trade.stop_loss}

        else:  # SHORT (symmetric)
            if not trade.tp1_hit and low <= trade.take_profit_1:
                return {'should_exit': False, 'partial_tp': 'TP1',
                        'reason': 'TP1', 'price': trade.take_profit_1}
            if trade.tp1_hit and not trade.tp2_hit and low <= trade.take_profit_2:
                return {'should_exit': False, 'partial_tp': 'TP2',
                        'reason': 'TP2', 'price': trade.take_profit_2}
            if trade.tp2_hit and low <= trade.take_profit_3:
                return {'should_exit': True, 'partial_tp': None,
                        'reason': 'TP3', 'price': trade.take_profit_3}
            if high >= trade.stop_loss:
                reason = 'TRAIL_STOP' if trade.breakeven_activated else 'STOP_LOSS'
                return {'should_exit': True, 'partial_tp': None,
                        'reason': reason, 'price': trade.stop_loss}

        return {'should_exit': False, 'partial_tp': None, 'reason': None, 'price': None}

    # ──────────────────────────────────────────────────────
    # EXIT EXECUTION
    # ──────────────────────────────────────────────────────

    def _execute_exit(self, row: pd.Series,
                      exit_reason: str = None) -> Trade:
        """Execute exit with slippage and fees."""
        trade = self.current_trade

        # Use stored reason from IN_TRADE check, fallback to re-check
        if exit_reason is None:
            exit_check = self._check_exit(row)
            reason = exit_check.get('reason', 'MANUAL')
        else:
            reason = exit_reason

        if reason == 'STOP_LOSS':
            exit_price = trade.stop_loss
        elif reason == 'TP3':
            exit_price = trade.take_profit_3
        elif reason == 'TP2':
            exit_price = trade.take_profit_2
        elif reason == 'TP1':
            exit_price = trade.take_profit_1
        else:
            exit_price = float(row.get('close', row.get('Close', 0)))

        # Apply exit slippage (opposite direction)
        if trade.direction == 'LONG':
            exit_price *= (1 - self.slippage_pct)
        else:
            exit_price *= (1 + self.slippage_pct)

        # Effective remaining quantity for this final exit
        exit_qty = trade.remaining_qty if trade.remaining_qty > 0 else trade.quantity

        # Calculate PnL on remaining slice
        if trade.direction == 'LONG':
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

        # Fees (taker both sides) on remaining qty
        total_fees = (self.taker_fee * 2) * exit_qty * trade.entry_price
        gross_pnl  = pnl_pct * exit_qty * trade.entry_price
        # Add realized PnL from any partial exits already taken
        net_pnl    = gross_pnl - total_fees + trade.realized_pnl

        # PnL in R multiples
        sl_distance = abs(trade.entry_price - trade.stop_loss)
        if sl_distance > 0:
            if trade.direction == 'LONG':
                pnl_r = (exit_price - trade.entry_price) / sl_distance
            else:
                pnl_r = (trade.entry_price - exit_price) / sl_distance
        else:
            pnl_r = 0.0

        # Update trade record
        trade.exit_time = row.name
        trade.exit_price = exit_price
        trade.pnl = net_pnl
        trade.pnl_r = pnl_r
        trade.pnl_pct = pnl_pct * 100
        trade.exit_reason = reason
        trade.status = "CLOSED"

        return trade

    # ──────────────────────────────────────────────────────
    # PARTIAL EXIT
    # ──────────────────────────────────────────────────────

    def _execute_partial_exit(self, row: pd.Series, level: str) -> float:
        """
        Execute partial position exit at TP1 or TP2.

        TP1 (40%): lock profit, move SL to breakeven — SL never goes below entry.
        TP2 (30%): lock more profit, trail SL at 1× ATR, clamped to breakeven.

        Returns realized PnL for the exited slice.
        """
        trade = self.current_trade
        close = float(row.get('close', row.get('Close', 0)))
        atr   = float(row.get('atr_14', row.get('atr', trade.entry_price * 0.02)) or
                      trade.entry_price * 0.02)

        if level == 'TP1':
            exit_price = trade.take_profit_1
            exit_qty   = trade.quantity * 0.40
            trade.tp1_hit = True
            trade.breakeven_activated = True
            # Move SL to breakeven — never below entry for LONG, never above for SHORT
            if trade.direction == 'LONG':
                trade.stop_loss = max(trade.stop_loss, trade.entry_price)
            else:
                trade.stop_loss = min(trade.stop_loss, trade.entry_price)

        elif level == 'TP2':
            exit_price = trade.take_profit_2
            exit_qty   = trade.quantity * 0.30
            trade.tp2_hit = True
            # Trail at 1× ATR from current close, but never worse than breakeven
            if trade.direction == 'LONG':
                new_sl = close - atr
                # Clamp: must be at least at breakeven (entry price)
                new_sl = max(new_sl, trade.entry_price)
                if new_sl > trade.stop_loss:
                    trade.stop_loss = new_sl
            else:
                new_sl = close + atr
                # Clamp: must be at most at breakeven (entry price)
                new_sl = min(new_sl, trade.entry_price)
                if new_sl < trade.stop_loss:
                    trade.stop_loss = new_sl
        else:
            return 0.0

        trade.remaining_qty = max(trade.remaining_qty - exit_qty, 0.0)

        # PnL for the exited slice
        if trade.direction == 'LONG':
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

        fees     = self.taker_fee * 2 * exit_qty * trade.entry_price
        gross    = pnl_pct * exit_qty * trade.entry_price
        net      = gross - fees

        trade.realized_pnl += net
        trade.partial_exits.append({
            'level': level, 'price': exit_price,
            'qty': exit_qty, 'pnl': net, 'time': row.name,
        })
        return net

    # ──────────────────────────────────────────────────────
    # TRAILING STOP
    # ──────────────────────────────────────────────────────

    def _update_trailing_stop(self, row: pd.Series):
        """
        Tiered trailing stop — only activates after minimum +0.5× ATR profit.
        Prevents the SL from immediately moving closer to entry on flat/losing bars.

        Tier 0 (< +0.5 ATR profit) : SL stays fixed at entry-level — NO trailing
        Tier 1 (≥ +0.5 ATR profit) : breakeven lock (SL ≥ entry)
        Tier 2 (≥ +1 ATR profit)   : trail at 1.5× ATR
        Tier 3 (≥ +2 ATR profit)   : trail at 1.0× ATR
        Tier 4 (≥ +3 ATR profit)   : trail at 0.7× ATR (tightest)
        """
        if self.current_trade is None:
            return

        trade = self.current_trade
        close = float(row.get('close', row.get('Close', 0)))
        atr   = float(row.get('atr_14', row.get('atr', close * 0.02)) or close * 0.02)

        if trade.direction == 'LONG':
            profit     = close - trade.entry_price          # absolute profit
            atr_thresh = atr * 0.5                           # minimum profit before trailing

            if profit >= 3 * atr:                            # Tier 4 — tightest
                new_sl = close - 0.7 * atr
            elif profit >= 2 * atr:                          # Tier 3
                new_sl = close - 1.0 * atr
            elif profit >= atr:                              # Tier 2
                new_sl = max(trade.entry_price, close - 1.5 * atr)
                if not trade.breakeven_activated:
                    trade.breakeven_activated = True
            elif profit >= atr_thresh:                       # Tier 1 — breakeven only
                new_sl = trade.entry_price
                if not trade.breakeven_activated:
                    trade.breakeven_activated = True
            else:
                return  # Tier 0 — too early, keep SL fixed

            # SL can only move UP for LONG
            if new_sl > trade.stop_loss:
                trade.stop_loss = new_sl

        else:  # SHORT (symmetric)
            profit     = trade.entry_price - close
            atr_thresh = atr * 0.5

            if profit >= 3 * atr:                            # Tier 4
                new_sl = close + 0.7 * atr
            elif profit >= 2 * atr:                          # Tier 3
                new_sl = close + 1.0 * atr
            elif profit >= atr:                              # Tier 2
                new_sl = min(trade.entry_price, close + 1.5 * atr)
                if not trade.breakeven_activated:
                    trade.breakeven_activated = True
            elif profit >= atr_thresh:                       # Tier 1 — breakeven only
                new_sl = trade.entry_price
                if not trade.breakeven_activated:
                    trade.breakeven_activated = True
            else:
                return  # Tier 0 — too early, keep SL fixed

            # SL can only move DOWN for SHORT
            if new_sl < trade.stop_loss:
                trade.stop_loss = new_sl

    # ──────────────────────────────────────────────────────
    # RESET HELPERS
    # ──────────────────────────────────────────────────────

    def reset_daily(self):
        """Reset daily counters."""
        self.daily_trades = 0
        self.risk_engine.reset_daily()

    def reset_all(self):
        """Full reset."""
        self.state = TradeState.IDLE
        self.current_trade = None
        self.pending_signal = None
        self._pending_exit_reason = None
        self.trade_history = []
        self.trade_count = 0
        self.daily_trades = 0
        self.risk_engine.reset_all()


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Full backtesting engine with realistic simulation.

    Features:
    - Slippage and fee modelling
    - Warmup period handling
    - Daily state resets
    - Comprehensive results calculation
    """

    def __init__(self, config: dict):
        self.config = config
        self.results: Optional[Dict] = None

    # ──────────────────────────────────────────────────────
    # MAIN BACKTEST LOOP
    # ──────────────────────────────────────────────────────

    def run_backtest(self,
                     df: pd.DataFrame,
                     signal_engine,
                     risk_engine,
                     start_date: str = None,
                     end_date: str = None,
                     show_progress: bool = True) -> Dict:
        """
        Run complete backtest over a DataFrame of featured OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame — featured data with signal columns
        signal_engine : SignalEngine
        risk_engine : RiskEngine
        start_date / end_date : optional date filter (ISO format)
        show_progress : print progress bar

        Returns
        -------
        dict — comprehensive results
        """
        # Filter date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        logger.info(f"Running backtest: {df.index[0]} → {df.index[-1]}")
        logger.info(f"Total candles: {len(df):,}")

        # Initialise
        initial_capital = self.config.get('initial_capital', 100_000)
        risk_engine.reset_all(initial_capital)

        execution = ExecutionEngine(
            self.config,
            signal_engine,
            risk_engine,
        )

        # Equity tracking
        equity_curve = [initial_capital]
        equity_dates = [df.index[0]]

        # Warmup (skip first N candles for indicators)
        warmup = self.config.get('warmup_periods', 200)

        # Progress
        total_steps = len(df) - warmup
        progress_interval = max(total_steps // 20, 1)

        # Main loop
        current_date = None
        for idx in range(warmup, len(df)):
            row = df.iloc[idx]

            # Daily reset
            if current_date is None:
                current_date = row.name.date() if hasattr(row.name, 'date') else None
            elif hasattr(row.name, 'date') and row.name.date() != current_date:
                current_date = row.name.date()
                execution.reset_daily()

            # Run state machine step
            result = execution.run_step(df, idx, row.name)

            # Track equity
            equity = risk_engine.account_balance
            equity_curve.append(equity)
            equity_dates.append(row.name)

            # Progress logging
            if show_progress and (idx - warmup) % progress_interval == 0:
                progress = (idx - warmup) / total_steps * 100
                logger.info(
                    f"  Progress: {progress:5.1f}% │ "
                    f"Equity: ${equity:>12,.2f} │ "
                    f"Trades: {len(execution.trade_history)}"
                )

        # Compile results
        trades = execution.trade_history
        equity_series = pd.Series(equity_curve, index=equity_dates)

        self.results = self._calculate_results(
            trades, equity_series, initial_capital
        )
        return self.results

    # ──────────────────────────────────────────────────────
    # RESULTS CALCULATION
    # ──────────────────────────────────────────────────────

    def _calculate_results(self,
                           trades: List[Trade],
                           equity: pd.Series,
                           initial_capital: float) -> Dict:
        """Calculate comprehensive backtest results."""

        if not trades:
            return {"error": "No trades executed", "total_trades": 0}

        # Basic metrics
        final_equity = float(equity.iloc[-1])
        total_return = (final_equity - initial_capital) / initial_capital

        # Trade statistics
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        breakeven = [t for t in trades if t.pnl == 0]

        win_rate = len(wins) / len(trades) if trades else 0

        avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([abs(t.pnl) for t in losses])) if losses else 0.0

        avg_win_r = float(np.mean([t.pnl_r for t in wins])) if wins else 0.0
        avg_loss_r = float(np.mean([abs(t.pnl_r) for t in losses])) if losses else 0.0

        # Returns series
        returns = equity.pct_change().dropna()

        # Risk metrics
        max_dd_info = self._calc_max_dd(equity)
        sharpe = self._calc_sharpe(returns)
        sortino = self._calc_sortino(returns)

        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

        expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

        # CAGR
        if hasattr(equity.index[0], 'date'):
            days = (equity.index[-1] - equity.index[0]).days
        else:
            days = len(equity)

        years = days / 365.25 if days > 0 else 1.0
        cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

        max_dd = max_dd_info['max_drawdown']
        calmar = cagr / max_dd if max_dd > 0 else np.inf

        # Max consecutive losses
        max_consec_losses = self._max_consecutive_losses(trades)

        # ── Exit reason distribution ──────────────────────────────────
        exit_reasons: Dict[str, int] = {}
        for t in trades:
            r = t.exit_reason or 'UNKNOWN'
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        n_trades = len(trades)
        sl_exits    = exit_reasons.get('STOP_LOSS', 0)
        trail_exits = exit_reasons.get('TRAIL_STOP', 0)
        tp3_exits   = exit_reasons.get('TP3', 0)
        time_exits  = exit_reasons.get('TIME_EXIT', 0)
        # Any trade that touched TP1 is a partial winner
        partial_tp1 = sum(1 for t in trades if t.tp1_hit)
        partial_tp2 = sum(1 for t in trades if t.tp2_hit)

        return {
            # Capital
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_pnl": final_equity - initial_capital,

            # Returns
            "total_return_pct": total_return * 100,
            "cagr_pct": cagr * 100,

            # Trade Stats
            "total_trades": n_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "breakeven_trades": len(breakeven),
            "win_rate_pct": win_rate * 100,

            # Averages
            "avg_win_usd": avg_win,
            "avg_loss_usd": avg_loss,
            "avg_win_r": avg_win_r,
            "avg_loss_r": avg_loss_r,

            # Risk Metrics
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown_pct": max_dd * 100,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy,
            "max_consecutive_losses": max_consec_losses,

            # Exit reason breakdown
            "exit_reasons": exit_reasons,
            "sl_exit_pct":    sl_exits    / n_trades * 100 if n_trades else 0,
            "trail_exit_pct": trail_exits / n_trades * 100 if n_trades else 0,
            "tp3_exit_pct":   tp3_exits   / n_trades * 100 if n_trades else 0,
            "time_exit_pct":  time_exits  / n_trades * 100 if n_trades else 0,
            "partial_tp1_pct": partial_tp1 / n_trades * 100 if n_trades else 0,
            "partial_tp2_pct": partial_tp2 / n_trades * 100 if n_trades else 0,

            # Data
            "equity_curve": equity.to_dict(),
            "trades": [self._trade_to_dict(t) for t in trades],
        }

    # ──────────────────────────────────────────────────────
    # HELPER METRICS
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _calc_max_dd(equity: pd.Series) -> Dict:
        """Calculate max drawdown from equity curve."""
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        return {
            "max_drawdown": abs(float(drawdowns.min())),
            "max_dd_date": drawdowns.idxmin(),
        }

    @staticmethod
    def _calc_sharpe(returns: pd.Series, rf: float = 0.05) -> float:
        """Annualised Sharpe ratio."""
        excess = returns - (rf / 252)
        std = excess.std()
        if std == 0:
            return 0.0
        return float(np.sqrt(252) * excess.mean() / std)

    @staticmethod
    def _calc_sortino(returns: pd.Series, rf: float = 0.05) -> float:
        """Annualised Sortino ratio."""
        excess = returns - (rf / 252)
        downside = returns[returns < 0].std()
        if downside == 0:
            return float('inf')
        return float(np.sqrt(252) * excess.mean() / downside)

    @staticmethod
    def _max_consecutive_losses(trades: List[Trade]) -> int:
        """Count longest losing streak."""
        max_streak = 0
        current = 0
        for t in trades:
            if t.pnl < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @staticmethod
    def _trade_to_dict(trade: Trade) -> Dict:
        """Convert Trade dataclass to serialisable dict."""
        return {
            "id": trade.id,
            "entry_time": str(trade.entry_time),
            "exit_time": str(trade.exit_time),
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "stop_loss": trade.stop_loss,
            "quantity": trade.quantity,
            "pnl": trade.pnl,
            "pnl_r": trade.pnl_r,
            "pnl_pct": trade.pnl_pct,
            "exit_reason": trade.exit_reason,
            "signal_score": trade.signal_score,
            "confidence": trade.confidence,
            "tp1_hit": trade.tp1_hit,
            "tp2_hit": trade.tp2_hit,
            "realized_pnl": trade.realized_pnl,
            "breakeven_activated": trade.breakeven_activated,
            "partial_exits": trade.partial_exits,
        }

    # ──────────────────────────────────────────────────────
    # SAVE
    # ──────────────────────────────────────────────────────

    def save_results(self, filepath: str):
        """Save results to JSON."""
        if self.results is None:
            raise ValueError("No results to save — run backtest first.")

        results_copy = self.results.copy()
        # Serialise equity curve keys
        results_copy['equity_curve'] = {
            str(k): v for k, v in results_copy['equity_curve'].items()
        }

        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")
