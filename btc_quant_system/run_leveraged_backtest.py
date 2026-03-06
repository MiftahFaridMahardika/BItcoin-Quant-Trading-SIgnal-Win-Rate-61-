#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════
  BTC QUANT SYSTEM — LEVERAGED BACKTEST
  Capital  : $1,000 (modal awal)
  Leverage : 10x
  Hard SL  : 20% of account per trade (max risk = $200/trade awal)
  Period   : 2019-2024 (year-by-year)
══════════════════════════════════════════════════════════════

Mekanisme Leverage 10x + Hard SL 20%:
─────────────────────────────────────
  • Per trade, margin dialokasikan = Kelly% × capital
  • Position notional = margin × 10x
  • Loss cap per trade = 20% × current capital
  • Liquidation guard: jika ATR-SL > 10% harga (= 1/leverage),
    maka loss di-cap ke 100% margin (bukan 20%)
  • Drawdown scaling tetap aktif (reduce size saat DD besar)
  • Jika capital < $50 → account blown up

Contoh ($1,000):
  Trade masuk LONG BTC@$30,000, Kelly = 20% → risk = $200
  Position notional = $200 / sl_pct (mis. sl_pct=3%) = $6,667
  Margin = $6,667 / 10 = $667
  Jika SL hit: lose $200 (20% account) ✓
  Jika TP3 (pnl_r = 3R): gain = 3 × $200 = $600 (+60% account!)
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

from engines.signal_engine import SignalEngine
from engines.risk_engine import RiskEngine
from engines.execution_engine import BacktestEngine, ExecutionEngine, Trade, TradeState
from engines.trend_follower import TrendAwareSignalEngine

# ── ANSI ─────────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"
SEP = f"{C}{'═' * 68}{R}"

logging.basicConfig(
    level=logging.WARNING,
    format=f"  {D}%(asctime)s{R} │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lev_bt")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

INITIAL_CAPITAL = 1_000.0
LEVERAGE        = 10
HARD_SL_PCT     = 0.20          # 20% of capital per trade
LIQUIDATION_PCT = 1.0 / LEVERAGE  # 10% price drop = liquidation

YEARS = list(range(2019, 2025))

# Backtest engine config
BT_CONFIG = {
    "initial_capital":   INITIAL_CAPITAL,
    "slippage_pct":      0.0005,
    "maker_fee_pct":     0.0002,
    "taker_fee_pct":     0.0004,
    "leverage":          LEVERAGE,
    "warmup_periods":    0,
    "use_adaptive_sltp": True,
    "use_pullback_entry": True,
    "max_pullback_wait":  5,
    # Risk
    "max_risk_per_trade": HARD_SL_PCT,   # 20%  ← hard SL in dollar terms
}

YEAR_NOTES = {
    2019: "Bull Recovery",
    2020: "COVID + Rally",
    2021: "Peak Bull",
    2022: "Bear Market",
    2023: "Recovery",
    2024: "ETF Bull",
}


# ══════════════════════════════════════════════════════════════
# HIGH-SELECTIVITY SIGNAL WRAPPER (same as optimized)
# ══════════════════════════════════════════════════════════════

class HighSelectivityEngine:
    def __init__(self, base_engine, min_confidence=0.70):
        self._engine = base_engine
        self.min_confidence = min_confidence

    def get_current_bias(self):
        return self._engine.get_current_bias()

    def calculate_signal_score(self, df, idx):
        return self._engine.signal_engine.calculate_signal_score(df, idx)

    def generate_signals_batch(self, df, start_idx=1):
        return self._engine.signal_engine.generate_signals_batch(df, start_idx)

    def generate_trading_signal(self, df, idx):
        sig = self._engine.generate_trading_signal(df, idx)
        if sig.get("signal") not in ("SKIP", None):
            if sig.get("confidence", 0.0) < self.min_confidence:
                return {
                    "signal": "SKIP",
                    "score": sig.get("score", 0),
                    "regime": sig.get("regime", "NORMAL"),
                    "timestamp": sig.get("timestamp"),
                    "reasons": sig.get("reasons", []) + ["[QualityFilter] conf below 0.70"],
                    "market_bias": sig.get("market_bias", "NEUTRAL"),
                }
        return sig


# ══════════════════════════════════════════════════════════════
# LEVERAGED EXECUTION ENGINE
# Overrides ExecutionEngine to:
#  1. Cap per-trade loss at hard_sl_pct × account
#  2. Apply liquidation guard (loss ≤ margin_required)
#  3. Amplify wins with 10x position notional
# ══════════════════════════════════════════════════════════════

class LeveragedExecutionEngine(ExecutionEngine):
    """
    ExecutionEngine with 10x leverage semantics.

    Key differences:
      • position is sized with leverage parameter = 10
      • per-trade dollar_risk = hard_sl_pct × current_balance (capped)
      • winning trades: P&L = pnl_r × dollar_risk (same as base)
      • losing trades:  P&L = -dollar_risk (same as base)
      • LIQUIDATION GUARD: if ATR_sl_pct > 1/leverage, loss capped
        at margin_required (= position_notional / leverage)
    """

    def __init__(self, config, signal_engine, risk_engine):
        super().__init__(config, signal_engine, risk_engine)
        self.leverage       = config.get("leverage", 10)
        self.hard_sl_pct    = config.get("max_risk_per_trade", 0.20)
        self.liq_threshold  = 1.0 / self.leverage   # 10% for 10x

        # Liquidation stats
        self.liquidation_count = 0

    def _execute_entry(self, row):
        """Override entry with leverage-aware position sizing."""
        signal    = self.pending_signal
        direction = signal["direction"]

        raw_price = signal["entry_price"]
        entry_price = (
            raw_price * (1 + self.slippage_pct) if direction == "LONG"
            else raw_price * (1 - self.slippage_pct)
        )

        # Adaptive SL/TP
        import pandas as pd
        atr = float(row.get("atr_14", entry_price * 0.02) or entry_price * 0.02)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.02

        if self.use_adaptive_sltp:
            regime  = signal.get("regime", "NORMAL")
            vol_reg = int(row.get("vol_regime", 1) or 1)
            vol_pct = {0: 15.0, 1: 50.0, 2: 75.0, 3: 95.0}.get(vol_reg, 50.0)
            t_str   = float(signal.get("confidence", 0.5))
            levels  = self.risk_engine.calculate_adaptive_sl_tp(
                entry_price=entry_price, direction=direction, atr=atr,
                regime=regime, volatility_percentile=vol_pct, trend_strength=t_str,
            )
        else:
            levels = None

        if levels is not None:
            sl  = levels.stop_loss
            tp1 = levels.take_profit_1
            tp2 = levels.take_profit_2
            tp3 = levels.take_profit_3
        else:
            sl  = signal["stop_loss"]
            tp1 = signal.get("tp1", entry_price + atr * 2.0)
            tp2 = signal.get("tp2", entry_price + atr * 3.5)
            tp3 = signal.get("tp3", entry_price + atr * 5.0)

        # ── Leverage-aware risk sizing ─────────────────────────
        balance   = self.risk_engine.account_balance
        # Hard SL dollar amount (20% of account, scaled by drawdown)
        base_risk = balance * self.hard_sl_pct
        risk_res  = self.risk_engine.apply_drawdown_scaling(base_risk)
        risk_amount = risk_res["adjusted_risk"]

        # SL distance as fraction of entry price
        sl_pct = abs(entry_price - sl) / entry_price if entry_price > 0 else 0.02

        # ── LIQUIDATION GUARD ──────────────────────────────────
        # If ATR SL is wider than liquidation threshold (10% at 10x),
        # cap sl_pct at liq_threshold for position sizing
        # (trade will be flagged as liquidation-risk)
        liq_risk = sl_pct > self.liq_threshold
        eff_sl_pct = min(sl_pct, self.liq_threshold) if liq_risk else sl_pct
        if eff_sl_pct <= 0:
            eff_sl_pct = 0.02

        # Position notional and margin
        # position_notional = risk_amount / eff_sl_pct
        # margin             = position_notional / leverage
        position = self.risk_engine.calculate_position_size(
            entry_price=entry_price,
            stop_loss=entry_price * (1 - eff_sl_pct) if direction == "LONG"
                      else entry_price * (1 + eff_sl_pct),
            risk_amount=risk_amount,
            leverage=self.leverage,
        )

        self.trade_count  += 1
        self.daily_trades += 1

        trade = Trade(
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
            signal_score=signal.get("score", 0),
            confidence=signal.get("confidence", 0),
            status="OPEN",
            remaining_qty=position.quantity,
        )

        # Store margin and liq flag as custom attrs
        trade._margin_used   = position.margin_required
        trade._liq_risk      = liq_risk
        trade._eff_sl_pct    = eff_sl_pct
        return trade

    def _execute_exit(self, row, exit_reason=None):
        """Override exit with liquidation cap."""
        import pandas as pd

        trade = self.current_trade
        if exit_reason is None:
            exit_check = self._check_exit(row)
            reason = exit_check.get("reason", "MANUAL")
        else:
            reason = exit_reason

        if reason == "STOP_LOSS":
            exit_price = trade.stop_loss
        elif reason == "TP3":
            exit_price = trade.take_profit_3
        elif reason == "TP2":
            exit_price = trade.take_profit_2
        elif reason == "TP1":
            exit_price = trade.take_profit_1
        else:
            exit_price = float(row.get("close", row.get("Close", 0)))

        # Slippage
        if trade.direction == "LONG":
            exit_price *= (1 - self.slippage_pct)
        else:
            exit_price *= (1 + self.slippage_pct)

        exit_qty = trade.remaining_qty if trade.remaining_qty > 0 else trade.quantity

        # Raw P&L on position
        if trade.direction == "LONG":
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

        total_fees = (self.taker_fee * 2) * exit_qty * trade.entry_price
        gross_pnl  = pnl_pct * exit_qty * trade.entry_price
        net_pnl    = gross_pnl - total_fees + trade.realized_pnl

        # ── LIQUIDATION CAP ────────────────────────────────────
        # Max loss cannot exceed what was staked as margin
        margin = getattr(trade, "_margin_used", trade.risk_amount)
        if net_pnl < -margin:
            net_pnl = -margin
            reason += "_LIQ"
            self.liquidation_count += 1

        # P&L in R multiples (based on original SL distance)
        sl_distance = abs(trade.entry_price - trade.stop_loss)
        if sl_distance > 0:
            if trade.direction == "LONG":
                pnl_r = (exit_price - trade.entry_price) / sl_distance
            else:
                pnl_r = (trade.entry_price - exit_price) / sl_distance
        else:
            pnl_r = 0.0

        trade.exit_time  = row.name
        trade.exit_price = exit_price
        trade.pnl        = net_pnl
        trade.pnl_r      = pnl_r
        trade.pnl_pct    = pnl_pct * 100
        trade.exit_reason = reason
        trade.status     = "CLOSED"
        return trade


# ══════════════════════════════════════════════════════════════
# LEVERAGED BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

class LeveragedBacktestEngine(BacktestEngine):
    """BacktestEngine using LeveragedExecutionEngine."""

    def run_backtest(self, df, signal_engine, risk_engine,
                     start_date=None, end_date=None, show_progress=False):
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        if df.empty:
            return {"error": "No data in range", "total_trades": 0}

        initial_capital = self.config.get("initial_capital", 1_000)
        risk_engine.reset_all(initial_capital)

        # Use LeveragedExecutionEngine instead of base ExecutionEngine
        execution = LeveragedExecutionEngine(self.config, signal_engine, risk_engine)

        equity_curve   = [initial_capital]
        equity_dates   = [df.index[0]]
        warmup         = self.config.get("warmup_periods", 0)
        current_date   = None

        for idx in range(warmup, len(df)):
            # Account blown up
            if risk_engine.account_balance < 1.0:
                logger.warning("Account blown up!")
                break

            row = df.iloc[idx]
            if current_date is None:
                current_date = row.name.date() if hasattr(row.name, "date") else None
            elif hasattr(row.name, "date") and row.name.date() != current_date:
                current_date = row.name.date()
                execution.reset_daily()

            execution.run_step(df, idx, row.name)

            equity = max(risk_engine.account_balance, 0.0)
            equity_curve.append(equity)
            equity_dates.append(row.name)

        trades = execution.trade_history
        equity_series = pd.Series(equity_curve, index=equity_dates)

        results = self._calculate_results(trades, equity_series, initial_capital)
        results["liquidation_count"] = execution.liquidation_count
        return results


# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════

def load_data():
    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    df = pd.read_parquet(feat_path)
    df = df[(df.index >= "2017-01-01") & (df.index <= "2024-12-31 23:59:59")]
    return df


# ══════════════════════════════════════════════════════════════
# RUN YEAR-BY-YEAR
# ══════════════════════════════════════════════════════════════

def run_leveraged_yearly(df, signal_engine):
    all_results = {}
    running_capital = INITIAL_CAPITAL   # ← capital CARRIES OVER year-to-year

    for year in YEARS:
        start = f"{year}-01-01"
        end   = f"{year}-12-31"

        mask = (df.index >= pd.Timestamp(start, tz="UTC")) & \
               (df.index <= pd.Timestamp(end, tz="UTC"))
        if not mask.any():
            mask = (df.index >= start) & (df.index <= end)
        if not mask.any():
            all_results[str(year)] = {"status": "no_data", "capital": running_capital}
            continue

        if running_capital < 1.0:
            all_results[str(year)] = {"status": "blown_up", "capital": 0.0}
            continue

        # Risk engine initialised with RUNNING capital (carries over!)
        risk_engine = RiskEngine(str(PROJECT_ROOT / "configs" / "risk_config.yaml"))
        risk_engine.max_risk_per_trade = HARD_SL_PCT   # 20%
        risk_engine.max_total_drawdown = 0.99           # don't auto-stop, show real picture
        risk_engine.max_daily_loss     = 0.99           # don't block on daily loss

        # Override drawdown scaling to show realistic leverage scenario
        # Keep scaling active so we can see capital protection
        config_year = dict(BT_CONFIG)
        config_year["initial_capital"] = running_capital

        backtest = LeveragedBacktestEngine(config_year)
        results  = backtest.run_backtest(
            df=df,
            signal_engine=signal_engine,
            risk_engine=risk_engine,
            start_date=start,
            end_date=end,
            show_progress=False,
        )

        if not results or results.get("total_trades", 0) == 0:
            all_results[str(year)] = {
                "status": "no_trades",
                "capital_start": running_capital,
                "capital_end": running_capital,
                "total_trades": 0,
            }
            continue

        capital_end = results.get("final_equity", running_capital)
        pnl         = capital_end - running_capital
        pct_return  = (capital_end - running_capital) / running_capital * 100

        all_results[str(year)] = {
            "status": "ok",
            "capital_start": round(running_capital, 2),
            "capital_end":   round(capital_end, 2),
            "pnl":           round(pnl, 2),
            "return_pct":    round(pct_return, 4),
            "total_trades":  results["total_trades"],
            "winning_trades": results.get("winning_trades", 0),
            "losing_trades":  results.get("losing_trades", 0),
            "win_rate_pct":  round(results["win_rate_pct"], 2),
            "max_drawdown_pct": round(results["max_drawdown_pct"], 4),
            "sharpe_ratio":  round(results.get("sharpe_ratio", 0), 4),
            "profit_factor": round(results.get("profit_factor", 0), 4),
            "avg_win_r":     round(results.get("avg_win_r", 0), 4),
            "avg_loss_r":    round(results.get("avg_loss_r", 0), 4),
            "sl_exit_pct":   round(results.get("sl_exit_pct", 0), 2),
            "tp3_exit_pct":  round(results.get("tp3_exit_pct", 0), 2),
            "liquidations":  results.get("liquidation_count", 0),
            "exit_reasons":  results.get("exit_reasons", {}),
            "trades":        results.get("trades", []),
        }

        running_capital = max(capital_end, 0.0)

    all_results["FINAL_CAPITAL"] = round(running_capital, 2)
    return all_results


# ══════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_report(all_results):
    final_capital = all_results.get("FINAL_CAPITAL", 0.0)
    total_return  = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_pnl     = final_capital - INITIAL_CAPITAL

    print(f"""
{C}╔{'═' * 70}╗
║{'LEVERAGED BACKTEST REPORT — 10x Leverage + Hard SL 20%':^70s}║
║{'Modal Awal: $1,000 │ Period: 2019-2024 │ Capital KUMULATIF':^70s}║
╠{'═' * 70}╣{R}""")

    print(f"""║
║  {B}PARAMETER AKTIF{R}
║  {'─' * 50}
║  Modal Awal       : $1,000
║  Leverage         : 10x (position notional = margin × 10)
║  Hard SL per trade: 20% of capital (max loss $200 dari $1,000)
║  Liquidation at   : 10% adverse price move (= 100% margin)
║  Signal Filter    : min_confidence = 0.70
║  Engine           : TrendAwareSignalEngine (bias-adjusted)
║  Capital          : KUMULATIF antar tahun (bukan reset)
║""")

    print(f"""{C}║{'═' * 70}║{R}""")
    print(f"""║  {'YEAR':<6} │ {'START':>9} │ {'END':>9} │ {'PNL':>10} │ {'RETURN':>8} │ {'TRADES':>7} │ {'WR%':>6} │ STATUS""")
    print(f"""║  {'─' * 6}─┼─{'─' * 9}─┼─{'─' * 9}─┼─{'─' * 10}─┼─{'─' * 8}─┼─{'─' * 7}─┼─{'─' * 6}─┼─{'─' * 12}""")

    for year in YEARS:
        yr = all_results.get(str(year), {})
        status = yr.get("status", "no_data")

        if status in ("blown_up", "no_data"):
            lbl = f"{RED}BLOWN UP{R}" if status == "blown_up" else f"{Y}NO DATA{R}"
            print(f"║  {year:<6} │ {'—':>9} │ {'—':>9} │ {'—':>10} │ {'—':>8} │ {'—':>7} │ {'—':>6} │ {lbl}")
            continue

        if status == "no_trades":
            cs = yr.get("capital_start", 0)
            print(f"║  {year:<6} │ ${cs:>8,.0f} │ ${cs:>8,.0f} │ {'$0':>10} │ {'0.00%':>8} │ {'0':>7} │ {'—':>6} │ {Y}No trades{R}")
            continue

        cs   = yr["capital_start"]
        ce   = yr["capital_end"]
        pnl  = yr["pnl"]
        ret  = yr["return_pct"]
        t    = yr["total_trades"]
        wr   = yr["win_rate_pct"]
        liq  = yr.get("liquidations", 0)
        note = YEAR_NOTES.get(year, "")

        col = G if pnl >= 0 else RED
        liq_str = f" {RED}LIQ:{liq}{R}" if liq > 0 else ""
        dd  = yr.get("max_drawdown_pct", 0)
        dd_col = RED if dd > 30 else Y if dd > 15 else G

        print(
            f"║  {year:<6} │ ${cs:>8,.0f} │ "
            f"{col}${ce:>8,.0f}{R} │ "
            f"{col}${pnl:>+9,.0f}{R} │ "
            f"{col}{ret:>+7.2f}%{R} │ "
            f"{t:>7} │ "
            f"{wr:>5.1f}% │ "
            f"{dd_col}DD:{dd:.1f}%{R}{liq_str}  {D}{note}{R}"
        )

    # Final
    col = G if total_pnl >= 0 else RED
    mult = final_capital / INITIAL_CAPITAL

    print(f"""║
{C}╠{'═' * 70}╣{R}
║  {'TOTAL (6 tahun)':^68s}║
║{'─' * 70}║
║  Modal Awal   : {G}$1,000{R}
║  Modal Akhir  : {col}${final_capital:,.2f}{R}
║  Total PnL    : {col}${total_pnl:>+,.2f}{R}
║  Total Return : {col}{total_return:>+.2f}%{R}
║  Multiplier   : {col}{mult:.2f}x{R}  {'(+profit)' if mult >= 1 else '(−rugi)'}
║
{C}╚{'═' * 70}╝{R}""")

    # ── Per-year detail ────────────────────────────────────────
    print(f"\n{B}  DETAIL PER TAHUN{R}")
    print(f"  {'═' * 95}")

    for year in YEARS:
        yr = all_results.get(str(year), {})
        if yr.get("status") not in ("ok",):
            continue

        print(f"\n  {C}{B}[ {year} ] — {YEAR_NOTES.get(year,'')} {R}")
        print(f"  Start: ${yr['capital_start']:,.2f}  →  End: ${yr['capital_end']:,.2f}  "
              f"({yr['return_pct']:+.2f}%)")
        print(f"  Trades: {yr['total_trades']}  │  "
              f"Win: {yr['winning_trades']} ({yr['win_rate_pct']:.1f}%)  │  "
              f"Loss: {yr['losing_trades']}  │  "
              f"Max DD: {yr['max_drawdown_pct']:.2f}%  │  "
              f"Sharpe: {yr['sharpe_ratio']:.3f}")
        print(f"  Avg Win R: {yr['avg_win_r']:.2f}R  │  "
              f"Avg Loss R: {yr['avg_loss_r']:.2f}R  │  "
              f"PF: {yr['profit_factor']:.2f}  │  "
              f"Liquidations: {yr['liquidations']}")

        # Exit reasons
        er = yr.get("exit_reasons", {})
        if er:
            total_t = yr["total_trades"]
            items = sorted(er.items(), key=lambda x: -x[1])
            row_parts = []
            for reason, cnt in items:
                pct = cnt / total_t * 100
                row_parts.append(f"{reason}:{cnt}({pct:.0f}%)")
            print(f"  Exits: {' │ '.join(row_parts)}")

    # ── Risk Warning ───────────────────────────────────────────
    print(f"""
{Y}{'═' * 68}
  ⚠  PERINGATAN RISIKO LEVERAGE 10x
  {'─' * 66}
  • Setiap kali harga bergerak 10% melawan posisi  → LIQUIDASI penuh
  • Dengan SL 20%/trade: 5 loss beruntun dalam 1 tahun bisa
    menghilangkan ~67% modal (compound 20% loss × 5)
  • BTC sangat volatile — satu crash 50% bisa memicu banyak SL
  • Untuk produksi, gunakan leverage ≤ 3x dan SL ≤ 5% per trade
  • Hasil backtest TIDAK menjamin return di masa depan
{'=' * 68}{R}""")


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════

def save_results(all_results):
    out_dir = PROJECT_ROOT / "backtests" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "leveraged_backtest.json"

    compact = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            compact[k] = {x: y for x, y in v.items() if x != "trades"}
        else:
            compact[k] = v

    with open(path, "w") as f:
        json.dump(compact, f, indent=2, default=str)
    return path


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()

    print(f"""
{C}╔{'═' * 68}╗
║{'BTC QUANT SYSTEM — LEVERAGED BACKTEST (10x)':^68s}║
║{'Modal: $1,000 │ Hard SL: 20%/trade │ 2019-2024':^68s}║
╚{'═' * 68}╝{R}""")

    # Load data
    print(f"\n  {D}Loading cached features...{R}", end=" ", flush=True)
    df = load_data()
    print(f"{G}✓{R} {len(df):,} candles ({df.index[0].date()} → {df.index[-1].date()})")

    # Build engines
    print(f"  {D}Building signal engines...{R}", end=" ", flush=True)
    base_signal   = SignalEngine()
    trend_engine  = TrendAwareSignalEngine(base_signal, bias_recalc_bars=24)
    signal_engine = HighSelectivityEngine(trend_engine, min_confidence=0.70)
    print(f"{G}✓{R} TrendAware + HighSelectivity(70%)")

    # Run
    print(f"\n  Running year-by-year backtest with $1,000 capital (kumulatif)...")
    print(f"  {'─' * 50}")
    all_results = run_leveraged_yearly(df, signal_engine)

    # Save
    path = save_results(all_results)

    # Print
    print_report(all_results)

    elapsed = time.perf_counter() - t0
    print(f"\n  {G}✓ Done in {elapsed:.1f}s │ Saved: {path.relative_to(PROJECT_ROOT)}{R}\n")


if __name__ == "__main__":
    main()
