#!/usr/bin/env python3
"""
Adaptive SL/TP Strategy — Old vs New Comparison
=================================================
Simulates trade outcomes for every TAKE signal in 2023-2024 using:

  OLD  — Fixed SL=1.5×ATR, TP3=5.0×ATR, trailing at 1.5×ATR (no partials)
  NEW  — Adaptive SL/TP + tiered trailing + partial exits (40% TP1 / 30% TP2 / 30% runner)

Prints exit reason distribution, win rate, and average PnL (in R) for both.
Target: reduce stop-loss exits from 89% → <60%.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from utils.logger import setup_all_loggers
from utils.helpers import format_number, format_duration
from engines.signal_engine import SignalEngine, SignalQualityEngine
from engines.risk_engine import RiskEngine


C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"

MAX_TRADE_BARS = 168    # ~28 days at 4H = max hold time


# ══════════════════════════════════════════════════════════════
# CORE SIMULATION FUNCTIONS
# ══════════════════════════════════════════════════════════════

def simulate_old(
    entry_price: float,
    direction: str,
    atr: float,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
) -> Dict:
    """
    OLD strategy: fixed SL=1.5×ATR, TP3=5.0×ATR.
    No partial exits — full position exits at SL or TP3.
    Trailing at flat 1.5×ATR from close each bar.
    """
    sign = 1 if direction == "LONG" else -1
    sl   = entry_price - sign * atr * 1.5
    tp3  = entry_price + sign * atr * 5.0

    sl_dist = abs(entry_price - sl)
    cur_sl  = sl

    for i, (h, l, c) in enumerate(zip(high_arr, low_arr, close_arr)):
        if direction == "LONG":
            if l <= cur_sl:
                pnl_r = (cur_sl - entry_price) / sl_dist
                return {'exit': 'STOP_LOSS', 'pnl_r': pnl_r, 'bars': i, 'tp1': False, 'tp2': False}
            if h >= tp3:
                pnl_r = (tp3 - entry_price) / sl_dist
                return {'exit': 'TP3', 'pnl_r': pnl_r, 'bars': i, 'tp1': False, 'tp2': False}
            # Flat trailing
            new_sl = c - 1.5 * atr
            if new_sl > cur_sl:
                cur_sl = new_sl
        else:
            if h >= cur_sl:
                pnl_r = (entry_price - cur_sl) / sl_dist
                return {'exit': 'STOP_LOSS', 'pnl_r': pnl_r, 'bars': i, 'tp1': False, 'tp2': False}
            if l <= tp3:
                pnl_r = (entry_price - tp3) / sl_dist
                return {'exit': 'TP3', 'pnl_r': pnl_r, 'bars': i, 'tp1': False, 'tp2': False}
            new_sl = c + 1.5 * atr
            if new_sl < cur_sl:
                cur_sl = new_sl

    # Time exit
    last_c = close_arr[-1]
    if direction == "LONG":
        pnl_r = (last_c - entry_price) / sl_dist
    else:
        pnl_r = (entry_price - last_c) / sl_dist
    return {'exit': 'TIME_EXIT', 'pnl_r': pnl_r, 'bars': len(high_arr), 'tp1': False, 'tp2': False}


def simulate_new(
    entry_price: float,
    direction: str,
    atr: float,
    regime: str,
    vol_pct: float,
    confidence: float,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    atr_arr: np.ndarray = None,   # per-bar live ATR (optional)
) -> Dict:
    """
    NEW strategy: adaptive SL/TP + tiered trailing + partial exits.

    Mirrors the fixed ExecutionEngine logic:
      - Trailing only after profit >= +0.5× ATR (Tier 0 = no trail)
      - Breakeven SL clamped at entry after TP1
      - TP2 trail clamped to breakeven (entry)

    Partial exit plan:
      TP1 (40% position) → move SL to breakeven (entry)
      TP2 (30% position) → trail at 1× ATR (≥ entry)
      TP3 (30% runner)   → full close or tight trail
    """
    mults = RiskEngine.get_adaptive_sl_tp_multipliers(regime, vol_pct, confidence)
    if mults is None:
        return {'exit': 'REGIME_SKIP', 'pnl_r': 0.0, 'bars': 0, 'tp1': False, 'tp2': False}

    sign   = 1 if direction == "LONG" else -1
    sl     = entry_price - sign * atr * mults['sl']
    tp1    = entry_price + sign * atr * mults['tp1']
    tp2    = entry_price + sign * atr * mults['tp2']
    tp3    = entry_price + sign * atr * mults['tp3']

    sl_dist  = abs(entry_price - sl)
    cur_sl   = sl
    tp1_hit  = False
    tp2_hit  = False
    breakeven = False
    total_r  = 0.0

    for i, (h, l, c) in enumerate(zip(high_arr, low_arr, close_arr)):
        # Use live ATR per bar if available, otherwise fall back to entry ATR
        bar_atr = float(atr_arr[i]) if (atr_arr is not None and i < len(atr_arr)
                                         and not np.isnan(atr_arr[i]) and atr_arr[i] > 0) else atr

        if direction == "LONG":
            # ── TP1 partial (40%) ──────────────────────────────────────
            if not tp1_hit and h >= tp1:
                total_r  += (tp1 - entry_price) / sl_dist * 0.40
                tp1_hit   = True
                # Breakeven: SL moves to entry, never below it
                cur_sl    = max(cur_sl, entry_price)
                breakeven = True
                continue   # re-evaluate same bar on next iteration

            # ── TP2 partial (30%) ──────────────────────────────────────
            if tp1_hit and not tp2_hit and h >= tp2:
                total_r  += (tp2 - entry_price) / sl_dist * 0.30
                tp2_hit   = True
                # Trail at 1× ATR, clamped to breakeven (entry)
                new_sl    = max(c - bar_atr, entry_price)
                if new_sl > cur_sl:
                    cur_sl = new_sl
                continue

            # ── TP3 runner (30%) ──────────────────────────────────────
            if tp2_hit and h >= tp3:
                total_r += (tp3 - entry_price) / sl_dist * 0.30
                return {'exit': 'TP3', 'pnl_r': total_r, 'bars': i, 'tp1': tp1_hit, 'tp2': tp2_hit}

            # ── SL / trail ─────────────────────────────────────────────
            if l <= cur_sl:
                remain = 1.0 - (0.40 if tp1_hit else 0) - (0.30 if tp2_hit else 0)
                total_r += (cur_sl - entry_price) / sl_dist * remain
                reason   = 'TRAIL_STOP' if breakeven else 'STOP_LOSS'
                return {'exit': reason, 'pnl_r': total_r, 'bars': i, 'tp1': tp1_hit, 'tp2': tp2_hit}

            # ── Tiered trailing (mirrors ExecutionEngine._update_trailing_stop) ──
            profit = c - entry_price           # absolute price profit
            if profit >= 3 * bar_atr:          # Tier 4 — tightest
                new_sl = c - 0.7 * bar_atr
            elif profit >= 2 * bar_atr:        # Tier 3
                new_sl = c - 1.0 * bar_atr
            elif profit >= bar_atr:            # Tier 2
                new_sl = max(entry_price, c - 1.5 * bar_atr)
                if not breakeven:
                    breakeven = True
            elif profit >= 0.5 * bar_atr:      # Tier 1 — breakeven only
                new_sl = entry_price
                if not breakeven:
                    breakeven = True
            else:
                continue   # Tier 0 — too early, keep SL fixed
            if new_sl > cur_sl:
                cur_sl = new_sl

        else:  # SHORT (symmetric)
            if not tp1_hit and l <= tp1:
                total_r  += (entry_price - tp1) / sl_dist * 0.40
                tp1_hit   = True
                cur_sl    = min(cur_sl, entry_price)   # breakeven, never above entry
                breakeven = True
                continue
            if tp1_hit and not tp2_hit and l <= tp2:
                total_r  += (entry_price - tp2) / sl_dist * 0.30
                tp2_hit   = True
                new_sl    = min(c + bar_atr, entry_price)   # clamped to entry
                if new_sl < cur_sl:
                    cur_sl = new_sl
                continue
            if tp2_hit and l <= tp3:
                total_r += (entry_price - tp3) / sl_dist * 0.30
                return {'exit': 'TP3', 'pnl_r': total_r, 'bars': i, 'tp1': tp1_hit, 'tp2': tp2_hit}
            if h >= cur_sl:
                remain  = 1.0 - (0.40 if tp1_hit else 0) - (0.30 if tp2_hit else 0)
                total_r += (entry_price - cur_sl) / sl_dist * remain
                reason   = 'TRAIL_STOP' if breakeven else 'STOP_LOSS'
                return {'exit': reason, 'pnl_r': total_r, 'bars': i, 'tp1': tp1_hit, 'tp2': tp2_hit}
            profit = entry_price - c
            if profit >= 3 * bar_atr:
                new_sl = c + 0.7 * bar_atr
            elif profit >= 2 * bar_atr:
                new_sl = c + 1.0 * bar_atr
            elif profit >= bar_atr:
                new_sl = min(entry_price, c + 1.5 * bar_atr)
                if not breakeven:
                    breakeven = True
            elif profit >= 0.5 * bar_atr:
                new_sl = entry_price
                if not breakeven:
                    breakeven = True
            else:
                continue   # Tier 0 — keep SL fixed
            if new_sl < cur_sl:
                cur_sl = new_sl

    # Time exit — close remaining at last bar
    remain  = 1.0 - (0.40 if tp1_hit else 0) - (0.30 if tp2_hit else 0)
    last_c  = close_arr[-1]
    if direction == "LONG":
        total_r += (last_c - entry_price) / sl_dist * remain
    else:
        total_r += (entry_price - last_c) / sl_dist * remain
    return {'exit': 'TIME_EXIT', 'pnl_r': total_r, 'bars': len(high_arr), 'tp1': tp1_hit, 'tp2': tp2_hit}


# ══════════════════════════════════════════════════════════════
# REPORT HELPERS
# ══════════════════════════════════════════════════════════════

def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:5.1f}%" if total else "  N/A"


def print_strategy_report(label: str, results: list, color: str) -> None:
    total = len(results)
    if total == 0:
        print(f"  No trades.")
        return

    by_exit: Dict[str, list] = {}
    for r in results:
        by_exit.setdefault(r['exit'], []).append(r['pnl_r'])

    wins  = [r for r in results if r['pnl_r'] > 0]
    losses = [r for r in results if r['pnl_r'] <= 0]
    win_rate = len(wins) / total * 100

    sl_count    = len(by_exit.get('STOP_LOSS', []))
    trail_count = len(by_exit.get('TRAIL_STOP', []))
    tp3_count   = len(by_exit.get('TP3', []))
    time_count  = len(by_exit.get('TIME_EXIT', []))
    skip_count  = len(by_exit.get('REGIME_SKIP', []))
    bad_exit_pct = (sl_count + trail_count) / max(total - skip_count, 1) * 100

    tp1_touched = sum(1 for r in results if r.get('tp1'))
    tp2_touched = sum(1 for r in results if r.get('tp2'))

    avg_win_r  = float(np.mean([r['pnl_r'] for r in wins]))  if wins  else 0.0
    avg_loss_r = float(np.mean([r['pnl_r'] for r in losses])) if losses else 0.0
    avg_r      = float(np.mean([r['pnl_r'] for r in results]))
    expectancy = (win_rate / 100) * avg_win_r + (1 - win_rate / 100) * avg_loss_r

    print(f"\n{color}{'─' * 55}{R}")
    print(f"{B}{color}  {label}{R}")
    print(f"{color}{'─' * 55}{R}")
    print(f"  Total trades       : {total:>6,}")
    print(f"  Win rate           : {color}{win_rate:>5.1f}%{R}")
    print(f"  Avg PnL (R)        : {G if avg_r >= 0 else RED}{avg_r:>+.3f}R{R}")
    print(f"  Expectancy         : {G if expectancy >= 0 else RED}{expectancy:>+.3f}R{R}")
    print(f"  Avg win (R)        : {G}{avg_win_r:>+.3f}R{R}")
    print(f"  Avg loss (R)       : {RED}{avg_loss_r:>+.3f}R{R}")
    print(f"\n  Exit Breakdown:")
    print(f"  {'─' * 45}")
    order = [
        ('STOP_LOSS',    RED,  "Original SL hit (bad)"),
        ('TRAIL_STOP',   Y,    "Trailed out (partial win possible)"),
        ('TP3',          G,    "Full TP3 runner"),
        ('TIME_EXIT',    D,    "Time exit (max hold)"),
        ('REGIME_SKIP',  D,    "Skipped — HIGH_VOL regime"),
    ]
    for key, col, desc in order:
        n = len(by_exit.get(key, []))
        if n == 0:
            continue
        avg = float(np.mean(by_exit[key]))
        print(f"  {col}{key:<14s}{R} │ {n:>5,} ({_pct(n, total)}) │ avg={avg:>+.2f}R │ {desc}")

    if tp1_touched > 0 or tp2_touched > 0:
        print(f"\n  Partial exits touched:")
        print(f"    TP1 (40%) hit : {tp1_touched:,} ({_pct(tp1_touched, total)})")
        print(f"    TP2 (30%) hit : {tp2_touched:,} ({_pct(tp2_touched, total)})")

    print(f"\n  {RED}SL + Trail exits : {sl_count + trail_count:,} ({bad_exit_pct:.1f}%){R}")


def print_comparison(old_results: list, new_results: list, period: str) -> None:
    total = len(old_results)
    if total == 0:
        return

    old_sl    = sum(1 for r in old_results if r['exit'] == 'STOP_LOSS')
    new_sl    = sum(1 for r in new_results if r['exit'] == 'STOP_LOSS')
    new_trail = sum(1 for r in new_results if r['exit'] == 'TRAIL_STOP')
    new_bad   = new_sl + new_trail

    old_wr = sum(1 for r in old_results if r['pnl_r'] > 0) / total * 100
    new_wr = sum(1 for r in new_results if r['pnl_r'] > 0) / total * 100

    old_avg = float(np.mean([r['pnl_r'] for r in old_results]))
    new_avg = float(np.mean([r['pnl_r'] for r in new_results]))

    print(f"\n{C}{'═' * 65}")
    print(f"  COMPARISON — {period}")
    print(f"{'═' * 65}{R}")
    print(f"\n  {'Metric':<32s} │ {'OLD':>10s} │ {'NEW':>10s} │ {'Delta':>8s}")
    print(f"  {'─' * 62}")

    rows = [
        ("Win rate (%)",        f"{old_wr:.1f}%",      f"{new_wr:.1f}%",
         f"{new_wr - old_wr:>+.1f}pp"),
        ("Avg PnL (R)",         f"{old_avg:>+.3f}R",   f"{new_avg:>+.3f}R",
         f"{new_avg - old_avg:>+.3f}R"),
        ("Original SL exits",   f"{old_sl:,} ({old_sl/total*100:.1f}%)",
         f"{new_sl:,} ({new_sl/total*100:.1f}%)",
         f"{new_sl - old_sl:>+,}"),
        ("SL + Trail exits",    f"{old_sl/total*100:.1f}%",
         f"{new_bad/total*100:.1f}%",
         f"{(new_bad - old_sl) / total * 100:>+.1f}pp"),
        ("TP3 exits",
         f"{sum(1 for r in old_results if r['exit']=='TP3'):,}",
         f"{sum(1 for r in new_results if r['exit']=='TP3'):,}", ""),
    ]
    for metric, old_v, new_v, delta in rows:
        delta_col = G if delta.startswith('+') and 'exit' not in metric.lower() else (
                    G if delta.startswith('-') and 'exit' in metric.lower() else R)
        print(f"  {metric:<32s} │ {old_v:>10s} │ {new_v:>10s} │ {delta_col}{delta:>8s}{R}")

    print(f"\n  SL exit reduction  : {RED}{old_sl/total*100:.1f}%{R} → "
          f"{G}{new_sl/total*100:.1f}%{R}  "
          f"(target: <60% of all exits)")
    print(f"  Win rate change    : {RED}{old_wr:.1f}%{R} → {G}{new_wr:.1f}%{R}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"""
{C}{'═' * 65}
  ADAPTIVE SL/TP — OLD vs NEW SIMULATION (2023-2024)
{'═' * 65}{R}
""")
    setup_all_loggers(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")

    # ── Load data ──────────────────────────────────────────────────────
    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if not feat_path.exists():
        print(f"{RED}Feature cache not found: {feat_path}{R}")
        return

    print(f"{B}[1/4]{R} Loading data and generating quality signals...")
    t0 = time.perf_counter()
    df = pd.read_parquet(feat_path)

    signal_engine  = SignalEngine()
    quality_engine = SignalQualityEngine()
    df = signal_engine.generate_signals_batch(df, start_idx=200)
    df = quality_engine.generate_quality_signals_batch(df, default_ml_confidence=0.70)
    print(f"  ✓ {len(df):,} bars loaded and signals computed ({format_duration(time.perf_counter()-t0)})")

    # ── Select TAKE signals 2023-2024 ──────────────────────────────────
    take_df = df.loc["2023-01-01":"2024-12-31"]
    take_df = take_df[take_df["quality_decision"] == "TAKE"].copy()
    print(f"  ✓ {len(take_df):,} TAKE signals selected (2023-2024)")

    if len(take_df) == 0:
        print(f"{RED}No TAKE signals found — check quality engine output.{R}")
        return

    # ── Simulate each signal ───────────────────────────────────────────
    print(f"\n{B}[2/4]{R} Simulating trades...")
    t1 = time.perf_counter()

    close_all = df["Close"].values
    high_all  = df["High"].values
    low_all   = df["Low"].values
    atr_all   = df.get("atr_14", df["Close"] * 0.02).fillna(df["Close"] * 0.02).values
    n_total   = len(df)

    old_results = []
    new_results = []

    for ts, row in take_df.iterrows():
        idx       = df.index.get_loc(ts)
        direction = str(row["signal_type"]).replace("STRONG_", "")   # LONG or SHORT
        if direction not in ("LONG", "SHORT"):
            continue

        entry_price = float(row["Close"])
        atr         = float(atr_all[idx])
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.02

        # Forward data (next MAX_TRADE_BARS bars)
        end_idx   = min(idx + MAX_TRADE_BARS + 1, n_total)
        fwd_high  = high_all[idx + 1: end_idx]
        fwd_low   = low_all[idx + 1: end_idx]
        fwd_close = close_all[idx + 1: end_idx]
        fwd_atr   = atr_all[idx + 1: end_idx]     # live ATR per bar

        if len(fwd_high) == 0:
            continue

        # Old simulation
        old_r = simulate_old(entry_price, direction, atr, fwd_high, fwd_low, fwd_close)
        old_results.append(old_r)

        # New simulation
        regime    = str(row.get("signal_regime", "NORMAL"))
        vol_reg   = int(row.get("vol_regime", 1) or 1)
        vol_pct   = {0: 15.0, 1: 50.0, 2: 75.0, 3: 95.0}.get(vol_reg, 50.0)
        confidence = float(row.get("signal_confidence", 0.70))

        new_r = simulate_new(
            entry_price, direction, atr, regime, vol_pct, confidence,
            fwd_high, fwd_low, fwd_close,
            atr_arr=fwd_atr,             # pass live ATR
        )
        new_results.append(new_r)

    sim_time = time.perf_counter() - t1
    print(f"  ✓ {len(old_results):,} trades simulated in {format_duration(sim_time)}")

    # ── Reports ────────────────────────────────────────────────────────
    print(f"\n{B}[3/4]{R} Strategy reports...")

    print_strategy_report("OLD Strategy (fixed 1.5/5.0 ATR, no partials)", old_results, RED)
    print_strategy_report("NEW Strategy (adaptive + tiered trail + 40/30/30 partials)", new_results, G)

    # ── Comparison ─────────────────────────────────────────────────────
    print(f"\n{B}[4/4]{R} Comparison table...")
    print_comparison(old_results, new_results, "2023-2024 (TAKE signals)")

    # ── Per-year drill-down ─────────────────────────────────────────────
    old_2023 = [old_results[i] for i, ts in enumerate(take_df.index) if str(ts)[:4] == "2023"]
    new_2023 = [new_results[i] for i, ts in enumerate(take_df.index) if str(ts)[:4] == "2023"]
    old_2024 = [old_results[i] for i, ts in enumerate(take_df.index) if str(ts)[:4] == "2024"]
    new_2024 = [new_results[i] for i, ts in enumerate(take_df.index) if str(ts)[:4] == "2024"]

    if old_2023:
        print_comparison(old_2023, new_2023, "2023 only")
    if old_2024:
        print_comparison(old_2024, new_2024, "2024 only")

    # ── Adaptive multiplier examples ───────────────────────────────────
    print(f"\n{C}{'═' * 65}")
    print(f"  ADAPTIVE MULTIPLIER EXAMPLES (vs fixed defaults)")
    print(f"{'═' * 65}{R}")
    print(f"\n  {'Regime':<10s} │ {'Vol%':>6s} │ {'Trend':>6s} │  SL   │  TP1  │  TP2  │  TP3")
    print(f"  {'─' * 60}")
    combos = [
        ("BULL",     50,  0.5), ("BULL",     80,  0.8), ("BULL",     20,  0.5),
        ("BEAR",     50,  0.5), ("SIDEWAYS", 50,  0.5), ("HIGH_VOL", 50,  0.5),
        ("NORMAL",   50,  0.5), ("HIGH",     75,  0.7),
    ]
    print(f"  {D}{'FIXED':<10s} │ {'  —':>6s} │ {'  —':>6s} │ 1.50  │ 2.00  │ 3.50  │ 5.00{R}")
    for regime, vp, ts in combos:
        m = RiskEngine.get_adaptive_sl_tp_multipliers(regime, vp, ts)
        if m is None:
            print(f"  {RED}{regime:<10s}{R} │ {vp:>6.0f} │ {ts:>6.2f} │ SKIP (HIGH_VOL — no trade)")
        else:
            print(f"  {regime:<10s} │ {vp:>6.0f} │ {ts:>6.2f} │ {m['sl']:>5.2f} │ {m['tp1']:>5.2f} │ {m['tp2']:>5.2f} │ {m['tp3']:>5.2f}")

    # ── Final summary ──────────────────────────────────────────────────
    n    = len(old_results)
    old_sl_pct = sum(1 for r in old_results if r['exit'] == 'STOP_LOSS') / max(n, 1) * 100
    new_sl_pct = sum(1 for r in new_results if r['exit'] == 'STOP_LOSS') / max(n, 1) * 100
    new_bad_pct = (
        sum(1 for r in new_results if r['exit'] in ('STOP_LOSS', 'TRAIL_STOP'))
        / max(n, 1) * 100
    )
    old_wr = sum(1 for r in old_results if r['pnl_r'] > 0) / max(n, 1) * 100
    new_wr = sum(1 for r in new_results if r['pnl_r'] > 0) / max(n, 1) * 100
    old_exp = float(np.mean([r['pnl_r'] for r in old_results])) if old_results else 0
    new_exp = float(np.mean([r['pnl_r'] for r in new_results])) if new_results else 0

    target_met_sl  = "✓" if new_sl_pct < 60 else "✗"
    target_met_wr  = "✓" if new_wr > 50   else "✗"

    print(f"""
{C}{'═' * 65}
  FINAL SUMMARY
{'═' * 65}{R}
  Signals simulated : {n:,} TAKE signals (2023-2024)
  Max hold          : {MAX_TRADE_BARS} bars (~28 days at 4H)

  ┌─────────────────────────────────────────────────────┐
  │ Metric             │    OLD     │    NEW     │ Target │
  ├─────────────────────────────────────────────────────┤
  │ Win rate           │ {old_wr:>6.1f}%   │ {new_wr:>6.1f}%   │  >50%  │
  │ Avg PnL (R)        │ {old_exp:>+6.3f}R  │ {new_exp:>+6.3f}R  │  >0    │
  │ Original SL exits  │ {old_sl_pct:>6.1f}%   │ {new_sl_pct:>6.1f}%   │  <60%  │
  │ SL + Trail exits   │ {old_sl_pct:>6.1f}%   │ {new_bad_pct:>6.1f}%   │  <60%  │
  └─────────────────────────────────────────────────────┘

  Targets:
    SL exits < 60%  : {G if target_met_sl == '✓' else RED}{target_met_sl} {new_sl_pct:.1f}% (was {old_sl_pct:.1f}%){R}
    Win rate > 50%  : {G if target_met_wr == '✓' else RED}{target_met_wr} {new_wr:.1f}% (was {old_wr:.1f}%){R}

  Total time : {format_duration(time.perf_counter() - t0)}
{G}{'═' * 65}{R}
""")


if __name__ == "__main__":
    main()
