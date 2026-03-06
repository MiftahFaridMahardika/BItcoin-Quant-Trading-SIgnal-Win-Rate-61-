#!/usr/bin/env python3
"""
Entry Filters: Impact Test
============================
Compares entry timing with and without the 4 entry filters:
  1. Time filter (low-liquidity hour blocking)
  2. Candle pattern filter (body/wick structure)
  3. Pullback entry (RSI-5 wait zone)
  4. S/R clearance (0.3× ATR from nearest swing level)

Shows:
  - How many signals each filter blocks
  - SL hit rate: filtered vs unfiltered entries
  - Win rate improvement
  - Per-bar entry distribution
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from utils.logger import setup_all_loggers
from utils.helpers import format_duration
from engines.signal_engine import SignalEngine
from engines.entry_filters import (
    EntryFilterManager,
    PullbackEntry,
    CandlePatternFilter,
    SRClearanceFilter,
    TimeFilter,
)

# ── Colours ──────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"

MAX_BARS = 168   # ~28 days at 4H


# ──────────────────────────────────────────────────────────────
# SIMULATE ONE TRADE
# ──────────────────────────────────────────────────────────────

def simulate_trade(
    entry_price: float,
    direction:   str,
    atr:         float,
    high_arr:    np.ndarray,
    low_arr:     np.ndarray,
    close_arr:   np.ndarray,
    atr_arr:     np.ndarray,
) -> dict:
    """Simulate with tiered trailing stop (mirrors execution_engine logic)."""
    sign    = 1 if direction == "LONG" else -1
    sl_dist = atr * 1.5
    sl      = entry_price - sign * sl_dist
    tp1     = entry_price + sign * atr * 1.65
    tp2     = entry_price + sign * atr * 2.75
    tp3     = entry_price + sign * atr * 5.0

    cur_sl    = sl
    tp1_hit   = False
    tp2_hit   = False
    breakeven = False
    total_r   = 0.0

    for i in range(len(high_arr)):
        h   = high_arr[i]
        l   = low_arr[i]
        c   = close_arr[i]
        bar_atr = (float(atr_arr[i])
                   if i < len(atr_arr) and not np.isnan(atr_arr[i]) and atr_arr[i] > 0
                   else atr)

        if direction == "LONG":
            if not tp1_hit and h >= tp1:
                total_r += (tp1 - entry_price) / sl_dist * 0.40
                tp1_hit  = True
                cur_sl   = max(cur_sl, entry_price)
                breakeven = True
                continue
            if tp1_hit and not tp2_hit and h >= tp2:
                total_r += (tp2 - entry_price) / sl_dist * 0.30
                tp2_hit  = True
                new_sl   = max(c - bar_atr, entry_price)
                if new_sl > cur_sl: cur_sl = new_sl
                continue
            if tp2_hit and h >= tp3:
                total_r += (tp3 - entry_price) / sl_dist * 0.30
                return {"exit": "TP3", "pnl_r": total_r, "bars": i,
                        "tp1": tp1_hit, "tp2": tp2_hit}
            if l <= cur_sl:
                remain   = 1.0 - (0.40 if tp1_hit else 0) - (0.30 if tp2_hit else 0)
                total_r += (cur_sl - entry_price) / sl_dist * remain
                return {"exit": "TRAIL_STOP" if breakeven else "STOP_LOSS",
                        "pnl_r": total_r, "bars": i, "tp1": tp1_hit, "tp2": tp2_hit}
            profit = c - entry_price
            if profit >= 3 * bar_atr:   new_sl = c - 0.7 * bar_atr
            elif profit >= 2 * bar_atr: new_sl = c - 1.0 * bar_atr
            elif profit >= bar_atr:
                new_sl = max(entry_price, c - 1.5 * bar_atr)
                if not breakeven: breakeven = True
            elif profit >= 0.5 * bar_atr:
                new_sl = entry_price
                if not breakeven: breakeven = True
            else: continue
            if new_sl > cur_sl: cur_sl = new_sl
        else:
            if not tp1_hit and l <= tp1:
                total_r += (entry_price - tp1) / sl_dist * 0.40
                tp1_hit  = True
                cur_sl   = min(cur_sl, entry_price)
                breakeven = True
                continue
            if tp1_hit and not tp2_hit and l <= tp2:
                total_r += (entry_price - tp2) / sl_dist * 0.30
                tp2_hit  = True
                new_sl   = min(c + bar_atr, entry_price)
                if new_sl < cur_sl: cur_sl = new_sl
                continue
            if tp2_hit and l <= tp3:
                total_r += (entry_price - tp3) / sl_dist * 0.30
                return {"exit": "TP3", "pnl_r": total_r, "bars": i,
                        "tp1": tp1_hit, "tp2": tp2_hit}
            if h >= cur_sl:
                remain   = 1.0 - (0.40 if tp1_hit else 0) - (0.30 if tp2_hit else 0)
                total_r += (entry_price - cur_sl) / sl_dist * remain
                return {"exit": "TRAIL_STOP" if breakeven else "STOP_LOSS",
                        "pnl_r": total_r, "bars": i, "tp1": tp1_hit, "tp2": tp2_hit}
            profit = entry_price - c
            if profit >= 3 * bar_atr:   new_sl = c + 0.7 * bar_atr
            elif profit >= 2 * bar_atr: new_sl = c + 1.0 * bar_atr
            elif profit >= bar_atr:
                new_sl = min(entry_price, c + 1.5 * bar_atr)
                if not breakeven: breakeven = True
            elif profit >= 0.5 * bar_atr:
                new_sl = entry_price
                if not breakeven: breakeven = True
            else: continue
            if new_sl < cur_sl: cur_sl = new_sl

    remain = 1.0 - (0.40 if tp1_hit else 0) - (0.30 if tp2_hit else 0)
    pnl_r  = (close_arr[-1] - entry_price) / sl_dist * remain * sign
    return {"exit": "TIME_EXIT", "pnl_r": pnl_r, "bars": len(high_arr),
            "tp1": tp1_hit, "tp2": tp2_hit}


# ──────────────────────────────────────────────────────────────
# PRINT HELPERS
# ──────────────────────────────────────────────────────────────

def print_result_block(label: str, results: list, color: str) -> None:
    valid = [r for r in results if r.get("active", True)]
    if not valid:
        print(f"  {label}: No valid trades")
        return
    wins   = [r for r in valid if r["pnl_r"] > 0]
    sl_ex  = [r for r in valid if r["exit"] == "STOP_LOSS"]
    trail  = [r for r in valid if r["exit"] == "TRAIL_STOP"]
    tp3    = [r for r in valid if r["exit"] == "TP3"]
    wr     = len(wins) / len(valid) * 100
    sl_pct = (len(sl_ex) + len(trail)) / len(valid) * 100
    avg_r  = float(np.mean([r["pnl_r"] for r in valid]))

    print(f"\n{color}{'─' * 58}{R}")
    print(f"{B}{color}  {label}{R}")
    print(f"{color}{'─' * 58}{R}")
    print(f"  Trades          : {len(valid):>5,}")
    print(f"  Win rate        : {color}{wr:>5.1f}%{R}")
    print(f"  Avg PnL (R)     : {(G if avg_r >= 0 else RED)}{avg_r:>+.3f}R{R}")
    print(f"  SL+Trail exits  : {(RED if sl_pct > 60 else G)}{sl_pct:>5.1f}%{R}")
    print(f"  TP3 exits       : {tp3.__len__():>4}")


def print_filter_block_table(filter_stats: dict, total_signals: int) -> None:
    print(f"\n{C}{'═'*60}")
    print(f"  FILTER BLOCK BREAKDOWN")
    print(f"{'═'*60}{R}")
    print(f"  {'Filter':<22} │ {'Blocked':>8} │ {'Block%':>8}")
    print(f"  {'─'*50}")
    fmap = [
        ("Time (low-liquidity)",  "blocked_time"),
        ("Candle pattern",        "blocked_candle"),
        ("Pullback wait",         "blocked_pullback"),
        ("S/R clearance",         "blocked_sr"),
    ]
    for label, key in fmap:
        n   = filter_stats.get(key, 0)
        pct = n / max(total_signals, 1) * 100
        col = Y if n > 0 else D
        print(f"  {label:<22} │ {col}{n:>8,}{R} │ {col}{pct:>7.1f}%{R}")
    passed = filter_stats.get("passed", 0)
    total  = filter_stats.get("checked", 0)
    print(f"  {'─'*50}")
    print(f"  {'PASSED':<22} │ {G}{passed:>8,}{R} │ {G}{passed/max(total,1)*100:>7.1f}%{R}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print(f"""
{C}{'═'*60}
  ENTRY FILTERS: IMPACT TEST  (2023-2024)
{'═'*60}{R}
""")
    setup_all_loggers(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")

    # ── Load data ─────────────────────────────────────────────────
    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if not feat_path.exists():
        print(f"{RED}Feature cache not found: {feat_path}{R}")
        return

    print(f"{B}[1/4]{R} Loading data...")
    t0 = time.perf_counter()
    df = pd.read_parquet(feat_path)
    signal_engine = SignalEngine()
    df = signal_engine.generate_signals_batch(df, start_idx=200)

    df_range = df.loc["2023-01-01":"2024-12-31"]
    print(f"  ✓ {len(df_range):,} bars, signals computed")

    # ── Get TAKE signals ───────────────────────────────────────────
    take_df = df_range[df_range["signal_type"].isin(
        ["STRONG_LONG", "LONG", "SHORT", "STRONG_SHORT"]
    )].copy()
    print(f"  ✓ {len(take_df):,} actionable signals in 2023-2024")

    # ── Build forward arrays ───────────────────────────────────────
    close_arr = df["Close"].values
    high_arr  = df["High"].values
    low_arr   = df["Low"].values
    atr_arr   = df.get("atr_14", df["Close"] * 0.02).fillna(df["Close"] * 0.02).values
    n_total   = len(df)

    # ── Instantiate filter manager ─────────────────────────────────
    efm = EntryFilterManager()

    # ── Run simulation ─────────────────────────────────────────────
    print(f"\n{B}[2/4]{R} Simulating OLD (no filters) vs NEW (with filters)...")
    t1 = time.perf_counter()

    old_results = []
    new_results = []
    per_filter_blocks = {
        "time": 0, "candle": 0, "pullback": 0, "sr": 0, "passed": 0
    }

    # Individual filter instances for granular tracking
    time_f    = TimeFilter()
    candle_f  = CandlePatternFilter()
    pb_f      = PullbackEntry()
    sr_f      = SRClearanceFilter()

    for ts, row in take_df.iterrows():
        idx = df.index.get_loc(ts)
        if idx < 200:
            continue

        direction   = str(row["signal_type"]).replace("STRONG_", "")
        if direction not in ("LONG", "SHORT"):
            continue

        entry_price = float(row["Close"])
        atr         = float(atr_arr[idx])
        if np.isnan(atr) or atr <= 0:
            atr = entry_price * 0.02

        end_idx   = min(idx + MAX_BARS + 1, n_total)
        fwd_high  = high_arr[idx + 1: end_idx]
        fwd_low   = low_arr[idx + 1: end_idx]
        fwd_close = close_arr[idx + 1: end_idx]
        fwd_atr   = atr_arr[idx + 1: end_idx]

        if len(fwd_high) == 0:
            continue

        # ── OLD: always enter, no filter ───
        old_r = simulate_trade(entry_price, direction, atr,
                               fwd_high, fwd_low, fwd_close, fwd_atr)
        old_results.append({**old_r, "direction": direction})

        # ── NEW: check filters one by one (individually tracked) ───
        time_result   = time_f.check_time_filter(ts)
        if not time_result.allowed:
            per_filter_blocks["time"] += 1
            new_results.append({"exit": "FILTERED_TIME", "pnl_r": 0.0,
                                 "direction": direction, "active": False})
            continue

        candle_result = candle_f.check_entry_candle(row, direction)
        if not candle_result.allowed:
            per_filter_blocks["candle"] += 1
            new_results.append({"exit": "FILTERED_CANDLE", "pnl_r": 0.0,
                                 "direction": direction, "active": False})
            continue

        pb_state = pb_f.should_wait_for_pullback(df.iloc[:idx + 1], direction, atr)
        if pb_state["wait"]:
            per_filter_blocks["pullback"] += 1
            # Simulate that we skip (conservative: treat as no trade)
            new_results.append({"exit": "FILTERED_PULLBACK", "pnl_r": 0.0,
                                 "direction": direction, "active": False})
            continue

        sr_result = sr_f.check_sr_clearance(df, idx, entry_price, direction, atr)
        if not sr_result.allowed:
            per_filter_blocks["sr"] += 1
            new_results.append({"exit": "FILTERED_SR", "pnl_r": 0.0,
                                 "direction": direction, "active": False})
            continue

        # Passed all filters → simulate entry
        per_filter_blocks["passed"] += 1
        new_r = simulate_trade(entry_price, direction, atr,
                               fwd_high, fwd_low, fwd_close, fwd_atr)
        new_results.append({**new_r, "direction": direction, "active": True})

    sim_time = time.perf_counter() - t1
    total_signals = len(old_results)
    print(f"  ✓ {total_signals:,} signals processed in {format_duration(sim_time)}")

    # ── Print filter breakdown ─────────────────────────────────────
    print(f"\n{B}[3/4]{R} Filter breakdown...")
    fs = {
        "blocked_time":     per_filter_blocks["time"],
        "blocked_candle":   per_filter_blocks["candle"],
        "blocked_pullback": per_filter_blocks["pullback"],
        "blocked_sr":       per_filter_blocks["sr"],
        "passed":           per_filter_blocks["passed"],
        "checked":          total_signals,
    }
    print_filter_block_table(fs, total_signals)

    # ── Strategy comparison ────────────────────────────────────────
    print(f"\n{B}[4/4]{R} Strategy comparison...")
    new_active = [r for r in new_results if r.get("active", False)]

    print_result_block("OLD — No Entry Filters (all signals)", old_results, RED)
    print_result_block("NEW — With Entry Filters (precision only)", new_active, G)

    # ── Final summary table ────────────────────────────────────────
    old_valid  = old_results
    new_valid  = new_active

    old_wr  = sum(1 for r in old_valid if r["pnl_r"] > 0) / max(len(old_valid), 1) * 100
    new_wr  = sum(1 for r in new_valid if r["pnl_r"] > 0) / max(len(new_valid), 1) * 100
    old_avg = float(np.mean([r["pnl_r"] for r in old_valid])) if old_valid else 0
    new_avg = float(np.mean([r["pnl_r"] for r in new_valid])) if new_valid else 0
    old_sl  = sum(1 for r in old_valid if r["exit"] == "STOP_LOSS") / max(len(old_valid), 1) * 100
    new_sl  = sum(1 for r in new_valid if r["exit"] == "STOP_LOSS") / max(len(new_valid), 1) * 100
    filtered_pct = (total_signals - len(new_valid)) / max(total_signals, 1) * 100

    print(f"""
{C}{'═'*60}
  FINAL SUMMARY — 2023-2024
{'═'*60}{R}

  {'Metric':<30} │ {'OLD':>10} │ {'NEW':>10} │ {'Delta':>8}
  {'─'*57}
  {'Signals taken':<30} │ {len(old_valid):>10,} │ {len(new_valid):>10,} │ {len(new_valid)-len(old_valid):>+8}
  {'Filtered out (%)':<30} │ {'—':>10} │ {filtered_pct:>9.1f}% │
  {'Win rate (%)':<30} │ {old_wr:>9.1f}% │ {new_wr:>9.1f}% │ {new_wr-old_wr:>+7.1f}pp
  {'Avg PnL (R)':<30} │ {old_avg:>+9.3f}R │ {new_avg:>+9.3f}R │ {new_avg-old_avg:>+7.3f}R
  {'SL exits (%)':<30} │ {old_sl:>9.1f}% │ {new_sl:>9.1f}% │ {new_sl-old_sl:>+7.1f}pp

  Target: 20%+ better entries
  Win rate improvement : {G if new_wr > old_wr else RED}{new_wr - old_wr:>+.1f}pp{R}
  SL exit reduction    : {G if new_sl < old_sl else RED}{new_sl - old_sl:>+.1f}pp{R}

  Total time : {format_duration(time.perf_counter() - t0)}
{G}{'═'*60}{R}
""")


if __name__ == "__main__":
    main()
