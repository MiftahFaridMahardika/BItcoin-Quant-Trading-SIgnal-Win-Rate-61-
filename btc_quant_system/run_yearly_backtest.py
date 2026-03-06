"""
BTC Quant Trading System — Yearly Backtest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run backtesting year by year from 2022 to 2025.
Initial capital: $10,000.
Displays win rate and PNL per year.
"""

import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import json
import os

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Core Engines
from engines.data_pipeline import DataPipeline
from engines.feature_engine import FeatureEngine
from engines.ml_models import MLSignalModel
from engines.deep_learning import SequenceDataPreparer, DeepLearningEnsemble
from engines.signal_engine import SignalEngine
from engines.risk_engine import RiskEngine
from engines.execution_engine import BacktestEngine

# ══════════════════════════════════════════════════════════════
# SETUP LOGGING
# ══════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.WARNING) # Set to WARNING to reduce noise, we'll print custom summaries
logger = logging.getLogger("trading")
logger.setLevel(logging.INFO)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
DATA_PATH = PROJECT_ROOT.parent / "btcusd_1-min_data.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "trained"
DL_MODEL_DIR = PROJECT_ROOT / "models" / "deep_learning"
OPTIMIZED_PARAMS_PATH = PROJECT_ROOT / "configs" / "optimized_params.yaml"

CONFIG = {
    "initial_capital": 10000,
    "risk_per_trade": 0.02,
    "max_drawdown": 0.15,
    "leverage": 1,
    "slippage_pct": 0.0005,
    "taker_fee_pct": 0.0004,
    "warmup_periods": 200,
}

def run_yearly_backtest():
    # 1. LOAD & PREPARE DATA ONCE
    print("Loading data... (this may take a moment)")
    pipeline = DataPipeline()
    try:
        df_4h = pipeline.get_data(
            timeframe="4h", 
            use_cache=True # Use cache since we probably ran optimized_system recently
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Computing features...")
    fe = FeatureEngine()
    df_featured = fe.compute_all_features(df_4h)

    # Note: We assume the models are already trained from earlier!
    # If not, let's just allow SignalEngine to load whatever is available.

    print("Setting up engines...")
    signal_engine = SignalEngine({
        "rule_weight": 0.40,
        "ml_weight": 0.40,
        "dl_weight": 0.20,
        "load_models": True
    })

    print(f"\n{'═' * 50}")
    print(f" YEARLY BACKTEST SUMMARY (2022 - 2025)")
    print(f" Initial Capital: ${CONFIG['initial_capital']:,}")
    print(f"{'═' * 50}\n")

    combined_results = {}

    for year in range(2022, 2026): # 2022, 2023, 2024, 2025
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # Initialize Risk Engine & Backtest Engine per year (resets capital)
        risk_engine = RiskEngine(str(PROJECT_ROOT / "configs" / "risk_config.yaml"))
        risk_engine.initial_capital = CONFIG["initial_capital"]
        risk_engine.account_balance = CONFIG["initial_capital"]
        risk_engine.peak_balance = CONFIG["initial_capital"]
        backtest = BacktestEngine(CONFIG)

        # Ensure dataframe has data for this year
        year_mask = (df_featured.index >= pd.Timestamp(start_date, tz="UTC")) & (df_featured.index <= pd.Timestamp(end_date, tz="UTC"))
        if not year_mask.any():
            print(f"  [{year}] No data available for this year.")
            continue

        results = backtest.run_backtest(
            df_featured,
            signal_engine,
            risk_engine,
            start_date=start_date,
            end_date=end_date,
            show_progress=False # Hide per-candle logs
        )

        
        if "total_trades" in results and results["total_trades"] > 0:
            trades = results["total_trades"]
            win_rate = results["win_rate_pct"]
            pnl = results["total_pnl"]
            ret_pct = results["total_return_pct"]
            
            color = "\033[32m" if pnl >= 0 else "\033[31m"
            reset = "\033[0m"

            print(f"  {year} │ Trades: {trades:>4d} │ Win Rate: {win_rate:>6.2f}% │ PNL: {color}${pnl:>9,.2f}{reset} ({color}{ret_pct:>6.2f}%{reset})")
            combined_results[str(year)] = results
        else:
            print(f"  {year} │ No trades executed.")


    print(f"\n{'═' * 50}\n")
    
    # Save results
    os.makedirs("backtests/results", exist_ok=True)
    report_path = f"backtests/results/yearly_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # We can't dump the full BacktestEngine directly due to dataframe internals, 
    # but we can save the summary metrics
    summary_to_save = {}
    for y, res in combined_results.items():
        summary_to_save[y] = {
            "initial_capital": res.get("initial_capital"),
            "final_equity": res.get("final_equity"),
            "total_pnl": res.get("total_pnl"),
            "total_return_pct": res.get("total_return_pct"),
            "total_trades": res.get("total_trades"),
            "win_rate_pct": res.get("win_rate_pct")
        }

    with open(report_path, "w") as f:
        json.dump(summary_to_save, f, indent=4)

if __name__ == "__main__":
    run_yearly_backtest()
