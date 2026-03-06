"""
Diagnostic Script: Why are there so few trades?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Checks signal counts at each stage:
1. Raw technical score distribution
2. Effect of ML/DL ensemble (if any)
3. Effect of Confidence threshold (0.6)
4. Effect of R:R threshold (1.5)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from engines.data_pipeline import DataPipeline
from engines.feature_engine import FeatureEngine
from engines.signal_engine import SignalEngine

def diagnose():
    print("Loading data and computing features...")
    pipeline = DataPipeline()
    df_4h = pipeline.get_data(timeframe="4h", use_cache=True)
    
    fe = FeatureEngine()
    df_featured = fe.compute_all_features(df_4h)
    
    print("Running Signal Engine (Batch)...")
    se = SignalEngine({"load_models": True})
    df_signals = se.generate_signals_batch(df_featured)
    
    total_bars = len(df_signals)
    non_skip_bars = (df_signals["signal_type"] != "SKIP").sum()
    
    print(f"\nTotal bars (4H): {total_bars:,}")
    print(f"Actionable Signals (Score >= 4 or <= -4): {non_skip_bars:,} ({non_skip_bars/total_bars:.2%})")
    
    # Check Confidence Filter (0.6)
    high_conf = df_signals[(df_signals["signal_type"] != "SKIP") & (df_signals["signal_confidence"] >= 0.6)]
    low_conf = df_signals[(df_signals["signal_type"] != "SKIP") & (df_signals["signal_confidence"] < 0.6)]
    
    print(f"Signals passing Confidence >= 0.6: {len(high_conf):,} ({len(high_conf)/max(1, non_skip_bars):.2%} of actionable)")
    print(f"Signals killed by Confidence < 0.6: {len(low_conf):,}")

    # Check R:R Filter (1.5)
    # Note: R:R is calculated in generate_trading_signal which is bar-by-bar
    # We'll sample 100 actionable signals to check R:R distribution
    actionable_indices = df_signals[df_signals["signal_type"] != "SKIP"].index
    if len(actionable_indices) > 0:
        rr_list = []
        for idx_val in actionable_indices[:500]: # Sample 500
            pos = df_signals.index.get_loc(idx_val)
            sig = se.generate_trading_signal(df_signals, pos)
            rr_list.append(sig.get('risk_reward', 0))
        
        rr_arr = np.array(rr_list)
        pass_rr = (rr_arr >= 1.5).sum()
        print(f"Sampled R:R >= 1.5: {pass_rr/len(rr_arr):.2%} pass rate")
        print(f"Average sampled R:R: {rr_arr.mean():.2f}")

if __name__ == "__main__":
    diagnose()
