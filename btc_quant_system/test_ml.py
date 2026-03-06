#!/usr/bin/env python3
"""
ML Signal Model — End-to-End Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Load featured 4H data
2. Prepare data with chronological splits
     Train: 2015-01-01 → 2020-12-31
     Val:   2021-01-01 → 2022-12-31
     Test:  2023-01-01 → 2024-12-31
3. Train all 4 models
4. Evaluate ensemble on validation + test
5. Print feature importance
6. Save trained models
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from utils.logger import setup_all_loggers
from utils.helpers import format_number, format_duration
from engines.ml_models import MLSignalModel

# ── ANSI ─────────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"


def main():
    print(f"""
{C}{'═' * 72}
  ML SIGNAL MODEL — END-TO-END TEST
{'═' * 72}{R}
""")

    loggers = setup_all_loggers(
        log_dir=str(PROJECT_ROOT / "logs"),
        level="INFO",
    )

    total_t0 = time.perf_counter()

    # ══════════════════════════════════════════════════════════
    # STEP 1: Load featured data
    # ══════════════════════════════════════════════════════════
    print(f"{B}{C}[STEP 1/7]{R} Loading featured 4H data...")
    print(f"{'─' * 72}")

    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if not feat_path.exists():
        print(f"{RED}Feature cache not found. Run test_features.py first.{R}")
        return

    df = pd.read_parquet(feat_path)
    print(f"  ✓ Loaded {format_number(len(df), 0)} candles, {len(df.columns)} columns")
    print(f"  ✓ Range: {df.index[0]} → {df.index[-1]}")

    # ══════════════════════════════════════════════════════════
    # STEP 2: Prepare data
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 2/7]{R} Preparing train/val/test splits...")
    print(f"{'─' * 72}")

    model = MLSignalModel({
        "forward_period": 6,     # 6 × 4H = 24 hours ahead
        "buy_threshold": 0.015,  # 1.5% up = BUY
        "sell_threshold": -0.015,# 1.5% down = SELL
    })

    train_end = "2020-12-31"
    val_end   = "2022-12-31"

    data = model.prepare_data(df, train_end, val_end)

    print(f"\n  Split summary:")
    print(f"  {'─' * 50}")
    print(f"  Train : {format_number(len(data['y_train']), 0)} samples "
          f"({data['train_dates'][0].date()} → {data['train_dates'][-1].date()})")
    print(f"  Val   : {format_number(len(data['y_val']), 0)} samples "
          f"({data['val_dates'][0].date()} → {data['val_dates'][-1].date()})")
    print(f"  Test  : {format_number(len(data['y_test']), 0)} samples "
          f"({data['test_dates'][0].date()} → {data['test_dates'][-1].date()})")
    print(f"  Features: {len(data['feature_names'])}")

    # Label distribution per split
    print(f"\n  Label distribution:")
    print(f"  {'─' * 50}")
    for label_name, split_key in [("Train", "y_train"), ("Val", "y_val"), ("Test", "y_test")]:
        arr = data[split_key]
        n = len(arr)
        buy  = (arr == 2).sum()
        hold = (arr == 1).sum()
        sell = (arr == 0).sum()
        print(
            f"  {label_name:<6s}: BUY={buy:>5,}({buy/n:5.1%})  "
            f"HOLD={hold:>5,}({hold/n:5.1%})  "
            f"SELL={sell:>5,}({sell/n:5.1%})"
        )

    # ══════════════════════════════════════════════════════════
    # STEP 3: Train all models
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 3/7]{R} Training 4 models...")
    print(f"{'─' * 72}")

    t0 = time.perf_counter()
    results = model.train_all(data)
    train_time = time.perf_counter() - t0

    print(f"  ✓ All models trained in {format_duration(train_time)}")

    # ══════════════════════════════════════════════════════════
    # STEP 4: Training results table
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 4/7]{R} Training results...")
    print(f"{'─' * 72}")

    model.print_training_report(results)

    # ══════════════════════════════════════════════════════════
    # STEP 5: Evaluate ensemble on val + test
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 5/7]{R} Evaluating ensemble...")
    print(f"{'─' * 72}")

    val_metrics = model.evaluate(data["X_val"], data["y_val"], label="Validation")
    model.print_evaluation_report(val_metrics, label="VALIDATION SET")

    test_metrics = model.evaluate(data["X_test"], data["y_test"], label="Test")
    model.print_evaluation_report(test_metrics, label="TEST SET (2023-2024)")

    # ══════════════════════════════════════════════════════════
    # STEP 6: Feature importance
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 6/7]{R} Feature importance...")
    print(f"{'─' * 72}")

    model.print_feature_importance(top_n=20)

    # ══════════════════════════════════════════════════════════
    # STEP 7: Save models
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 7/7]{R} Saving trained models...")
    print(f"{'─' * 72}")

    save_dir = PROJECT_ROOT / "models" / "trained"
    model.save(str(save_dir))

    # Verify load
    model2 = MLSignalModel()
    model2.load(str(save_dir))
    reload_ok = (
        len(model2.models) == len(model.models)
        and model2.is_trained
        and model2.scaler is not None
    )
    print(f"  ✓ Saved to {save_dir}")
    print(f"  ✓ Reload verified: {'MATCH' if reload_ok else 'MISMATCH!'}")

    # Quick prediction check
    result = model2.ensemble_predict(data["X_test"][:5])
    print(f"  ✓ Reload prediction check: {result['prediction']}")

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    total_time = time.perf_counter() - total_t0

    # Best individual model
    best_name = max(results, key=lambda n: results[n]["val_f1"])
    best_f1 = results[best_name]["val_f1"]

    print(f"""
{C}{'═' * 72}
  FINAL SUMMARY
{'═' * 72}{R}
  Features used       : {len(data['feature_names'])}
  Train / Val / Test  : {len(data['y_train']):,} / {len(data['y_val']):,} / {len(data['y_test']):,}
  Forward horizon     : {model.forward_period} bars (24h @ 4H)
  Label thresholds    : BUY>{model.buy_threshold:.1%}  SELL<{model.sell_threshold:.1%}

  {B}Best Individual{R}    : {best_name} (val F1={best_f1:.4f})

  {B}Ensemble Val{R}       : acc={val_metrics['accuracy']:.2%}  f1={val_metrics['f1']:.4f}
  {B}Ensemble Test{R}      : acc={test_metrics['accuracy']:.2%}  f1={test_metrics['f1']:.4f}

  Models saved        : {save_dir}
  Total time          : {format_duration(total_time)}
{G}{'═' * 72}
  ✓ ML model test complete — ready for integration
{'═' * 72}{R}
""")


if __name__ == "__main__":
    main()
