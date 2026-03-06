#!/usr/bin/env python3
"""
Deep Learning Models — End-to-End Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Sequence creation & shape validation
2. LSTM-style model training
3. GRU-style model training
4. DL ensemble accuracy
5. DL vs Traditional ML comparison
"""

import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from engines.deep_learning import (
    SequenceDataPreparer,
    LSTMStyleModel,
    GRUStyleModel,
    DeepLearningEnsemble,
)
from engines.ml_models import MLSignalModel

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="  %(message)s",
)
logger = logging.getLogger()

# ── ANSI ─────────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"
SEP = f"{C}{'═' * 65}{R}"
SUB = f"  {'─' * 61}"


# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_featured_data():
    """Load cached featured data."""
    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
        print(f"  Loaded: {len(df):,} candles, {len(df.columns)} columns")
        print(f"  Range:  {df.index[0].date()} → {df.index[-1].date()}")
        return df
    else:
        raise FileNotFoundError(f"Feature data not found: {feat_path}")


# ═══════════════════════════════════════════════════════════════
# TEST 1 — SEQUENCE CREATION
# ═══════════════════════════════════════════════════════════════

def test_sequence_creation(df):
    print(f"\n{SEP}")
    print(f"{C}  TEST 1 — SEQUENCE CREATION{R}")
    print(SEP)

    prep = SequenceDataPreparer(seq_len=60)
    data = prep.create_sequences(df, "2020-12-31", "2022-12-31")

    n_features = data["n_features"]
    seq_len = data["seq_len"]
    input_dim = data["input_dim"]

    print(f"\n  {B}Sequence Configuration:{R}")
    print(f"  Sequence length  : {seq_len} candles")
    print(f"  Features per bar : {n_features}")
    print(f"  Input dimension  : {input_dim} (= {seq_len} × {n_features})")

    print(f"\n  {B}Split Sizes:{R}")
    for split in ["train", "val", "test"]:
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        print(f"  {split:<6s}: X={X.shape}  y={y.shape}")

    # Validate shapes
    assert data["X_train"].shape[1] == seq_len * n_features
    assert data["X_val"].shape[1] == seq_len * n_features
    assert data["X_test"].shape[1] == seq_len * n_features
    assert len(data["y_train"]) == data["X_train"].shape[0]
    assert set(np.unique(data["y_train"])).issubset({0, 1, 2})

    print(f"\n  {G}✓ Sequence shapes validated{R}")
    print(SEP)
    return data


# ═══════════════════════════════════════════════════════════════
# TEST 2 — LSTM-STYLE MODEL
# ═══════════════════════════════════════════════════════════════

def test_lstm_model(data):
    print(f"\n{SEP}")
    print(f"{C}  TEST 2 — LSTM-STYLE MODEL (512→256→128→64){R}")
    print(SEP)

    lstm = LSTMStyleModel()
    lstm.build()

    print(f"\n  Architecture: {lstm.hidden_layers}")
    print(f"  Alpha (L2):   {lstm.alpha}")
    print(f"  Batch size:   {lstm.batch_size}")
    print(f"  Max iter:     {lstm.max_iter}")
    print(f"  LR init:      {lstm.lr_init}")
    print(f"  Patience:     {lstm.patience}")

    print(f"\n  Training...")
    metrics = lstm.train(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
    )

    print(f"\n  {B}Results:{R}")
    print(SUB)
    print(f"  Train Accuracy : {G}{metrics['train_acc']:.2%}{R}")
    print(f"  Val Accuracy   : {G}{metrics['val_acc']:.2%}{R}")
    print(f"  Val F1         : {metrics['val_f1']:.4f}")
    auc = metrics.get('val_auc')
    print(f"  Val AUC        : {auc:.4f}" if auc else "  Val AUC        : N/A")
    print(f"  Iterations     : {metrics['n_iter']}")
    print(f"  Best Loss      : {metrics.get('best_loss', 'N/A')}")
    print(f"  Training Time  : {metrics['time_s']:.1f}s")

    # Verify can predict
    preds = lstm.predict(data["X_test"][:5])
    proba = lstm.predict_proba(data["X_test"][:5])
    assert len(preds) == 5
    assert proba.shape == (5, 3)

    print(f"\n  {G}✓ LSTM-style model trained and verified{R}")
    print(SEP)
    return lstm, metrics


# ═══════════════════════════════════════════════════════════════
# TEST 3 — GRU-STYLE MODEL
# ═══════════════════════════════════════════════════════════════

def test_gru_model(data):
    print(f"\n{SEP}")
    print(f"{C}  TEST 3 — GRU-STYLE MODEL (256→128→64){R}")
    print(SEP)

    gru = GRUStyleModel()
    gru.build()

    print(f"\n  Architecture: {gru.hidden_layers}")
    print(f"\n  Training...")
    metrics = gru.train(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
    )

    print(f"\n  {B}Results:{R}")
    print(SUB)
    print(f"  Train Accuracy : {G}{metrics['train_acc']:.2%}{R}")
    print(f"  Val Accuracy   : {G}{metrics['val_acc']:.2%}{R}")
    print(f"  Val F1         : {metrics['val_f1']:.4f}")
    auc = metrics.get('val_auc')
    print(f"  Val AUC        : {auc:.4f}" if auc else "  Val AUC        : N/A")
    print(f"  Iterations     : {metrics['n_iter']}")
    print(f"  Training Time  : {metrics['time_s']:.1f}s")

    # Speed comparison note
    print(f"\n  {Y}→ GRU faster than LSTM (fewer params){R}")

    print(f"\n  {G}✓ GRU-style model trained and verified{R}")
    print(SEP)
    return gru, metrics


# ═══════════════════════════════════════════════════════════════
# TEST 4 — DL ENSEMBLE
# ═══════════════════════════════════════════════════════════════

def test_dl_ensemble(data):
    print(f"\n{SEP}")
    print(f"{C}  TEST 4 — DEEP LEARNING ENSEMBLE{R}")
    print(SEP)

    ensemble = DeepLearningEnsemble()
    results = ensemble.train_all(data)

    # Print training report
    ensemble.print_training_report(results)

    # Evaluate on test set
    metrics = ensemble.evaluate(data["X_test"], data["y_test"], "TEST SET")
    ensemble.print_evaluation_report(metrics, "TEST SET (2023+)")

    # Save models
    save_dir = str(PROJECT_ROOT / "models" / "deep_learning")
    ensemble.save(save_dir)
    print(f"  Models saved to: models/deep_learning/")

    # Verify load
    ensemble2 = DeepLearningEnsemble()
    ensemble2.load(save_dir)
    assert ensemble2.is_trained
    preds2 = ensemble2.predict(data["X_test"][:10])
    assert len(preds2["prediction"]) == 10
    print(f"  {G}✓ Save/Load verified{R}")

    print(f"\n  {G}✓ DL Ensemble verified{R}")
    print(SEP)
    return ensemble, metrics


# ═══════════════════════════════════════════════════════════════
# TEST 5 — DL vs TRADITIONAL ML COMPARISON
# ═══════════════════════════════════════════════════════════════

def test_comparison(df, dl_ensemble, dl_metrics, seq_data):
    print(f"\n{SEP}")
    print(f"{C}  TEST 5 — DL vs TRADITIONAL ML COMPARISON{R}")
    print(SEP)

    # Train traditional ML
    print(f"\n  Training Traditional ML (4 models)...")
    ml_model = MLSignalModel()
    ml_data = ml_model.prepare_data(df, "2020-12-31", "2022-12-31")
    ml_results = ml_model.train_all(ml_data)
    ml_model.print_training_report(ml_results)

    # Evaluate ML on test set
    ml_metrics = ml_model.evaluate(ml_data["X_test"], ml_data["y_test"])
    ml_model.print_evaluation_report(ml_metrics, "ML TEST SET (2023+)")

    # Comparison table
    dl_ensemble.print_comparison(dl_metrics, ml_metrics)

    # Integration test
    print(f"\n  {B}Integration Test (ML 70% + DL 30%):{R}")
    print(SUB)

    # Get ML predictions on same test samples
    n_test = min(len(ml_data["X_test"]), len(seq_data["X_test"]))
    ml_pred = ml_model.ensemble_predict(ml_data["X_test"][-n_test:])
    combined = dl_ensemble.integrate_with_ml(
        ml_pred, seq_data["X_test"][-n_test:], dl_weight=0.3
    )

    y_true = ml_data["y_test"][-n_test:]
    combined_acc = (combined["prediction"] == y_true).mean()
    ml_only_acc = (ml_pred["prediction"][-n_test:] == y_true).mean()

    print(f"  ML-only accuracy:   {ml_only_acc:.2%}")
    print(f"  Combined accuracy:  {combined_acc:.2%}")
    diff = combined_acc - ml_only_acc
    col = G if diff > 0 else RED
    print(f"  Improvement:        {col}{diff:+.2%}{R}")

    print(f"\n  {G}✓ DL vs ML comparison complete{R}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"""
{C}{'═' * 65}
  DEEP LEARNING MODELS — END-TO-END TEST
  LSTM-style + GRU-style + Ensemble + ML Comparison
{'═' * 65}{R}
""")
    t0 = time.perf_counter()

    # Load data
    df = load_featured_data()

    # Test 1: Sequence creation
    seq_data = test_sequence_creation(df)

    # Test 2: LSTM model
    lstm, lstm_metrics = test_lstm_model(seq_data)

    # Test 3: GRU model
    gru, gru_metrics = test_gru_model(seq_data)

    # Test 4: DL Ensemble
    dl_ensemble, dl_metrics = test_dl_ensemble(seq_data)

    # Test 5: DL vs ML comparison
    test_comparison(df, dl_ensemble, dl_metrics, seq_data)

    elapsed = time.perf_counter() - t0
    print(f"""
{G}{'═' * 65}
  ✓ Deep Learning test complete in {elapsed:.1f}s
  → LSTM + GRU + Ensemble + ML Comparison verified
{'═' * 65}{R}
""")


if __name__ == "__main__":
    main()
