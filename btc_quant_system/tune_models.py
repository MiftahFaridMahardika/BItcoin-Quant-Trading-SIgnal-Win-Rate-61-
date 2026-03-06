#!/usr/bin/env python3
"""
Hyperparameter Tuning — End-to-End Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Load featured 4H data
2. Prepare train / val / test splits
3. Baseline training (default params)
4. Optuna tuning — XGBoost, LightGBM, Random Forest, MLP
5. Save best params → configs/optimized_params.yaml
6. Retrain with best params
7. Before vs After comparison table

Usage
-----
    python tune_models.py                 # 50 trials per model (recommended)
    python tune_models.py --trials 100    # full production run
    python tune_models.py --trials 20     # quick smoke-test
    python tune_models.py --model xgboost --trials 100   # single model

Config: N_TRIALS constant below overrides --trials if you prefer hardcoding.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from utils.logger import setup_all_loggers
from utils.helpers import format_number, format_duration
from engines.ml_models import MLSignalModel
from engines.tuner import OptunaTuner

# ── ANSI ─────────────────────────────────────────────────────
C   = "\033[36m"
G   = "\033[32m"
Y   = "\033[33m"
RED = "\033[31m"
B   = "\033[1m"
D   = "\033[2m"
R   = "\033[0m"

W   = 72                          # line width
SEP = f"{C}{'═' * W}{R}"
SUB = f"  {'─' * (W - 2)}"

# ── Default tuning budget ─────────────────────────────────────
N_TRIALS    = 50        # trials per model  (change to 100 for production)
N_CV_SPLITS = 5         # TimeSeriesSplit folds


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _delta_str(baseline: float, tuned: float) -> str:
    delta = tuned - baseline
    if delta > 0.001:
        return f"{G}+{delta:+.4f}{R}"
    if delta < -0.001:
        return f"{RED}{delta:+.4f}{R}"
    return f"{D}{delta:+.4f}{R}"


def _pct_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return f"{G}{'█' * filled}{D}{'░' * (width - filled)}{R}"


# ═══════════════════════════════════════════════════════════════
# STEP: BASELINE TRAINING
# ═══════════════════════════════════════════════════════════════

def run_baseline(model: MLSignalModel, data: dict) -> dict:
    """Train with default params and return per-model val/test F1 + acc."""
    results = model.train_all(data)

    baseline = {}
    for name, m_obj in model.models.items():
        val_pred  = m_obj.predict(data["X_val"])
        test_pred = m_obj.predict(data["X_test"])
        baseline[name] = {
            "val_f1":   float(f1_score(data["y_val"],  val_pred,  average="weighted")),
            "val_acc":  float(accuracy_score(data["y_val"],  val_pred)),
            "test_f1":  float(f1_score(data["y_test"], test_pred, average="weighted")),
            "test_acc": float(accuracy_score(data["y_test"], test_pred)),
            "train_time_s": results[name]["time_s"],
        }
    return baseline


# ═══════════════════════════════════════════════════════════════
# STEP: PRINT TUNING SUMMARY PER MODEL
# ═══════════════════════════════════════════════════════════════

def print_tuning_summary(tuner: OptunaTuner) -> None:
    print(f"\n{SEP}")
    print(f"{C}  TUNING SUMMARY — Best Params Found{R}")
    print(SEP)

    for name in tuner.MODEL_NAMES:
        if name not in tuner.best_params:
            continue
        res = tuner.best_params[name]
        study = tuner.studies.get(name)

        print(f"\n  {B}{name.upper()}{R}")
        print(f"  {'─' * 60}")
        print(f"  Best CV F1   : {G}{res['best_cv_f1']:.4f}{R}")
        print(f"  Complete     : {res['n_complete']}  Pruned: {res['n_pruned']}  "
              f"Elapsed: {format_duration(res['elapsed_s'])}")

        print(f"\n  {B}Best Parameters:{R}")
        for k, v in res["best_params"].items():
            if isinstance(v, float):
                print(f"    {k:<24s}: {v:.6g}")
            else:
                print(f"    {k:<24s}: {v}")

        # Parameter importances
        imp = tuner.param_importance(name)
        if imp:
            top5 = list(imp.items())[:5]
            print(f"\n  {B}Top-5 Parameter Importances:{R}")
            max_imp = top5[0][1] if top5 else 1.0
            for param, importance in top5:
                bar = _pct_bar(importance / max_imp if max_imp > 0 else 0, width=16)
                print(f"    {param:<24s}: {bar}  {importance:.3f}")

    print(f"\n{SEP}")


# ═══════════════════════════════════════════════════════════════
# STEP: BEFORE vs AFTER COMPARISON
# ═══════════════════════════════════════════════════════════════

def print_comparison(baseline: dict, tuner: OptunaTuner) -> None:
    print(f"\n{SEP}")
    print(f"{C}  BEFORE vs AFTER TUNING — Performance Comparison{R}")
    print(SEP)

    models = [n for n in OptunaTuner.MODEL_NAMES if n in baseline and n in tuner.best_params]

    # ── Validation set ─────────────────────────────────────────
    print(f"\n  {B}{'':20s}  {'── Validation F1 ──':^30s}  {'── Test F1 ──':^26s}{R}")
    print(f"  {B}{'Model':<20s}  {'Baseline':>8s}  {'Tuned':>8s}  {'Δ':>8s}"
          f"  {'Baseline':>8s}  {'Tuned':>8s}  {'Δ':>8s}{R}")
    print(SUB)

    total_val_delta  = 0.0
    total_test_delta = 0.0

    for name in models:
        bp = tuner.best_params[name]
        bsl_vf1  = baseline[name]["val_f1"]
        bsl_tf1  = baseline[name]["test_f1"]
        tnd_vf1  = bp.get("retrain_val_f1", float("nan"))
        tnd_tf1  = bp.get("retrain_test_f1", float("nan"))

        val_delta  = tnd_vf1  - bsl_vf1
        test_delta = tnd_tf1  - bsl_tf1
        total_val_delta  += val_delta
        total_test_delta += test_delta

        print(
            f"  {name:<20s}  "
            f"{bsl_vf1:8.4f}  {tnd_vf1:8.4f}  {_delta_str(bsl_vf1, tnd_vf1):>18s}  "
            f"{bsl_tf1:8.4f}  {tnd_tf1:8.4f}  {_delta_str(bsl_tf1, tnd_tf1):>18s}"
        )

    # Avg delta
    n = len(models)
    if n > 0:
        print(SUB)
        avg_val  = total_val_delta  / n
        avg_test = total_test_delta / n
        col_v = G if avg_val  > 0 else (RED if avg_val  < -0.001 else D)
        col_t = G if avg_test > 0 else (RED if avg_test < -0.001 else D)
        print(
            f"  {'Average Delta':<20s}  "
            f"{'':>8s}  {'':>8s}  {col_v}{avg_val:>+8.4f}{R}  "
            f"{'':>8s}  {'':>8s}  {col_t}{avg_test:>+8.4f}{R}"
        )

    # ── Accuracy table ──────────────────────────────────────────
    print(f"\n  {B}{'':20s}  {'── Validation Acc ──':^30s}  {'── Test Acc ──':^26s}{R}")
    print(f"  {B}{'Model':<20s}  {'Baseline':>8s}  {'Tuned':>8s}  {'Δ':>8s}"
          f"  {'Baseline':>8s}  {'Tuned':>8s}  {'Δ':>8s}{R}")
    print(SUB)

    for name in models:
        bp = tuner.best_params[name]
        bsl_va  = baseline[name]["val_acc"]
        bsl_ta  = baseline[name]["test_acc"]
        tnd_va  = bp.get("retrain_val_acc",  float("nan"))
        tnd_ta  = bp.get("retrain_test_acc", float("nan"))

        print(
            f"  {name:<20s}  "
            f"{bsl_va:8.4f}  {tnd_va:8.4f}  {_delta_str(bsl_va, tnd_va):>18s}  "
            f"{bsl_ta:8.4f}  {tnd_ta:8.4f}  {_delta_str(bsl_ta, tnd_ta):>18s}"
        )

    # ── CV F1 vs Retrain Val F1 ─────────────────────────────────
    print(f"\n  {B}CV vs Full-Train Validation F1:{R}")
    print(f"  {B}{'Model':<20s}  {'CV F1':>8s}  {'Val F1':>8s}  {'Gap':>8s}{R}")
    print(f"  {'─' * 52}")
    for name in models:
        bp = tuner.best_params[name]
        cv_f1  = bp["best_cv_f1"]
        val_f1 = bp.get("retrain_val_f1", float("nan"))
        gap    = val_f1 - cv_f1
        col = G if gap >= -0.01 else Y
        print(
            f"  {name:<20s}  {cv_f1:8.4f}  {val_f1:8.4f}  "
            f"{col}{gap:>+8.4f}{R}"
        )

    print(f"\n{SEP}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for BTC signal models")
    parser.add_argument("--trials",  type=int, default=N_TRIALS,
                        help=f"Trials per model (default {N_TRIALS})")
    parser.add_argument("--model",   type=str, default=None,
                        choices=["xgboost", "lightgbm", "random_forest", "mlp"],
                        help="Tune a single model only (default: all four)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per-model timeout in seconds (default: none)")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = [args.model] if args.model else OptunaTuner.MODEL_NAMES

    print(f"""
{C}{'═' * W}
  HYPERPARAMETER TUNING — Optuna + TimeSeriesSplit
  Models  : {', '.join(targets)}
  Trials  : {args.trials} per model
  CV folds: {N_CV_SPLITS}
  Pruner  : MedianPruner
{'═' * W}{R}
""")

    setup_all_loggers(log_dir=str(PROJECT_ROOT / "logs"), level="WARNING")
    total_t0 = time.perf_counter()

    # ══════════════════════════════════════════════════════════
    # STEP 1: Load featured data
    # ══════════════════════════════════════════════════════════
    print(f"{B}{C}[STEP 1/7]{R} Loading featured 4H data...")
    print(f"{'─' * W}")

    feat_path = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"
    if not feat_path.exists():
        print(f"{RED}  Feature cache not found — run test_features.py first.{R}")
        return

    df = pd.read_parquet(feat_path)
    print(f"  ✓ {format_number(len(df), 0)} candles  │  {len(df.columns)} columns")
    print(f"  ✓ Range: {df.index[0]}  →  {df.index[-1]}")

    # ══════════════════════════════════════════════════════════
    # STEP 2: Prepare data
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 2/7]{R} Preparing chronological splits...")
    print(f"{'─' * W}")

    ml_model = MLSignalModel({
        "forward_period":  6,
        "buy_threshold":   0.015,
        "sell_threshold": -0.015,
    })

    data = ml_model.prepare_data(df, train_end="2020-12-31", val_end="2022-12-31")

    for label, key in [("Train", "y_train"), ("Val", "y_val"), ("Test", "y_test")]:
        arr = data[key]
        n = len(arr)
        buy  = (arr == 2).sum()
        hold = (arr == 1).sum()
        sell = (arr == 0).sum()
        print(f"  {label:<6s}: {n:>6,}  BUY={buy:>5,}({buy/n:4.0%})"
              f"  HOLD={hold:>5,}({hold/n:4.0%})  SELL={sell:>5,}({sell/n:4.0%})")

    print(f"\n  Features: {len(data['feature_names'])}")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Baseline training
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 3/7]{R} Baseline training (default params)...")
    print(f"{'─' * W}")

    t0 = time.perf_counter()
    baseline = run_baseline(ml_model, data)
    base_time = time.perf_counter() - t0

    print(f"\n  {B}{'Model':<18s}  {'Val F1':>8s}  {'Val Acc':>8s}  {'Test F1':>8s}  {'Test Acc':>9s}  {'Time':>6s}{R}")
    print(f"  {'─' * 64}")
    for name, m in baseline.items():
        print(f"  {name:<18s}  {m['val_f1']:8.4f}  {m['val_acc']:8.2%}  "
              f"{m['test_f1']:8.4f}  {m['test_acc']:8.2%}  {m['train_time_s']:5.1f}s")
    print(f"\n  Total baseline time: {format_duration(base_time)}")

    # ══════════════════════════════════════════════════════════
    # STEP 4: Optuna tuning
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 4/7]{R} Optuna tuning ({args.trials} trials × {len(targets)} models)...")
    print(f"{'─' * W}")
    print(f"  {D}Pruner: MedianPruner  │  Sampler: TPE multivariate  │  CV: TimeSeriesSplit({N_CV_SPLITS}){R}\n")

    tuner = OptunaTuner(
        data=data,
        n_trials=args.trials,
        n_cv_splits=N_CV_SPLITS,
        timeout=args.timeout,
    )

    tune_t0 = time.perf_counter()
    tuner.tune_all(models=targets)
    tune_time = time.perf_counter() - tune_t0

    print(f"\n  ✓ Tuning complete in {format_duration(tune_time)}")

    # ══════════════════════════════════════════════════════════
    # STEP 5: Tuning summary
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 5/7]{R} Tuning summary...")
    print_tuning_summary(tuner)

    # ══════════════════════════════════════════════════════════
    # STEP 6: Save best params
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 6/7]{R} Saving best params...")
    print(f"{'─' * W}")

    params_path = str(PROJECT_ROOT / "configs" / "optimized_params.yaml")
    tuner.save_best_params(params_path)
    print(f"  ✓ Saved → {params_path}")

    # ══════════════════════════════════════════════════════════
    # STEP 7: Retrain with best params + compare
    # ══════════════════════════════════════════════════════════
    print(f"\n{B}{C}[STEP 7/7]{R} Retrain with best params + comparison...")
    print(f"{'─' * W}")

    retrain_t0 = time.perf_counter()
    trained_models = tuner.retrain_tuned_models(targets)
    retrain_time = time.perf_counter() - retrain_t0

    print(f"  ✓ Retrain complete in {format_duration(retrain_time)}")

    # Re-save with retrain metrics included
    tuner.save_best_params(params_path)

    print_comparison(baseline, tuner)

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    total_time = time.perf_counter() - total_t0

    # Best model overall (by tuned val F1)
    best_name = max(
        (n for n in targets if "retrain_val_f1" in tuner.best_params.get(n, {})),
        key=lambda n: tuner.best_params[n]["retrain_val_f1"],
        default="N/A",
    )

    # Average improvement
    improvements = [
        tuner.best_params[n]["retrain_val_f1"] - baseline[n]["val_f1"]
        for n in targets
        if "retrain_val_f1" in tuner.best_params.get(n, {})
    ]
    avg_imp = sum(improvements) / len(improvements) if improvements else 0.0
    imp_col = G if avg_imp > 0 else (RED if avg_imp < -0.001 else D)

    print(f"""
{C}{'═' * W}
  FINAL SUMMARY
{'═' * W}{R}
  Models tuned        : {', '.join(targets)}
  Trials / model      : {args.trials}
  CV folds            : {N_CV_SPLITS} (TimeSeriesSplit)
  Pruner              : MedianPruner

  Best model (tuned)  : {B}{best_name}{R}
  Avg val F1 delta    : {imp_col}{avg_imp:+.4f}{R}

  Baseline time       : {format_duration(base_time)}
  Tuning time         : {format_duration(tune_time)}
  Retrain time        : {format_duration(retrain_time)}
  Total time          : {format_duration(total_time)}

  Best params saved   : {params_path}
{G}{'═' * W}
  ✓ Hyperparameter tuning complete — ready for backtesting
{'═' * W}{R}
""")


if __name__ == "__main__":
    main()
