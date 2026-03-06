#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  OPTUNA HYPERPARAMETER TUNING
  XGBoost │ LightGBM │ Random Forest │ MLP
  Objective: Validation F1 │ TimeSeriesSplit CV
═══════════════════════════════════════════════════════════════
"""

import sys
import time
import logging
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from engines.ml_models import MLSignalModel, ML_FEATURE_COLS

# Suppress noisy logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="  %(message)s")
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

# ── Paths ────────────────────────────────────────────────────
CONFIGS_DIR = PROJECT_ROOT / "configs"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "btcusd_4h_features.parquet"

N_TRIALS = 50       # per model (reduce from 100 for speed)
N_CV_SPLITS = 3     # TimeSeriesSplit folds


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load features, create labels, prepare train data for tuning."""
    print(f"\n{SEP}")
    print(f"{C}  LOADING DATA{R}")
    print(SEP)

    df = pd.read_parquet(FEATURES_PATH)
    print(f"  Loaded: {len(df):,} candles")

    model = MLSignalModel()
    data = model.prepare_data(df, "2020-12-31", "2022-12-31")

    # Use train + val for CV tuning
    X_tune = np.vstack([data["X_train"], data["X_val"]])
    y_tune = np.concatenate([data["y_train"], data["y_val"]])

    print(f"  Tuning set:  {X_tune.shape[0]:,} samples × {X_tune.shape[1]} features")
    print(f"  Test set:    {data['X_test'].shape[0]:,} samples")

    return data, X_tune, y_tune


# ═══════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def xgboost_objective(trial, X, y):
    """XGBoost search space."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
    }

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = xgb.XGBClassifier(**params)
        model.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[val_idx], y[val_idx])],
                  verbose=False)
        preds = model.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], preds, average="weighted"))

        # Pruning
        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def lightgbm_objective(trial, X, y):
    """LightGBM search space."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[val_idx], y[val_idx])],)
        preds = model.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], preds, average="weighted"))

        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def random_forest_objective(trial, X, y):
    """Random Forest search space."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
        "n_jobs": -1,
    }

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = RandomForestClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], preds, average="weighted"))

        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def mlp_objective(trial, X, y):
    """MLP search space."""
    n_layers = trial.suggest_int("n_layers", 2, 4)
    layers = []
    for i in range(n_layers):
        units = trial.suggest_int(f"units_l{i}", 64, 512, step=64)
        layers.append(units)

    params = {
        "hidden_layer_sizes": tuple(layers),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "solver": "adam",
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "learning_rate": "adaptive",
        "learning_rate_init": trial.suggest_float("lr_init", 1e-4, 1e-2, log=True),
        "max_iter": 300,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
        "random_state": 42,
    }

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = MLPClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], preds, average="weighted"))

        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


# ═══════════════════════════════════════════════════════════════
# RUN OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

def run_study(name, objective_fn, X, y, n_trials):
    """Run an Optuna study for one model."""
    print(f"\n  {B}Optimizing {name}...{R}")
    print(f"  Trials: {n_trials} │ CV: TimeSeriesSplit({N_CV_SPLITS})")
    print(f"  Pruner: MedianPruner")

    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    t0 = time.perf_counter()

    def callback(study, trial):
        # Progress update every 10 trials
        if trial.number % 10 == 0 or trial.number == n_trials - 1:
            best = study.best_value
            print(
                f"    Trial {trial.number:3d}/{n_trials} │ "
                f"Best F1: {best:.4f} │ "
                f"Current: {trial.value:.4f}" if trial.value else
                f"    Trial {trial.number:3d}/{n_trials} │ "
                f"Best F1: {best:.4f} │ Pruned"
            )

    study.optimize(
        lambda trial: objective_fn(trial, X, y),
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=False,
    )

    elapsed = time.perf_counter() - t0

    print(f"\n  {G}✓ {name} complete in {elapsed:.1f}s{R}")
    print(f"    Best F1:    {study.best_value:.4f}")
    print(f"    Best trial: #{study.best_trial.number}")
    print(f"    Pruned:     {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}/{n_trials}")

    return study


# ═══════════════════════════════════════════════════════════════
# SAVE BEST PARAMS
# ═══════════════════════════════════════════════════════════════

def save_best_params(studies):
    """Save best parameters to YAML."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIGS_DIR / "optimized_params.yaml"

    params = {}
    for name, study in studies.items():
        best = study.best_params.copy()
        # Convert numpy types to Python native
        for k, v in best.items():
            if isinstance(v, (np.integer,)):
                best[k] = int(v)
            elif isinstance(v, (np.floating,)):
                best[k] = float(v)
        best["best_f1_cv"] = float(study.best_value)
        best["n_trials"] = len(study.trials)
        params[name] = best

    with open(path, "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    print(f"\n  {G}✓ Best params saved → {path.relative_to(PROJECT_ROOT)}{R}")
    return params


# ═══════════════════════════════════════════════════════════════
# RETRAIN WITH BEST PARAMS
# ═══════════════════════════════════════════════════════════════

def retrain_and_compare(data, best_params):
    """Train default vs optimized models and compare."""
    print(f"\n{SEP}")
    print(f"{C}  RETRAIN & COMPARE{R}")
    print(SEP)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    results = {}

    # ── Default Models ───────────────────────────────────
    print(f"\n  {B}Training DEFAULT models...{R}")

    default_models = {
        "xgboost": xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
            eval_metric="mlogloss",
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            num_leaves=50, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            solver="adam", alpha=0.001, batch_size=64,
            learning_rate="adaptive", learning_rate_init=0.001,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42,
        ),
    }

    for name, model in default_models.items():
        t0 = time.perf_counter()
        if name == "xgboost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif name == "lightgbm":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        results[f"{name}_default"] = {
            "val_f1": f1_score(y_val, val_pred, average="weighted"),
            "val_acc": accuracy_score(y_val, val_pred),
            "test_f1": f1_score(y_test, test_pred, average="weighted"),
            "test_acc": accuracy_score(y_test, test_pred),
            "time": elapsed,
        }
        print(f"    {name:<16s} val_f1={results[f'{name}_default']['val_f1']:.4f}  test_acc={results[f'{name}_default']['test_acc']:.2%}  ({elapsed:.1f}s)")

    # ── Optimized Models ─────────────────────────────────
    print(f"\n  {B}Training OPTIMIZED models...{R}")

    for name in ["xgboost", "lightgbm", "random_forest", "mlp"]:
        bp = best_params.get(name, {})
        bp_clean = {k: v for k, v in bp.items() if k not in ("best_f1_cv", "n_trials")}

        t0 = time.perf_counter()

        if name == "xgboost":
            bp_clean["random_state"] = 42
            bp_clean["n_jobs"] = -1
            bp_clean["eval_metric"] = "mlogloss"
            model = xgb.XGBClassifier(**bp_clean)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        elif name == "lightgbm":
            bp_clean["random_state"] = 42
            bp_clean["n_jobs"] = -1
            bp_clean["verbose"] = -1
            model = lgb.LGBMClassifier(**bp_clean)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        elif name == "random_forest":
            bp_clean["random_state"] = 42
            bp_clean["n_jobs"] = -1
            model = RandomForestClassifier(**bp_clean)
            model.fit(X_train, y_train)

        elif name == "mlp":
            n_layers = bp_clean.pop("n_layers", 3)
            layers = []
            for i in range(n_layers):
                key = f"units_l{i}"
                if key in bp_clean:
                    layers.append(bp_clean.pop(key))
            # Clean remaining layer keys
            for k in list(bp_clean.keys()):
                if k.startswith("units_l"):
                    bp_clean.pop(k)

            activation = bp_clean.pop("activation", "relu")
            alpha = bp_clean.pop("alpha", 0.001)
            batch_size = bp_clean.pop("batch_size", 64)
            lr_init = bp_clean.pop("lr_init", 0.001)

            model = MLPClassifier(
                hidden_layer_sizes=tuple(layers) if layers else (256, 128, 64),
                activation=activation,
                solver="adam",
                alpha=alpha,
                batch_size=batch_size,
                learning_rate="adaptive",
                learning_rate_init=lr_init,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            )
            model.fit(X_train, y_train)

        elapsed = time.perf_counter() - t0

        test_pred = model.predict(X_test)
        val_pred = model.predict(X_val)
        results[f"{name}_optimized"] = {
            "val_f1": f1_score(y_val, val_pred, average="weighted"),
            "val_acc": accuracy_score(y_val, val_pred),
            "test_f1": f1_score(y_test, test_pred, average="weighted"),
            "test_acc": accuracy_score(y_test, test_pred),
            "time": elapsed,
        }
        print(f"    {name:<16s} val_f1={results[f'{name}_optimized']['val_f1']:.4f}  test_acc={results[f'{name}_optimized']['test_acc']:.2%}  ({elapsed:.1f}s)")

    # ── Comparison Table ─────────────────────────────────
    print(f"\n{SEP}")
    print(f"{C}  BEFORE vs AFTER TUNING{R}")
    print(SEP)

    header = (
        f"  {'Model':<16s} │ "
        f"{'Default F1':>10s} │ {'Optimized F1':>12s} │ "
        f"{'Δ F1':>8s} │ "
        f"{'Default Acc':>11s} │ {'Optimized Acc':>13s} │ "
        f"{'Δ Acc':>8s}"
    )
    print(f"\n{B}{header}{R}")
    print(f"  {'─' * 90}")

    improvements = []
    for name in ["xgboost", "lightgbm", "random_forest", "mlp"]:
        d = results[f"{name}_default"]
        o = results[f"{name}_optimized"]

        d_f1 = d["test_f1"]
        o_f1 = o["test_f1"]
        d_acc = d["test_acc"]
        o_acc = o["test_acc"]
        delta_f1 = o_f1 - d_f1
        delta_acc = o_acc - d_acc

        f1_col = G if delta_f1 > 0 else RED if delta_f1 < 0 else Y
        acc_col = G if delta_acc > 0 else RED if delta_acc < 0 else Y

        print(
            f"  {name:<16s} │ "
            f"{d_f1:>10.4f} │ {o_f1:>12.4f} │ "
            f"{f1_col}{delta_f1:>+7.4f}{R} │ "
            f"{d_acc:>10.2%} │ {o_acc:>12.2%} │ "
            f"{acc_col}{delta_acc:>+7.2%}{R}"
        )
        improvements.append({
            "model": name, "default_f1": d_f1, "optimized_f1": o_f1,
            "delta_f1": delta_f1, "default_acc": d_acc, "optimized_acc": o_acc,
            "delta_acc": delta_acc,
        })

    # Average improvement
    avg_d_f1 = np.mean([r["default_f1"] for r in improvements])
    avg_o_f1 = np.mean([r["optimized_f1"] for r in improvements])
    avg_d_acc = np.mean([r["default_acc"] for r in improvements])
    avg_o_acc = np.mean([r["optimized_acc"] for r in improvements])

    print(f"  {'─' * 90}")
    f1_col = G if avg_o_f1 > avg_d_f1 else RED
    acc_col = G if avg_o_acc > avg_d_acc else RED
    print(
        f"  {'AVERAGE':<16s} │ "
        f"{avg_d_f1:>10.4f} │ {avg_o_f1:>12.4f} │ "
        f"{f1_col}{avg_o_f1 - avg_d_f1:>+7.4f}{R} │ "
        f"{avg_d_acc:>10.2%} │ {avg_o_acc:>12.2%} │ "
        f"{acc_col}{avg_o_acc - avg_d_acc:>+7.2%}{R}"
    )

    return improvements


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"""
{C}╔{'═' * 63}╗
║{'OPTUNA HYPERPARAMETER TUNING':^63s}║
║{'XGBoost │ LightGBM │ RF │ MLP  ×  50 trials':^63s}║
╚{'═' * 63}╝{R}
""")
    t_global = time.perf_counter()

    # Load data
    data, X_tune, y_tune = load_data()

    # Run studies
    studies = {}

    print(f"\n{SEP}")
    print(f"{C}  RUNNING OPTUNA STUDIES{R}")
    print(SEP)

    objectives = {
        "xgboost": xgboost_objective,
        "lightgbm": lightgbm_objective,
        "random_forest": random_forest_objective,
        "mlp": mlp_objective,
    }

    for name, obj_fn in objectives.items():
        studies[name] = run_study(name, obj_fn, X_tune, y_tune, N_TRIALS)

    # Print best params summary
    print(f"\n{SEP}")
    print(f"{C}  BEST PARAMETERS{R}")
    print(SEP)

    for name, study in studies.items():
        print(f"\n  {B}{name} (F1={study.best_value:.4f}):{R}")
        for k, v in study.best_params.items():
            if isinstance(v, float):
                print(f"    {k:<25s}: {v:.6f}")
            else:
                print(f"    {k:<25s}: {v}")

    # Save to YAML
    best_params = save_best_params(studies)

    # Retrain & compare
    improvements = retrain_and_compare(data, best_params)

    # Final summary
    total = time.perf_counter() - t_global
    print(f"""
{G}{'═' * 65}
  ✓ Optuna Tuning Complete in {total:.1f}s
  → Best params: configs/optimized_params.yaml
  → {N_TRIALS} trials × 4 models = {N_TRIALS * 4} total trials
{'═' * 65}{R}
""")


if __name__ == "__main__":
    main()
