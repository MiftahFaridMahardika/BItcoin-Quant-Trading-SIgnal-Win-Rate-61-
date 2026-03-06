"""
Optuna Hyperparameter Tuner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hedge-fund-grade hyperparameter search for all four ML classifiers.

Strategy
--------
* Objective  : Weighted F1 on TimeSeriesSplit cross-validation (train set only)
* Pruning    : Median pruner — kills bad trials after each fold
* Trials     : Configurable (default 100 per model)
* CV splits  : 5-fold TimeSeriesSplit

Search Spaces
-------------
XGBoost      : n_estimators, max_depth, lr, subsample, colsample_bytree,
               min_child_weight, reg_alpha, reg_lambda
LightGBM     : n_estimators, max_depth, lr, num_leaves, min_child_samples,
               subsample, colsample_bytree, reg_alpha, reg_lambda
RandomForest : n_estimators, max_depth, min_samples_split, min_samples_leaf,
               max_features
MLP          : architecture (n_layers + layer sizes), activation, alpha,
               learning_rate_init, batch_size
"""

from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

import lightgbm as lgb
import xgboost as xgb

# Suppress noisy library warnings during tuning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("trading")


# ══════════════════════════════════════════════════════════════
# PROGRESS CALLBACK
# ══════════════════════════════════════════════════════════════

class _TqdmCallback:
    """Show a live progress bar while Optuna runs trials."""

    def __init__(self, n_trials: int, model_name: str):
        self._pbar = tqdm(
            total=n_trials,
            desc=f"  {model_name:<14s}",
            ncols=80,
            leave=True,
            unit="trial",
        )
        self._best = -np.inf

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        self._pbar.update(1)
        try:
            if study.best_value > self._best:
                self._best = study.best_value
            self._pbar.set_postfix({"best_f1": f"{self._best:.4f}"})
        except ValueError:
            pass

    def close(self) -> None:
        self._pbar.close()


# ══════════════════════════════════════════════════════════════
# TUNER
# ══════════════════════════════════════════════════════════════

class OptunaTuner:
    """
    Hyperparameter tuner for XGBoost, LightGBM, RandomForest, and MLP.

    Parameters
    ----------
    data : dict
        Output of ``MLSignalModel.prepare_data()`` — must contain
        X_train, y_train, X_val, y_val, X_test, y_test.
    n_trials : int
        Number of Optuna trials per model (default 100).
    n_cv_splits : int
        TimeSeriesSplit folds on the training set (default 5).
    timeout : int, optional
        Wall-clock seconds budget per model (None = unlimited).
    """

    MODEL_NAMES: List[str] = ["xgboost", "lightgbm", "random_forest", "mlp"]

    def __init__(
        self,
        data: Dict,
        n_trials: int = 100,
        n_cv_splits: int = 5,
        timeout: Optional[int] = None,
    ) -> None:
        self.data = data
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.timeout = timeout

        self.best_params: Dict[str, Dict] = {}
        self.studies: Dict[str, optuna.Study] = {}
        self.tuning_times: Dict[str, float] = {}

    # ──────────────────────────────────────────────────────────
    # CROSS-VALIDATION HELPER
    # ──────────────────────────────────────────────────────────

    def _cv_score(
        self,
        trial: optuna.Trial,
        build_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        fit_kwargs: Optional[Dict] = None,
    ) -> float:
        """
        TimeSeriesSplit CV with Median Pruner integration.

        Reports intermediate F1 after each fold so Optuna can prune
        unpromising trials early.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        fold_scores: List[float] = []
        fit_kwargs = fit_kwargs or {}

        for fold_idx, (tr_idx, vl_idx) in enumerate(tscv.split(X)):
            X_tr, X_vl = X[tr_idx], X[vl_idx]
            y_tr, y_vl = y[tr_idx], y[vl_idx]

            model = build_fn()
            model.fit(X_tr, y_tr, **fit_kwargs)

            preds = model.predict(X_vl)
            score = f1_score(y_vl, preds, average="weighted", zero_division=0)
            fold_scores.append(score)

            # Report for pruning
            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    # ──────────────────────────────────────────────────────────
    # OBJECTIVE FUNCTIONS
    # ──────────────────────────────────────────────────────────

    def _objective_xgboost(self, trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_class":         3,
            "eval_metric":       "mlogloss",
            "random_state":      42,
            "n_jobs":            -1,
            "verbosity":         0,
        }

        def build():
            return xgb.XGBClassifier(**params)

        return self._cv_score(trial, build, self.data["X_train"], self.data["y_train"])

    def _objective_lightgbm(self, trial: optuna.Trial) -> float:
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":          trial.suggest_int("max_depth", 3, 12),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples":  trial.suggest_int("min_child_samples", 5, 50),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state":       42,
            "n_jobs":             -1,
            "verbose":            -1,
        }

        def build():
            return lgb.LGBMClassifier(**params)

        return self._cv_score(trial, build, self.data["X_train"], self.data["y_train"])

    def _objective_random_forest(self, trial: optuna.Trial) -> float:
        use_max_depth = trial.suggest_categorical("use_max_depth", [True, False])
        max_depth = (
            trial.suggest_int("max_depth", 3, 20)
            if use_max_depth
            else None
        )

        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 800),
            "max_depth":       max_depth,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state":    42,
            "n_jobs":          -1,
        }

        def build():
            return RandomForestClassifier(**params)

        return self._cv_score(trial, build, self.data["X_train"], self.data["y_train"])

    def _objective_mlp(self, trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        sizes = [
            trial.suggest_int(f"units_l{i}", 32, 512)
            for i in range(n_layers)
        ]
        # Enforce descending layer sizes for stable MLP
        sizes = sorted(sizes, reverse=True)

        params = {
            "hidden_layer_sizes":  tuple(sizes),
            "activation":          trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha":               trial.suggest_float("alpha", 1e-5, 0.1, log=True),
            "learning_rate_init":  trial.suggest_float("learning_rate_init", 1e-4, 0.01, log=True),
            "batch_size":          trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "solver":              "adam",
            "learning_rate":       "adaptive",
            "max_iter":            300,
            "early_stopping":      True,
            "validation_fraction": 0.1,
            "n_iter_no_change":    15,
            "random_state":        42,
        }

        def build():
            return MLPClassifier(**params)

        return self._cv_score(trial, build, self.data["X_train"], self.data["y_train"])

    # ──────────────────────────────────────────────────────────
    # TUNE ONE MODEL
    # ──────────────────────────────────────────────────────────

    def tune(self, model_name: str) -> Dict:
        """
        Run Optuna study for a single model.

        Returns the best params dict + best CV F1.
        """
        objectives = {
            "xgboost":       self._objective_xgboost,
            "lightgbm":      self._objective_lightgbm,
            "random_forest": self._objective_random_forest,
            "mlp":           self._objective_mlp,
        }

        if model_name not in objectives:
            raise ValueError(f"Unknown model: {model_name}")

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, self.n_trials // 10),
            n_warmup_steps=1,
            interval_steps=1,
        )
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)

        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=sampler,
            study_name=model_name,
        )

        cb = _TqdmCallback(self.n_trials, model_name)

        t0 = time.perf_counter()
        try:
            study.optimize(
                objectives[model_name],
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=[cb],
                show_progress_bar=False,
                n_jobs=1,  # parallel trials cause sklearn thread conflicts
            )
        finally:
            cb.close()

        elapsed = time.perf_counter() - t0
        self.tuning_times[model_name] = elapsed
        self.studies[model_name] = study

        best = study.best_trial
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        # Strip internal params that shouldn't be passed to constructors
        clean_params = {
            k: v for k, v in best.params.items()
            if k not in ("use_max_depth",)
        }
        # Rebuild max_depth for RandomForest
        if model_name == "random_forest" and not best.params.get("use_max_depth", True):
            clean_params["max_depth"] = None

        result = {
            "best_cv_f1": best.value,
            "best_params": clean_params,
            "n_complete": complete,
            "n_pruned": pruned,
            "elapsed_s": elapsed,
        }

        self.best_params[model_name] = result
        logger.info(
            f"  {model_name}: best_cv_f1={best.value:.4f}  "
            f"complete={complete}  pruned={pruned}  ({elapsed:.0f}s)"
        )
        return result

    # ──────────────────────────────────────────────────────────
    # TUNE ALL MODELS
    # ──────────────────────────────────────────────────────────

    def tune_all(self, models: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Run tuning for all (or a subset of) models sequentially.

        Parameters
        ----------
        models : list, optional
            Subset of MODEL_NAMES to tune.  Defaults to all four.
        """
        targets = models or self.MODEL_NAMES
        results = {}
        for name in targets:
            results[name] = self.tune(name)
        return results

    # ──────────────────────────────────────────────────────────
    # RETRAIN WITH BEST PARAMS
    # ──────────────────────────────────────────────────────────

    def build_tuned_model(self, model_name: str):
        """
        Instantiate a model with the best params found by Optuna.
        The returned model is *untrained*.
        """
        if model_name not in self.best_params:
            raise RuntimeError(f"Tune {model_name} before calling build_tuned_model.")

        params = dict(self.best_params[model_name]["best_params"])  # copy

        if model_name == "xgboost":
            # Remove non-constructor keys
            params.pop("num_class", None)
            params.pop("eval_metric", None)
            params.pop("verbosity", None)
            return xgb.XGBClassifier(
                **params,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )

        elif model_name == "lightgbm":
            return lgb.LGBMClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        elif model_name == "random_forest":
            # Strip Optuna-internal keys
            params.pop("use_max_depth", None)
            params.pop("units_l0", None)
            params.pop("units_l1", None)
            params.pop("units_l2", None)
            return RandomForestClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
            )

        elif model_name == "mlp":
            # Reconstruct hidden_layer_sizes from individual unit params
            n_layers = params.pop("n_layers")
            sizes = sorted(
                [params.pop(f"units_l{i}") for i in range(n_layers)],
                reverse=True,
            )
            return MLPClassifier(
                hidden_layer_sizes=tuple(sizes),
                activation=params.pop("activation"),
                alpha=params.pop("alpha"),
                learning_rate_init=params.pop("learning_rate_init"),
                batch_size=params.pop("batch_size"),
                solver="adam",
                learning_rate="adaptive",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            )

        raise ValueError(f"Unknown model: {model_name}")

    def retrain_tuned_models(
        self,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Retrain all tuned models on the full training set.

        Returns a dict {model_name: fitted_model}.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score as f1,
            precision_score,
            recall_score,
        )

        targets = model_names or list(self.best_params.keys())
        trained: Dict[str, object] = {}

        for name in targets:
            m = self.build_tuned_model(name)
            X_tr = self.data["X_train"]
            y_tr = self.data["y_train"]
            X_vl = self.data["X_val"]
            y_vl = self.data["y_val"]

            t0 = time.perf_counter()
            if name == "xgboost":
                m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            elif name == "lightgbm":
                m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)])
            else:
                m.fit(X_tr, y_tr)
            elapsed = time.perf_counter() - t0

            val_pred  = m.predict(X_vl)
            test_pred = m.predict(self.data["X_test"])

            self.best_params[name]["retrain_val_f1"]  = float(f1(y_vl, val_pred, average="weighted"))
            self.best_params[name]["retrain_val_acc"]  = float(accuracy_score(y_vl, val_pred))
            self.best_params[name]["retrain_test_f1"] = float(f1(self.data["y_test"], test_pred, average="weighted"))
            self.best_params[name]["retrain_test_acc"] = float(accuracy_score(self.data["y_test"], test_pred))
            self.best_params[name]["retrain_time_s"]   = elapsed

            trained[name] = m

        return trained

    # ──────────────────────────────────────────────────────────
    # SAVE BEST PARAMS
    # ──────────────────────────────────────────────────────────

    def save_best_params(self, path: str = "configs/optimized_params.yaml") -> None:
        """
        Persist all best params + metadata to a YAML file.
        """
        out: Dict = {
            "tuning_metadata": {
                "date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
                "n_trials":    self.n_trials,
                "n_cv_splits": self.n_cv_splits,
                "feature_count": int(self.data["X_train"].shape[1]),
                "train_samples": int(self.data["X_train"].shape[0]),
            },
        }

        for name, result in self.best_params.items():
            # Convert numpy scalars → native Python for YAML serialisation
            clean_best = _to_native(result["best_params"])
            entry: Dict = {
                "best_cv_f1": round(float(result["best_cv_f1"]), 6),
                "n_complete":  int(result["n_complete"]),
                "n_pruned":    int(result["n_pruned"]),
                "elapsed_s":   round(float(result["elapsed_s"]), 1),
                "params":      clean_best,
            }
            if "retrain_val_f1" in result:
                entry["retrain_val_f1"]  = round(float(result["retrain_val_f1"]), 6)
                entry["retrain_val_acc"] = round(float(result["retrain_val_acc"]), 6)
                entry["retrain_test_f1"] = round(float(result["retrain_test_f1"]), 6)
                entry["retrain_test_acc"]= round(float(result["retrain_test_acc"]), 6)
            out[name] = entry

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(out, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"Best params saved → {path}")

    # ──────────────────────────────────────────────────────────
    # IMPORTANCES SUMMARY
    # ──────────────────────────────────────────────────────────

    def param_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """Return Optuna's parameter importances for a completed study."""
        study = self.studies.get(model_name)
        if study is None or len(study.trials) < 5:
            return None
        try:
            return optuna.importance.get_param_importances(study)
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    """Recursively convert numpy/optuna types to Python natives for YAML."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
