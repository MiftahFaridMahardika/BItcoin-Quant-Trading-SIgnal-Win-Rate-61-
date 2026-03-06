"""
BTC Quant Trading System — ML Signal Enhancement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ensemble of 4 classifiers to predict forward price direction
from the 71 technical-indicator features.

Models:
  1. XGBoost     — gradient boosting (best for tabular)
  2. LightGBM    — fast gradient boosting
  3. Random Forest — bagging for stability
  4. MLP          — neural network (sklearn)
  5. Ensemble     — soft-voting across all four

Labels:
  +1  BUY   forward return > +threshold
   0  HOLD  return within threshold
  -1  SELL  forward return < -threshold

Train / Val / Test split is strictly chronological (NO shuffle).
Walk-forward evaluation supported.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger("trading")


# ══════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════

# Features safe for ML — normalised / bounded indicators only.
# Excludes: raw OHLCV, absolute-price columns (ema_21, bb_upper, …),
# OBV (cumulative, scale-dependent), and pre-computed *_signal columns
# (those encode the rule-based system — we let ML learn its own rules).

ML_FEATURE_COLS = [
    # Trend — relative / directional
    "ema_21_55_cross",      # {-1,0,+1}
    "ema_stack_signal",     # {-1,0,+1}
    "hma_slope",            # continuous
    "hma_signal",           # {-1,0,+1}
    "supertrend_dir",       # {-1,+1}
    "ichi_signal",          # {-1,0,+1}
    # Momentum — all bounded / normalised
    "rsi",                  # 0-100
    "macd_hist",            # continuous (centred ~0)
    "macd_cross",           # {-1,0,+1}
    "roc_10",               # pct
    "zscore_20",            # ~N(0,1)
    "stoch_k",              # 0-100
    "stoch_d",              # 0-100
    "williams_r",           # -100 to 0
    "cci_20",               # continuous
    "ult_osc",              # 0-100
    # Volatility — normalised
    "atr_pct",              # ATR as % of price
    "atr_ratio",            # ratio vs baseline
    "vol_regime",           # {0,1,2,3}
    "bb_pct_b",             # 0-1
    "bb_bandwidth",         # continuous
    # Volume — normalised
    "vol_ratio",            # ratio vs SMA
    "mfi_14",               # 0-100
    "cmf_20",               # -1 to +1
    "obv_signal",           # {-1,0,+1}
    "vwap_signal",          # {-1,0,+1}
    # Price action — pct / ordinal
    "ret_1",                # pct return
    "ret_6",                # pct return
    "ret_42",               # pct return
    "ret_180",              # pct return
    "trend_structure",      # {-1,0,+1}
    "hist_vol_20",          # annualised vol
]

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


class MLSignalModel:
    """
    ML-based signal enhancement using an ensemble of four classifiers.

    Workflow:
        model = MLSignalModel()
        data  = model.prepare_data(df_featured, "2020-12-31", "2022-12-31")
        results = model.train_all(data)
        metrics = model.evaluate(data["X_test"], data["y_test"])
        model.save("models/trained")
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.forward_period: int = cfg.get("forward_period", 6)
        self.buy_threshold: float = cfg.get("buy_threshold", 0.015)
        self.sell_threshold: float = cfg.get("sell_threshold", -0.015)
        self.feature_cols: List[str] = list(ML_FEATURE_COLS)

        self.models: Dict[str, object] = {}
        self.scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False
        self._train_results: Dict = {}

        # Optimized params from Optuna
        self.optimized_params: Dict = {}
        if "optimized_params_path" in cfg:
            self._load_optimized_params(cfg["optimized_params_path"])

    def _load_optimized_params(self, path: str):
        """Load optimized parameters from YAML."""
        try:
            p = Path(path)
            if p.exists():
                with open(p, "r") as f:
                    data = yaml.safe_load(f)
                    for model_name in self.MODEL_NAMES:
                        if model_name in data and "params" in data[model_name]:
                            self.optimized_params[model_name] = data[model_name]["params"]
                logger.info(f"Loaded optimized params for {list(self.optimized_params.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load optimized params: {e}")

    # ══════════════════════════════════════════════════════════
    # DATA PREPARATION
    # ══════════════════════════════════════════════════════════

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Forward-return classification labels.

        +1  BUY:  fwd_ret > +buy_threshold
         0  HOLD: within threshold
        -1  SELL: fwd_ret < sell_threshold

        Uses .shift(-N) so the last N rows become NaN (dropped later).
        """
        fwd_ret = df["Close"].shift(-self.forward_period) / df["Close"] - 1
        labels = pd.Series(0, index=df.index, dtype=int)
        labels[fwd_ret > self.buy_threshold] = 1
        labels[fwd_ret < self.sell_threshold] = -1
        labels[fwd_ret.isna()] = np.nan
        return labels

    def prepare_data(
        self,
        df: pd.DataFrame,
        train_end: str,
        val_end: str,
    ) -> Dict:
        """
        Build train / val / test arrays with chronological split.

        Parameters
        ----------
        df : pd.DataFrame
            Featured OHLCV (output of FeatureEngine).
        train_end : str
            Last date in the training set (inclusive).
        val_end : str
            Last date in the validation set (inclusive).

        Returns
        -------
        dict with X_train, y_train, X_val, y_val, X_test, y_test,
             plus date indices and feature names.
        """
        # Select only ML-safe features that exist in df
        available = [c for c in self.feature_cols if c in df.columns]
        missing = set(self.feature_cols) - set(available)
        if missing:
            logger.warning(f"Missing ML features (skipped): {missing}")
        self.feature_cols = available

        X = df[available].copy()
        y = self.create_labels(df)

        # Drop rows with any NaN in features or label
        valid = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid]
        y = y[valid].astype(int)

        # Timezone-aware timestamp comparison
        def _ts(date_str):
            ts = pd.Timestamp(date_str)
            if X.index.tz is not None and ts.tz is None:
                ts = ts.tz_localize(X.index.tz)
            return ts

        t_end = _ts(train_end)
        v_end = _ts(val_end)

        train_m = X.index <= t_end
        val_m = (X.index > t_end) & (X.index <= v_end)
        test_m = X.index > v_end

        # Scale
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X[train_m])
        X_val = self.scaler.transform(X[val_m])
        X_test = self.scaler.transform(X[test_m])

        # Remap labels {-1,0,1} → {0,1,2} for XGBoost compatibility
        # Inverse: subtract 1 after prediction
        remap = lambda a: a + 1

        data = {
            "X_train": X_train,
            "y_train": remap(y[train_m].values),
            "X_val": X_val,
            "y_val": remap(y[val_m].values),
            "X_test": X_test,
            "y_test": remap(y[test_m].values),
            "train_dates": X[train_m].index,
            "val_dates": X[val_m].index,
            "test_dates": X[test_m].index,
            "feature_names": available,
        }

        logger.info(
            f"Data prepared — train={train_m.sum():,}, "
            f"val={val_m.sum():,}, test={test_m.sum():,}  "
            f"features={len(available)}"
        )

        # Label distribution
        for split, arr in [("train", data["y_train"]),
                           ("val", data["y_val"]),
                           ("test", data["y_test"])]:
            buy = (arr == 2).sum()
            hold = (arr == 1).sum()
            sell = (arr == 0).sum()
            total = len(arr)
            logger.info(
                f"  {split:5s} labels — BUY={buy}({buy/total:.0%}) "
                f"HOLD={hold}({hold/total:.0%}) SELL={sell}({sell/total:.0%})"
            )

        return data

    # ══════════════════════════════════════════════════════════
    # MODEL BUILDERS
    # ══════════════════════════════════════════════════════════

    def _build_xgboost(self) -> xgb.XGBClassifier:
        params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1,
            "eval_metric": "mlogloss",
        }
        # Override with optimized if exists
        if "xgboost" in self.optimized_params:
            params.update(self.optimized_params["xgboost"])
        return xgb.XGBClassifier(**params)

    def _build_lightgbm(self) -> lgb.LGBMClassifier:
        params = {
            "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
            "num_leaves": 50, "min_child_samples": 20, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        if "lightgbm" in self.optimized_params:
            params.update(self.optimized_params["lightgbm"])
        return lgb.LGBMClassifier(**params)

    def _build_random_forest(self) -> RandomForestClassifier:
        params = {
            "n_estimators": 300, "max_depth": 10, "min_samples_split": 10,
            "min_samples_leaf": 5, "max_features": "sqrt", "random_state": 42, "n_jobs": -1,
        }
        if "random_forest" in self.optimized_params:
            params.update(self.optimized_params["random_forest"])
        return RandomForestClassifier(**params)

    def _build_mlp(self) -> MLPClassifier:
        # Default
        h_layers = (256, 128, 64)
        activation = "relu"
        alpha = 0.001
        batch_size = 64
        lr_init = 0.001

        if "mlp" in self.optimized_params:
            p = self.optimized_params["mlp"]
            n_layers = p.get("n_layers", 3)
            h_layers = tuple(p.get(f"units_l{i}", 128) for i in range(n_layers))
            activation = p.get("activation", activation)
            alpha = p.get("alpha", alpha)
            batch_size = p.get("batch_size", batch_size)
            lr_init = p.get("learning_rate_init", lr_init)

        return MLPClassifier(
            hidden_layer_sizes=h_layers,
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

    MODEL_NAMES = ["xgboost", "lightgbm", "random_forest", "mlp"]

    # ══════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════

    def _make_model(self, name: str):
        """Instantiate an untrained model by name."""
        builders = {
            "xgboost": self._build_xgboost,
            "lightgbm": self._build_lightgbm,
            "random_forest": self._build_random_forest,
            "mlp": self._build_mlp,
        }
        builder = builders.get(name)
        if builder is None:
            raise ValueError(f"Unknown model: {name}")
        return builder()

    def _train_one(
        self,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        """Train a single model and return metrics."""
        model = self._make_model(name)

        t0 = time.perf_counter()

        # XGBoost / LightGBM support eval_set for early stopping
        if name == "xgboost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif name == "lightgbm":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
        else:
            model.fit(X_train, y_train)

        elapsed = time.perf_counter() - t0

        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        metrics = {
            "train_acc": accuracy_score(y_train, train_pred),
            "train_f1": f1_score(y_train, train_pred, average="weighted"),
            "val_acc": accuracy_score(y_val, val_pred),
            "val_f1": f1_score(y_val, val_pred, average="weighted"),
            "val_precision": precision_score(y_val, val_pred, average="weighted", zero_division=0),
            "val_recall": recall_score(y_val, val_pred, average="weighted", zero_division=0),
            "time_s": elapsed,
        }

        # AUC (multi-class one-vs-rest)
        try:
            proba = model.predict_proba(X_val)
            metrics["val_auc"] = roc_auc_score(
                y_val, proba, multi_class="ovr", average="weighted",
            )
        except Exception:
            metrics["val_auc"] = None

        self.models[name] = model
        return metrics

    def train_all(self, data: Dict) -> Dict[str, Dict]:
        """
        Train all four models and return per-model metrics.
        """
        logger.info("Training all ML models...")
        t0 = time.perf_counter()

        results = {}
        for name in self.MODEL_NAMES:
            logger.info(f"  Training {name}...")
            metrics = self._train_one(
                name,
                data["X_train"], data["y_train"],
                data["X_val"], data["y_val"],
            )
            results[name] = metrics
            logger.info(
                f"    {name}: val_acc={metrics['val_acc']:.4f}  "
                f"val_f1={metrics['val_f1']:.4f}  ({metrics['time_s']:.1f}s)"
            )

        self.is_trained = True
        self._train_results = results

        total = time.perf_counter() - t0
        logger.info(f"All models trained in {total:.1f}s")

        return results

    # ══════════════════════════════════════════════════════════
    # ENSEMBLE PREDICTION
    # ══════════════════════════════════════════════════════════

    def ensemble_predict(
        self,
        X: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Soft-voting ensemble across all trained models.

        Parameters
        ----------
        X : np.ndarray
            Scaled feature matrix.
        weights : dict, optional
            Per-model weight (default: equal weighting).

        Returns
        -------
        dict with prediction, probabilities, confidence,
             individual_predictions.
        """
        if not self.models:
            raise RuntimeError("No models trained — call train_all() first.")

        if weights is None:
            weights = {n: 1.0 for n in self.models}

        total_w = sum(weights[n] for n in self.models)
        indiv_preds = {}
        weighted_proba = None

        for name, model in self.models.items():
            proba = model.predict_proba(X)       # shape (N, n_classes)
            w = weights.get(name, 1.0) / total_w
            if weighted_proba is None:
                weighted_proba = proba * w
            else:
                weighted_proba += proba * w
            indiv_preds[name] = model.predict(X)

        # Classes are sorted by sklearn: [0, 1, 2] (SELL, HOLD, BUY)
        classes = self.models[list(self.models)[0]].classes_
        final_idx = np.argmax(weighted_proba, axis=1)
        final_pred = classes[final_idx]
        confidence = np.max(weighted_proba, axis=1)

        return {
            "prediction": final_pred,
            "probabilities": weighted_proba,
            "confidence": confidence,
            "individual": indiv_preds,
        }

    # ══════════════════════════════════════════════════════════
    # EVALUATION
    # ══════════════════════════════════════════════════════════

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label: str = "Test",
    ) -> Dict:
        """
        Evaluate the ensemble on a hold-out set.

        Returns dict with accuracy, precision, recall, f1,
        confusion_matrix, and classification_report.
        """
        result = self.ensemble_predict(X)
        preds = result["prediction"]

        metrics = {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average="weighted", zero_division=0),
            "recall": recall_score(y, preds, average="weighted", zero_division=0),
            "f1": f1_score(y, preds, average="weighted"),
            "confusion_matrix": confusion_matrix(y, preds),
            "report": classification_report(
                y, preds, target_names=["SELL", "HOLD", "BUY"], zero_division=0,
            ),
        }

        try:
            metrics["auc"] = roc_auc_score(
                y, result["probabilities"],
                multi_class="ovr", average="weighted",
            )
        except Exception:
            metrics["auc"] = None

        # Per-model test accuracy
        per_model = {}
        for name, preds_i in result["individual"].items():
            per_model[name] = accuracy_score(y, preds_i)
        metrics["per_model_acc"] = per_model

        return metrics

    # ══════════════════════════════════════════════════════════
    # FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════

    def feature_importance(self) -> pd.DataFrame:
        """
        Aggregate feature importances from tree-based models.

        Returns DataFrame sorted by average importance.
        """
        imp = pd.DataFrame({"feature": self.feature_cols})
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp[name] = model.feature_importances_

        imp_cols = [c for c in imp.columns if c != "feature"]
        if imp_cols:
            imp["avg"] = imp[imp_cols].mean(axis=1)
            imp.sort_values("avg", ascending=False, inplace=True)
        return imp.reset_index(drop=True)

    # ══════════════════════════════════════════════════════════
    # SAVE / LOAD
    # ══════════════════════════════════════════════════════════

    def save(self, directory: str) -> None:
        """Persist all models, scaler, and feature list to disk."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            joblib.dump(model, d / f"{name}.joblib")

        if self.scaler is not None:
            joblib.dump(self.scaler, d / "scaler.joblib")

        with open(d / "features.json", "w") as f:
            json.dump(self.feature_cols, f)

        with open(d / "config.json", "w") as f:
            json.dump({
                "forward_period": self.forward_period,
                "buy_threshold": self.buy_threshold,
                "sell_threshold": self.sell_threshold,
            }, f)

        logger.info(f"Models saved → {d}")

    def load(self, directory: str) -> None:
        """Load persisted models from disk."""
        d = Path(directory)

        for name in self.MODEL_NAMES:
            p = d / f"{name}.joblib"
            if p.exists():
                self.models[name] = joblib.load(p)

        scaler_p = d / "scaler.joblib"
        if scaler_p.exists():
            self.scaler = joblib.load(scaler_p)

        feat_p = d / "features.json"
        if feat_p.exists():
            with open(feat_p) as f:
                self.feature_cols = json.load(f)

        cfg_p = d / "config.json"
        if cfg_p.exists():
            with open(cfg_p) as f:
                cfg = json.load(f)
                self.forward_period = cfg.get("forward_period", self.forward_period)
                self.buy_threshold = cfg.get("buy_threshold", self.buy_threshold)
                self.sell_threshold = cfg.get("sell_threshold", self.sell_threshold)

        self.is_trained = bool(self.models)
        logger.info(f"Models loaded ← {d}  ({len(self.models)} models)")

    # ══════════════════════════════════════════════════════════
    # REPORTING
    # ══════════════════════════════════════════════════════════

    def print_training_report(self, results: Dict[str, Dict]) -> None:
        """Pretty-print training results table."""
        C  = "\033[36m"
        G  = "\033[32m"
        Y  = "\033[33m"
        RED = "\033[31m"
        B  = "\033[1m"
        D  = "\033[2m"
        R  = "\033[0m"

        print(f"\n{C}{'═' * 72}")
        print(f"  ML MODEL TRAINING RESULTS")
        print(f"{'═' * 72}{R}")

        header = (
            f"  {'Model':<16s} │ {'Train Acc':>9s} │ {'Val Acc':>9s} │ "
            f"{'Val F1':>8s} │ {'Val AUC':>8s} │ {'Time':>6s}"
        )
        print(f"\n{B}{header}{R}")
        print(f"  {'─' * 66}")

        for name, m in results.items():
            auc_str = f"{m['val_auc']:.4f}" if m.get("val_auc") else "  N/A "
            print(
                f"  {name:<16s} │ {m['train_acc']:8.2%}  │ {m['val_acc']:8.2%}  │ "
                f"{m['val_f1']:7.4f}  │ {auc_str:>7s}  │ {m['time_s']:5.1f}s"
            )

        print(f"  {'─' * 66}")
        print(f"{C}{'═' * 72}{R}\n")

    def print_evaluation_report(
        self,
        metrics: Dict,
        label: str = "TEST SET",
    ) -> None:
        """Pretty-print evaluation metrics."""
        C  = "\033[36m"
        G  = "\033[32m"
        B  = "\033[1m"
        D  = "\033[2m"
        R  = "\033[0m"

        print(f"\n{C}{'═' * 72}")
        print(f"  ENSEMBLE EVALUATION — {label}")
        print(f"{'═' * 72}{R}")

        print(f"\n{B}  Aggregate Metrics:{R}")
        print(f"  {'─' * 50}")
        print(f"  Accuracy   : {metrics['accuracy']:.2%}")
        print(f"  Precision  : {metrics['precision']:.2%}")
        print(f"  Recall     : {metrics['recall']:.2%}")
        print(f"  F1 Score   : {metrics['f1']:.4f}")
        if metrics.get("auc"):
            print(f"  ROC AUC    : {metrics['auc']:.4f}")

        print(f"\n{B}  Per-Model Test Accuracy:{R}")
        print(f"  {'─' * 50}")
        for name, acc in metrics.get("per_model_acc", {}).items():
            print(f"  {name:<16s} : {acc:.2%}")

        cm = metrics["confusion_matrix"]
        print(f"\n{B}  Confusion Matrix (SELL / HOLD / BUY):{R}")
        print(f"  {'─' * 50}")
        labels = ["SELL", "HOLD", " BUY"]
        print(f"  {'':>12s}  {'SELL':>6s}  {'HOLD':>6s}  {'BUY':>6s}")
        for i, row_label in enumerate(labels):
            row_str = "  ".join(f"{cm[i][j]:5d}" for j in range(len(cm[i])))
            print(f"  {row_label:>12s}  {row_str}")

        print(f"\n{B}  Classification Report:{R}")
        print(metrics["report"])
        print(f"{C}{'═' * 72}{R}\n")

    def print_feature_importance(self, top_n: int = 20) -> None:
        """Print top N features by average importance."""
        C = "\033[36m"
        G = "\033[32m"
        B = "\033[1m"
        R = "\033[0m"

        imp = self.feature_importance()
        if "avg" not in imp.columns:
            print("No tree-based models with feature_importances_ found.")
            return

        print(f"\n{C}{'═' * 72}")
        print(f"  TOP {top_n} FEATURE IMPORTANCES (avg across tree models)")
        print(f"{'═' * 72}{R}")

        top = imp.head(top_n)
        max_imp = top["avg"].max() if len(top) > 0 else 1

        imp_model_cols = [c for c in top.columns if c not in ("feature", "avg")]

        header = f"  {'#':>3s}  {'Feature':<22s} │ {'Avg':>7s}"
        for mc in imp_model_cols:
            header += f" │ {mc[:6]:>6s}"
        header += " │ Bar"
        print(f"\n{B}{header}{R}")
        print(f"  {'─' * (56 + 9*len(imp_model_cols))}")

        for rank, (_, row) in enumerate(top.iterrows(), 1):
            bar_len = int(row["avg"] / max_imp * 25)
            bar = f"{G}{'█' * bar_len}{R}"
            line = f"  {rank:3d}  {row['feature']:<22s} │ {row['avg']:7.4f}"
            for mc in imp_model_cols:
                line += f" │ {row[mc]:6.4f}"
            line += f" │ {bar}"
            print(line)

        print(f"\n{C}{'═' * 72}{R}\n")
