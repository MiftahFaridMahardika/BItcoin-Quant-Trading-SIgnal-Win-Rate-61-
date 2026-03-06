"""
BTC Quant System — Deep Learning Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sequence-based models for price direction prediction.

Since PyTorch is not available, we use sklearn MLPClassifier
with architectures that approximate LSTM/GRU behaviour:
- Flattened sequence input (60 candles × N features)
- Deep MLP layers simulating recurrent memory

Models:
  1. LSTM-style  — Deep MLP (512→256→128→64), mimics LSTM depth
  2. GRU-style   — Lighter MLP (256→128→64), mimics GRU speed
  3. Ensemble    — Soft-voting across DL models

Integration:
  Adds DL predictions to the existing 4-model ML ensemble.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
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

logger = logging.getLogger("trading")


# ══════════════════════════════════════════════════════════════
# FEATURE SELECTION  (same as ml_models.py)
# ══════════════════════════════════════════════════════════════

DL_FEATURE_COLS = [
    # Trend
    "ema_21_55_cross", "ema_stack_signal", "hma_slope", "hma_signal",
    "supertrend_dir", "ichi_signal",
    # Momentum
    "rsi", "macd_hist", "macd_cross", "roc_10", "zscore_20",
    "stoch_k", "stoch_d", "williams_r", "cci_20", "ult_osc",
    # Volatility
    "atr_pct", "atr_ratio", "vol_regime", "bb_pct_b", "bb_bandwidth",
    # Volume
    "vol_ratio", "mfi_14", "cmf_20", "obv_signal", "vwap_signal",
    # Price action
    "ret_1", "ret_6", "ret_42", "ret_180",
    "trend_structure", "hist_vol_20",
]

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


# ══════════════════════════════════════════════════════════════
# SEQUENCE DATA PREPARER
# ══════════════════════════════════════════════════════════════

class SequenceDataPreparer:
    """
    Create sequence windows from feature data for DL models.

    Given a DataFrame of N rows × F features:
      - Creates rolling windows of `seq_len` candles
      - Flattens each window → vector of size (seq_len × F)
      - Splits chronologically into train / val / test
      - Scales flattened sequences
    """

    def __init__(self, seq_len: int = 60,
                 forward_period: int = 6,
                 buy_threshold: float = 0.015,
                 sell_threshold: float = -0.015):
        self.seq_len = seq_len
        self.forward_period = forward_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: List[str] = list(DL_FEATURE_COLS)
        self.n_features: int = 0

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Forward-return labels: BUY(+1), HOLD(0), SELL(-1)."""
        fwd_ret = df["Close"].shift(-self.forward_period) / df["Close"] - 1
        labels = pd.Series(0, index=df.index, dtype=int)
        labels[fwd_ret > self.buy_threshold] = 1
        labels[fwd_ret < self.sell_threshold] = -1
        labels[fwd_ret.isna()] = np.nan
        return labels

    def create_sequences(self,
                         df: pd.DataFrame,
                         train_end: str,
                         val_end: str) -> Dict:
        """
        Build sequence arrays with chronological split.

        Returns dict with X_train, y_train, X_val, y_val, X_test, y_test.
        Each X is shape (N, seq_len × n_features).
        Labels remapped to {0, 1, 2} for XGBoost compat.
        """
        # Select available features
        available = [c for c in self.feature_cols if c in df.columns]
        missing = set(self.feature_cols) - set(available)
        if missing:
            logger.warning(f"Missing DL features: {missing}")
        self.feature_cols = available
        self.n_features = len(available)

        features = df[available].values  # (T, F)
        labels_raw = self.create_labels(df)

        # Build sequences
        T = len(df)
        X_seq = []
        y_seq = []
        idx_seq = []

        for i in range(self.seq_len, T):
            # Check: no NaN in features window or label
            window = features[i - self.seq_len: i]
            label = labels_raw.iloc[i]

            if np.isnan(window).any() or np.isnan(label):
                continue

            X_seq.append(window.flatten())  # (seq_len * F,)
            y_seq.append(int(label) + 1)    # remap {-1,0,1} → {0,1,2}
            idx_seq.append(df.index[i])

        X_all = np.array(X_seq)
        y_all = np.array(y_seq)
        idx_all = pd.DatetimeIndex(idx_seq)

        logger.info(
            f"Sequences: {len(X_all):,} total │ "
            f"shape={X_all.shape} │ "
            f"seq_len={self.seq_len} × features={self.n_features}"
        )

        # Timezone-aware split
        def _ts(date_str):
            ts = pd.Timestamp(date_str)
            if idx_all.tz is not None and ts.tz is None:
                ts = ts.tz_localize(idx_all.tz)
            return ts

        t_end = _ts(train_end)
        v_end = _ts(val_end)

        train_m = idx_all <= t_end
        val_m = (idx_all > t_end) & (idx_all <= v_end)
        test_m = idx_all > v_end

        # Scale
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_all[train_m])
        X_val = self.scaler.transform(X_all[val_m])
        X_test = self.scaler.transform(X_all[test_m])

        data = {
            "X_train": X_train,
            "y_train": y_all[train_m],
            "X_val": X_val,
            "y_val": y_all[val_m],
            "X_test": X_test,
            "y_test": y_all[test_m],
            "train_dates": idx_all[train_m],
            "val_dates": idx_all[val_m],
            "test_dates": idx_all[test_m],
            "feature_names": available,
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "input_dim": X_train.shape[1],
        }

        for split, arr in [("train", data["y_train"]),
                           ("val", data["y_val"]),
                           ("test", data["y_test"])]:
            buy = (arr == 2).sum()
            hold = (arr == 1).sum()
            sell = (arr == 0).sum()
            total = len(arr)
            if total > 0:
                logger.info(
                    f"  {split:5s}: {total:,} │ "
                    f"BUY={buy}({buy/total:.0%}) "
                    f"HOLD={hold}({hold/total:.0%}) "
                    f"SELL={sell}({sell/total:.0%})"
                )

        return data


# ══════════════════════════════════════════════════════════════
# LSTM-STYLE MODEL (Deep MLP)
# ══════════════════════════════════════════════════════════════

class LSTMStyleModel:
    """
    LSTM-approximation using deep MLP.

    Architecture:
        Input (seq_len × features) → 512 → 256 → 128 → 64 → 3 classes
    Dropout: alpha=0.001 (L2 regularisation)
    Training: early stopping, adaptive LR
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.name = "lstm_style"
        self.model: Optional[MLPClassifier] = None
        self.is_trained = False

        self.hidden_layers = cfg.get("hidden_layers", (512, 256, 128, 64))
        self.alpha = cfg.get("alpha", 0.001)
        self.batch_size = cfg.get("batch_size", 64)
        self.max_iter = cfg.get("max_iter", 500)
        self.lr_init = cfg.get("lr_init", 0.001)
        self.patience = cfg.get("patience", 30)

    def build(self) -> MLPClassifier:
        """Create fresh model."""
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate="adaptive",
            learning_rate_init=self.lr_init,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=self.patience,
            random_state=42,
            verbose=False,
        )
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train and return metrics."""
        if self.model is None:
            self.build()

        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = {
            "train_acc": accuracy_score(y_train, train_pred),
            "train_f1": f1_score(y_train, train_pred, average="weighted"),
            "val_acc": accuracy_score(y_val, val_pred),
            "val_f1": f1_score(y_val, val_pred, average="weighted"),
            "val_precision": precision_score(y_val, val_pred, average="weighted", zero_division=0),
            "val_recall": recall_score(y_val, val_pred, average="weighted", zero_division=0),
            "time_s": elapsed,
            "n_iter": self.model.n_iter_,
            "best_loss": float(self.model.best_loss_) if (hasattr(self.model, 'best_loss_') and self.model.best_loss_ is not None) else None,
        }

        try:
            proba = self.model.predict_proba(X_val)
            metrics["val_auc"] = roc_auc_score(
                y_val, proba, multi_class="ovr", average="weighted"
            )
        except Exception:
            metrics["val_auc"] = None

        self.is_trained = True
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


# ══════════════════════════════════════════════════════════════
# GRU-STYLE MODEL (Lighter MLP)
# ══════════════════════════════════════════════════════════════

class GRUStyleModel:
    """
    GRU-approximation using lighter MLP.

    Architecture:
        Input (seq_len × features) → 256 → 128 → 64 → 3 classes
    Faster training than LSTM-style.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.name = "gru_style"
        self.model: Optional[MLPClassifier] = None
        self.is_trained = False

        self.hidden_layers = cfg.get("hidden_layers", (256, 128, 64))
        self.alpha = cfg.get("alpha", 0.001)
        self.batch_size = cfg.get("batch_size", 64)
        self.max_iter = cfg.get("max_iter", 500)
        self.lr_init = cfg.get("lr_init", 0.001)
        self.patience = cfg.get("patience", 30)

    def build(self) -> MLPClassifier:
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate="adaptive",
            learning_rate_init=self.lr_init,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=self.patience,
            random_state=42,
            verbose=False,
        )
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        if self.model is None:
            self.build()

        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = {
            "train_acc": accuracy_score(y_train, train_pred),
            "train_f1": f1_score(y_train, train_pred, average="weighted"),
            "val_acc": accuracy_score(y_val, val_pred),
            "val_f1": f1_score(y_val, val_pred, average="weighted"),
            "val_precision": precision_score(y_val, val_pred, average="weighted", zero_division=0),
            "val_recall": recall_score(y_val, val_pred, average="weighted", zero_division=0),
            "time_s": elapsed,
            "n_iter": self.model.n_iter_,
            "best_loss": float(self.model.best_loss_) if (hasattr(self.model, 'best_loss_') and self.model.best_loss_ is not None) else None,
        }

        try:
            proba = self.model.predict_proba(X_val)
            metrics["val_auc"] = roc_auc_score(
                y_val, proba, multi_class="ovr", average="weighted"
            )
        except Exception:
            metrics["val_auc"] = None

        self.is_trained = True
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


# ══════════════════════════════════════════════════════════════
# DEEP LEARNING ENSEMBLE
# ══════════════════════════════════════════════════════════════

class DeepLearningEnsemble:
    """
    Manages LSTM-style + GRU-style models with ensemble voting.

    Usage:
        prep = SequenceDataPreparer(seq_len=60)
        data = prep.create_sequences(df_featured, "2020-12-31", "2022-12-31")

        ensemble = DeepLearningEnsemble()
        results  = ensemble.train_all(data)
        metrics  = ensemble.evaluate(data["X_test"], data["y_test"])
        ensemble.save("models/deep_learning")
    """

    MODEL_NAMES = ["lstm_style", "gru_style"]

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.models: Dict[str, object] = {}
        self.is_trained = False
        self._train_results: Dict = {}

        # Model configs
        self.lstm_config = cfg.get("lstm", {})
        self.gru_config = cfg.get("gru", {})

    def train_all(self, data: Dict) -> Dict[str, Dict]:
        """Train both DL models."""
        logger.info("Training Deep Learning models...")
        t0 = time.perf_counter()
        results = {}

        # LSTM-style
        logger.info("  Training LSTM-style model...")
        lstm = LSTMStyleModel(self.lstm_config)
        lstm.build()
        metrics_lstm = lstm.train(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
        )
        self.models["lstm_style"] = lstm
        results["lstm_style"] = metrics_lstm
        logger.info(
            f"    LSTM: val_acc={metrics_lstm['val_acc']:.4f} "
            f"val_f1={metrics_lstm['val_f1']:.4f} "
            f"iters={metrics_lstm['n_iter']} ({metrics_lstm['time_s']:.1f}s)"
        )

        # GRU-style
        logger.info("  Training GRU-style model...")
        gru = GRUStyleModel(self.gru_config)
        gru.build()
        metrics_gru = gru.train(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
        )
        self.models["gru_style"] = gru
        results["gru_style"] = metrics_gru
        logger.info(
            f"    GRU:  val_acc={metrics_gru['val_acc']:.4f} "
            f"val_f1={metrics_gru['val_f1']:.4f} "
            f"iters={metrics_gru['n_iter']} ({metrics_gru['time_s']:.1f}s)"
        )

        self.is_trained = True
        self._train_results = results

        total = time.perf_counter() - t0
        logger.info(f"DL models trained in {total:.1f}s")
        return results

    def predict(self, X: np.ndarray,
                weights: Optional[Dict[str, float]] = None) -> Dict:
        """Soft-voting ensemble prediction."""
        if not self.models:
            raise RuntimeError("No models trained.")

        if weights is None:
            weights = {n: 1.0 for n in self.models}

        total_w = sum(weights.get(n, 1.0) for n in self.models)
        weighted_proba = None

        for name, model_obj in self.models.items():
            proba = model_obj.predict_proba(X)
            w = weights.get(name, 1.0) / total_w
            if weighted_proba is None:
                weighted_proba = proba * w
            else:
                weighted_proba += proba * w

        classes = self.models[list(self.models)[0]].model.classes_
        final_idx = np.argmax(weighted_proba, axis=1)
        final_pred = classes[final_idx]
        confidence = np.max(weighted_proba, axis=1)

        return {
            "prediction": final_pred,
            "probabilities": weighted_proba,
            "confidence": confidence,
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 label: str = "Test") -> Dict:
        """Evaluate ensemble on held-out set."""
        result = self.predict(X)
        preds = result["prediction"]

        metrics = {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average="weighted", zero_division=0),
            "recall": recall_score(y, preds, average="weighted", zero_division=0),
            "f1": f1_score(y, preds, average="weighted"),
            "confusion_matrix": confusion_matrix(y, preds),
            "report": classification_report(
                y, preds, target_names=["SELL", "HOLD", "BUY"], zero_division=0
            ),
        }

        try:
            metrics["auc"] = roc_auc_score(
                y, result["probabilities"],
                multi_class="ovr", average="weighted",
            )
        except Exception:
            metrics["auc"] = None

        # Per-model accuracy
        per_model = {}
        for name, model_obj in self.models.items():
            indiv = model_obj.predict(X)
            per_model[name] = accuracy_score(y, indiv)
        metrics["per_model_acc"] = per_model

        return metrics

    # ──────────────────────────────────────────────────────
    # INTEGRATION WITH TRADITIONAL ML
    # ──────────────────────────────────────────────────────

    def integrate_with_ml(self,
                          ml_prediction: Dict,
                          dl_X: np.ndarray,
                          dl_weight: float = 0.3) -> Dict:
        """
        Combine DL ensemble with traditional ML ensemble predictions.

        Parameters
        ----------
        ml_prediction : dict
            Output from MLSignalModel.ensemble_predict()
            Must contain 'probabilities' key.
        dl_X : np.ndarray
            Sequence features for DL prediction.
        dl_weight : float
            Weight for DL in combined ensemble (0-1).

        Returns
        -------
        dict with combined prediction, probabilities, confidence.
        """
        dl_result = self.predict(dl_X)

        ml_proba = ml_prediction["probabilities"]
        dl_proba = dl_result["probabilities"]

        # Align shapes (both should be N × 3)
        n = min(len(ml_proba), len(dl_proba))
        ml_proba = ml_proba[-n:]
        dl_proba = dl_proba[-n:]

        ml_w = 1.0 - dl_weight
        combined = ml_proba * ml_w + dl_proba * dl_weight

        classes = np.array([0, 1, 2])
        final_idx = np.argmax(combined, axis=1)
        final_pred = classes[final_idx]
        confidence = np.max(combined, axis=1)

        return {
            "prediction": final_pred,
            "probabilities": combined,
            "confidence": confidence,
            "ml_weight": ml_w,
            "dl_weight": dl_weight,
        }

    # ──────────────────────────────────────────────────────
    # SAVE / LOAD
    # ──────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save DL models to disk."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        for name, model_obj in self.models.items():
            joblib.dump(model_obj.model, d / f"{name}.joblib")

        logger.info(f"DL models saved → {d}")

    def load(self, directory: str) -> None:
        """Load DL models from disk."""
        d = Path(directory)

        for name, cls in [("lstm_style", LSTMStyleModel),
                          ("gru_style", GRUStyleModel)]:
            p = d / f"{name}.joblib"
            if p.exists():
                model_obj = cls()
                model_obj.model = joblib.load(p)
                model_obj.is_trained = True
                self.models[name] = model_obj

        self.is_trained = bool(self.models)
        logger.info(f"DL models loaded ← {d}  ({len(self.models)} models)")

    # ──────────────────────────────────────────────────────
    # REPORTING
    # ──────────────────────────────────────────────────────

    def print_training_report(self, results: Dict[str, Dict]) -> None:
        """Pretty-print DL training results."""
        C  = "\033[36m"
        G  = "\033[32m"
        B  = "\033[1m"
        R  = "\033[0m"

        print(f"\n{C}{'═' * 72}")
        print(f"  DEEP LEARNING TRAINING RESULTS")
        print(f"{'═' * 72}{R}")

        header = (
            f"  {'Model':<16s} │ {'Train Acc':>9s} │ {'Val Acc':>9s} │ "
            f"{'Val F1':>8s} │ {'Val AUC':>8s} │ {'Iters':>6s} │ {'Time':>6s}"
        )
        print(f"\n{B}{header}{R}")
        print(f"  {'─' * 70}")

        for name, m in results.items():
            auc_str = f"{m['val_auc']:.4f}" if m.get("val_auc") else "  N/A "
            print(
                f"  {name:<16s} │ {m['train_acc']:8.2%}  │ {m['val_acc']:8.2%}  │ "
                f"{m['val_f1']:7.4f}  │ {auc_str:>7s}  │ "
                f"{m.get('n_iter', '?'):>5}  │ {m['time_s']:5.1f}s"
            )

        print(f"  {'─' * 70}")
        print(f"{C}{'═' * 72}{R}\n")

    def print_evaluation_report(self, metrics: Dict,
                                label: str = "TEST SET") -> None:
        """Pretty-print DL evaluation metrics."""
        C = "\033[36m"
        G = "\033[32m"
        B = "\033[1m"
        R = "\033[0m"

        print(f"\n{C}{'═' * 72}")
        print(f"  DL ENSEMBLE EVALUATION — {label}")
        print(f"{'═' * 72}{R}")

        print(f"\n{B}  Aggregate Metrics:{R}")
        print(f"  {'─' * 50}")
        print(f"  Accuracy   : {metrics['accuracy']:.2%}")
        print(f"  Precision  : {metrics['precision']:.2%}")
        print(f"  Recall     : {metrics['recall']:.2%}")
        print(f"  F1 Score   : {metrics['f1']:.4f}")
        if metrics.get("auc"):
            print(f"  ROC AUC    : {metrics['auc']:.4f}")

        print(f"\n{B}  Per-Model Accuracy:{R}")
        print(f"  {'─' * 50}")
        for name, acc in metrics.get("per_model_acc", {}).items():
            print(f"  {name:<16s} : {acc:.2%}")

        cm = metrics["confusion_matrix"]
        print(f"\n{B}  Confusion Matrix (SELL / HOLD / BUY):{R}")
        print(f"  {'─' * 50}")
        labels = ["SELL", "HOLD", " BUY"]
        print(f"  {'':>12s}  {'SELL':>6s}  {'HOLD':>6s}  {'BUY':>6s}")
        for i, lbl in enumerate(labels):
            row_str = "  ".join(f"{cm[i][j]:5d}" for j in range(len(cm[i])))
            print(f"  {lbl:>12s}  {row_str}")

        print(f"\n{B}  Classification Report:{R}")
        print(metrics["report"])
        print(f"{C}{'═' * 72}{R}\n")

    def print_comparison(self, dl_metrics: Dict,
                         ml_metrics: Dict) -> None:
        """Print DL vs ML comparison table."""
        C = "\033[36m"
        G = "\033[32m"
        Y = "\033[33m"
        RED = "\033[31m"
        B = "\033[1m"
        R = "\033[0m"

        print(f"\n{C}{'═' * 72}")
        print(f"  DEEP LEARNING vs TRADITIONAL ML COMPARISON")
        print(f"{'═' * 72}{R}")

        header = (
            f"  {'Metric':<20s} │ {'DL Ensemble':>14s} │ {'ML Ensemble':>14s} │ {'Winner':>10s}"
        )
        print(f"\n{B}{header}{R}")
        print(f"  {'─' * 66}")

        comparisons = [
            ("Accuracy",  dl_metrics["accuracy"],  ml_metrics["accuracy"]),
            ("Precision", dl_metrics["precision"], ml_metrics["precision"]),
            ("Recall",    dl_metrics["recall"],    ml_metrics["recall"]),
            ("F1 Score",  dl_metrics["f1"],        ml_metrics["f1"]),
        ]

        if dl_metrics.get("auc") and ml_metrics.get("auc"):
            comparisons.append(
                ("ROC AUC", dl_metrics["auc"], ml_metrics["auc"])
            )

        for name, dl_val, ml_val in comparisons:
            if dl_val > ml_val:
                winner = f"{G}DL ✓{R}"
            elif ml_val > dl_val:
                winner = f"{Y}ML ✓{R}"
            else:
                winner = "  Tie"
            print(
                f"  {name:<20s} │ {dl_val:>13.4f}  │ {ml_val:>13.4f}  │ {winner}"
            )

        # Per-model breakdown
        print(f"\n{B}  Individual Model Accuracies:{R}")
        print(f"  {'─' * 50}")

        all_models = {}
        for name, acc in ml_metrics.get("per_model_acc", {}).items():
            all_models[name] = acc
        for name, acc in dl_metrics.get("per_model_acc", {}).items():
            all_models[name] = acc

        sorted_models = sorted(all_models.items(), key=lambda x: -x[1])
        for rank, (name, acc) in enumerate(sorted_models, 1):
            is_dl = name in ("lstm_style", "gru_style")
            tag = f"{C}[DL]{R}" if is_dl else f"{Y}[ML]{R}"
            print(f"  {rank}. {name:<16s} {tag}  {acc:.2%}")

        print(f"\n{C}{'═' * 72}{R}\n")
