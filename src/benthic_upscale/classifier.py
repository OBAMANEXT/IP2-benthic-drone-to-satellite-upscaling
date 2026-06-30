"""
LightGBM multi-class classifier with early stopping and evaluation utilities.

This module operates entirely on pre-extracted feature arrays (e.g. the
spectral feature columns in the processed sample tables shipped under
``data/``). It has no dependency on raw satellite imagery, atmospheric
correction, or drone raster handling.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, cohen_kappa_score,
)
from sklearn.preprocessing import LabelEncoder


class BenthicClassifier:
    """LightGBM multi-class classifier with early stopping and evaluation utilities."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 15,
        learning_rate: float = 0.05,
    ):
        self.params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }
        self.model: Optional[lgb.Booster] = None
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.feature_names: Optional[List[str]] = None
        self.class_names: Optional[np.ndarray] = None

    # -- Training ---------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit the LightGBM model with optional early stopping on a validation set."""
        print("\nTraining LightGBM classifier ...")

        y_enc = self.label_encoder.fit_transform(y_train)
        self.class_names = self.label_encoder.classes_
        self.feature_names = feature_names
        self.params["num_class"] = len(self.class_names)

        train_ds = lgb.Dataset(X_train, label=y_enc, feature_name=feature_names)
        valid_sets = [train_ds]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_ds = lgb.Dataset(
                X_val, label=self.label_encoder.transform(y_val),
                feature_name=feature_names, reference=train_ds,
            )
            valid_sets.append(val_ds)
            valid_names.append("valid")

        self.model = lgb.train(
            self.params, train_ds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50),
            ],
        )
        print(f"Training complete -- best iteration: {self.model.best_iteration}")

    # -- Inference ----------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class name labels."""
        proba = self.model.predict(X)
        return self.label_encoder.inverse_transform(np.argmax(proba, axis=1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-class probabilities."""
        return self.model.predict(X)

    # -- Evaluation -----------------------------------------------------------

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Compute accuracy, F1-macro, F1-weighted, kappa, and confusion matrix."""
        y_pred = self.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "kappa": cohen_kappa_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=self.class_names),
            "class_names": self.class_names,
        }

    # -- Persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        self.model.save_model(str(path))
        print(f"Model saved -> {path}")

    def load(self, path: Path) -> None:
        self.model = lgb.Booster(model_file=str(path))
        print(f"Model loaded <- {path}")

    # -- Visualisation helpers ----------------------------------------------

    def plot_feature_importance(self, top_n: int = 30, figsize=(10, 12)) -> None:
        imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)

        plt.figure(figsize=figsize)
        plt.barh(range(top_n), imp["importance"].head(top_n))
        plt.yticks(range(top_n), imp["feature"].head(top_n), fontsize=9)
        plt.gca().invert_yaxis()
        plt.xlabel("Gain")
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, results: Dict, title: str = "") -> None:
        cm = results["confusion_matrix"]
        class_names = results["class_names"]
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title or "Confusion Matrix", fontsize=14, fontweight="bold")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
