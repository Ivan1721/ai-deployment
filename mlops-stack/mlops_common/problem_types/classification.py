"""
classification.py
──────────────────
Tipo de problema "clasificación": métricas (accuracy/precision/recall/f1),
quality gates, forma de /predict (class_id/class_name/probability) y
concept drift vía chi2 sobre la distribución de clases predichas.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .base import ProblemType


class ClassificationProblem(ProblemType):
    name = "classification"
    primary_metric = "accuracy"

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    def quality_gate(
        self, metrics: Dict[str, float], gates_cfg: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        failures: List[str] = []

        min_accuracy = gates_cfg.get("min_accuracy")
        if min_accuracy is not None and metrics.get("accuracy", 0.0) < min_accuracy:
            failures.append(
                f"accuracy {metrics.get('accuracy', 0.0):.4f} < min_accuracy {min_accuracy:.4f}"
            )

        min_f1 = gates_cfg.get("min_f1")
        if min_f1 is not None and metrics.get("f1", 0.0) < min_f1:
            failures.append(f"f1 {metrics.get('f1', 0.0):.4f} < min_f1 {min_f1:.4f}")

        return (len(failures) == 0, failures)

    def build_predictions(
        self, y_pred: np.ndarray, y_proba: Optional[np.ndarray], bundle
    ) -> List[Dict[str, Any]]:
        target_names = bundle.target_names
        predictions = []
        for i, class_id in enumerate(y_pred):
            class_id = int(class_id)
            entry: Dict[str, Any] = {
                "class_id": class_id,
                "class_name": target_names[class_id],
            }
            if y_proba is not None:
                entry["probability"] = round(float(y_proba[i][class_id]), 4)
            predictions.append(entry)
        return predictions

    def concept_drift(
        self, reference: Any, current: Any, p_value_threshold: float
    ) -> Tuple[bool, Dict[str, Any]]:
        observed = np.asarray(current, dtype=float)
        expected = np.maximum(np.asarray(reference, dtype=float), 1e-6)
        expected = expected / expected.sum() * observed.sum()

        statistic, p_value = stats.chisquare(observed, f_exp=expected)
        drift_detected = bool(p_value < p_value_threshold)
        return drift_detected, {
            "method": "chi2",
            "statistic": float(statistic),
            "p_value": float(p_value),
        }
