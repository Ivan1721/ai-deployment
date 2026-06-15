"""
regression.py
──────────────
Tipo de problema "regresión": métricas (mae/rmse/r2), quality gates, forma
de /predict ({"value": ...}) y concept drift vía KS test sobre la
distribución de valores predichos.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base import ProblemType


class RegressionProblem(ProblemType):
    name = "regression"
    primary_metric = "r2"

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def quality_gate(
        self, metrics: Dict[str, float], gates_cfg: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        failures: List[str] = []

        min_r2 = gates_cfg.get("min_r2")
        if min_r2 is not None and metrics.get("r2", 0.0) < min_r2:
            failures.append(f"r2 {metrics.get('r2', 0.0):.4f} < min_r2 {min_r2:.4f}")

        max_rmse = gates_cfg.get("max_rmse")
        if max_rmse is not None and metrics.get("rmse", float("inf")) > max_rmse:
            failures.append(f"rmse {metrics.get('rmse', 0.0):.4f} > max_rmse {max_rmse:.4f}")

        return (len(failures) == 0, failures)

    def build_predictions(
        self, y_pred: np.ndarray, y_proba: Optional[np.ndarray], bundle
    ) -> List[Dict[str, Any]]:
        return [{"value": round(float(v), 4)} for v in y_pred]

    def concept_drift(
        self, reference: Any, current: Any, p_value_threshold: float
    ) -> Tuple[bool, Dict[str, Any]]:
        statistic, p_value = stats.ks_2samp(reference, current)
        drift_detected = bool(p_value < p_value_threshold)
        return drift_detected, {
            "method": "ks",
            "statistic": float(statistic),
            "p_value": float(p_value),
        }
