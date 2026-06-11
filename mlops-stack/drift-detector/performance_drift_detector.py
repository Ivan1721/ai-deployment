"""
performance_drift_detector.py
─────────────────────────────
Detects performance metric drifts (R², RMSE, MAE) using statistical
tests (effect size, t-test, EWMA) and alerts when significant changes
are detected. Agnostic to model type — works with any float metrics.
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy import stats
import logging

log = logging.getLogger(__name__)


class PerformanceDriftDetector:
    """
    Detects performance metric drifts using sliding windows and statistical tests.
    Compares current window metrics against a fixed baseline.
    """

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        p_value_threshold: float = 0.05,
        effect_size_threshold: float = 0.05,
        ewma_alpha: float = 0.3,
    ):
        self.baseline_metrics      = baseline_metrics
        self.p_value_threshold     = p_value_threshold
        self.effect_size_threshold = effect_size_threshold
        self.ewma_alpha            = ewma_alpha

        self.metric_history = {k: [] for k in baseline_metrics}
        self.ewma_values    = dict(baseline_metrics)

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics: R², RMSE, MAE."""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return {
            "r2":   float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae":  float(mean_absolute_error(y_true, y_pred)),
        }

    def detect_drift(self, current_metrics: Dict[str, float]) -> Tuple[bool, Dict]:
        """
        Detect drift using three tests:
          1. Effect size: |current - baseline| > threshold
          2. T-test:      one-sample t-test against baseline (≥5 observations)
          3. EWMA:        sustained deviation via exponential moving average

        Returns (is_drift, details_dict). Drift requires ≥2 metrics flagged.
        """
        details = {
            "drift_detected": False,
            "metrics":        current_metrics,
            "baseline":       self.baseline_metrics,
            "tests":          {},
            "drifted_metrics": [],
        }
        drifted_count = 0

        for name, current in current_metrics.items():
            baseline = self.baseline_metrics.get(name)
            if baseline is None:
                continue

            self.metric_history[name].append(current)
            history    = self.metric_history[name]
            abs_change = abs(current - baseline)

            result = {
                "current":    current,
                "baseline":   baseline,
                "change_pct": round(100 * (current - baseline) / (abs(baseline) + 1e-9), 2),
                "drift":      False,
                "reason":     None,
            }

            # 1. Effect size
            if abs_change > self.effect_size_threshold:
                result["drift"]  = True
                result["reason"] = f"effect_size={abs_change:.4f} > {self.effect_size_threshold}"
                drifted_count   += 1

            # 2. T-test (min 5 observations)
            if len(history) >= 5 and not result["drift"]:
                t_stat, p_val = stats.ttest_1samp(history[-5:], baseline)
                result["t_stat"]  = round(t_stat, 4)
                result["p_value"] = round(p_val, 4)
                if p_val < self.p_value_threshold and abs_change > 0.02:
                    result["drift"]  = True
                    result["reason"] = f"t-test p={p_val:.4f} < {self.p_value_threshold}"
                    drifted_count   += 1

            # 3. EWMA
            self.ewma_values[name] = (
                self.ewma_alpha * current
                + (1 - self.ewma_alpha) * self.ewma_values[name]
            )
            ewma_change = abs(self.ewma_values[name] - baseline)
            result["ewma"] = round(self.ewma_values[name], 4)
            if ewma_change > 0.07 and not result["drift"]:
                result["drift"]  = True
                result["reason"] = f"EWMA divergence={ewma_change:.4f}"
                drifted_count   += 1

            details["tests"][name] = result
            if result["drift"]:
                details["drifted_metrics"].append(name)

        details["drift_detected"] = drifted_count >= 2
        details["drifted_count"]  = drifted_count
        details["total_metrics"]  = len(current_metrics)
        return details["drift_detected"], details

    def get_history(self, metric_name: str) -> List[float]:
        return self.metric_history.get(metric_name, [])

    def reset_history(self):
        for k in self.metric_history:
            self.metric_history[k] = []


class PerformanceDriftMonitor:
    """
    Buffers predictions + ground truth and runs PerformanceDriftDetector
    over sliding windows to detect sustained performance degradation.
    """

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        window_size: int = 30,
        num_windows: int = 3,
        **detector_kwargs,
    ):
        self.baseline_metrics    = baseline_metrics
        self.window_size         = window_size
        self.detectors           = [
            PerformanceDriftDetector(baseline_metrics, **detector_kwargs)
            for _ in range(num_windows)
        ]
        self.predictions_buffer: list = []
        self.labels_buffer:      list = []

    def add_batch(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.predictions_buffer.extend(np.asarray(y_pred, dtype=float))
        self.labels_buffer.extend(np.asarray(y_true, dtype=float))

    def check_drift(self) -> Tuple[bool, list]:
        if len(self.predictions_buffer) < self.window_size:
            return False, []

        y_pred = np.array(self.predictions_buffer[-self.window_size:])
        y_true = np.array(self.labels_buffer[-self.window_size:])

        current_metrics = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)
        results  = []
        any_drift = False

        for i, detector in enumerate(self.detectors):
            drift, details = detector.detect_drift(current_metrics)
            details["detector_id"] = i
            results.append(details)
            if drift:
                any_drift = True

        return any_drift, results

    def reset_buffer(self):
        self.predictions_buffer = []
        self.labels_buffer      = []
