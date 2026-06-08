"""
performance_drift_detector.py
─────────────────────────────
Detects performance metric drifts (Accuracy, Precision, Recall, F1)
using statistical tests (t-test, EWMA) and alerts when significant
changes are detected.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy import stats
import logging

log = logging.getLogger(__name__)


class PerformanceDriftDetector:
    """
    Detects performance metric drifts using sliding windows and statistical tests.
    Compares current window metrics against baseline (reference) metrics.
    """

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        p_value_threshold: float = 0.05,
        effect_size_threshold: float = 0.05,
        ewma_alpha: float = 0.3,
    ):
        """
        Args:
            baseline_metrics: Dict with keys (accuracy, precision, recall, f1)
            p_value_threshold: p-value threshold for t-test (default 0.05)
            effect_size_threshold: Min % change to flag as drift (default 5%)
            ewma_alpha: Alpha for EWMA smoothing (default 0.3)
        """
        self.baseline_metrics = baseline_metrics
        self.p_value_threshold = p_value_threshold
        self.effect_size_threshold = effect_size_threshold
        self.ewma_alpha = ewma_alpha

        # Track history for EWMA
        self.metric_history = {k: [] for k in baseline_metrics.keys()}
        self.ewma_values = {k: v for k, v in baseline_metrics.items()}

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "weighted",
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method (weighted, macro, micro)

        Returns:
            Dict with metrics: accuracy, precision, recall, f1
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

    def detect_drift(
        self,
        current_metrics: Dict[str, float],
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Detect if current metrics show significant drift from baseline.

        Uses:
        1. Effect size check: |current - baseline| > threshold
        2. T-test: Compares current window against historical baseline
        3. EWMA trend: Exponential moving average to detect sustained changes

        Args:
            current_metrics: Dict with current window metrics

        Returns:
            (is_drift, details_dict)
        """
        details = {
            "drift_detected": False,
            "metrics": current_metrics,
            "baseline": self.baseline_metrics,
            "tests": {},
            "drifted_metrics": [],
        }

        drifted_count = 0

        for metric_name, current_value in current_metrics.items():
            baseline_value = self.baseline_metrics.get(metric_name)
            if baseline_value is None:
                continue

            self.metric_history[metric_name].append(current_value)

            # Ensure we have enough history for t-test
            history = self.metric_history[metric_name]
            test_result = {
                "current": current_value,
                "baseline": baseline_value,
                "change_pct": round(
                    100 * (current_value - baseline_value) / (baseline_value + 1e-6), 2
                ),
                "drift": False,
                "reason": None,
            }

            # 1. Effect size check
            abs_change = abs(current_value - baseline_value)
            if abs_change > self.effect_size_threshold:
                test_result["drift"] = True
                test_result["reason"] = (
                    f"Effect size ({abs_change:.4f}) exceeds threshold ({self.effect_size_threshold})"
                )
                drifted_count += 1

            # 2. T-test if we have enough history (min 5 samples)
            if len(history) >= 5 and not test_result["drift"]:
                # One-sample t-test: is current mean different from baseline?
                t_stat, p_val = stats.ttest_1samp(history[-5:], baseline_value)
                test_result["t_stat"] = round(t_stat, 4)
                test_result["p_value"] = round(p_val, 4)

                if p_val < self.p_value_threshold and abs_change > 0.02:
                    test_result["drift"] = True
                    test_result["reason"] = (
                        f"t-test p={p_val:.4f} < {self.p_value_threshold} "
                        f"(change: {abs_change:.4f})"
                    )
                    drifted_count += 1

            # 3. EWMA trend
            self.ewma_values[metric_name] = (
                self.ewma_alpha * current_value
                + (1 - self.ewma_alpha) * self.ewma_values[metric_name]
            )
            ewma_change = abs(self.ewma_values[metric_name] - baseline_value)

            if ewma_change > 0.07 and not test_result["drift"]:
                test_result["drift"] = True
                test_result["reason"] = (
                    f"EWMA divergence ({ewma_change:.4f}) indicates sustained trend"
                )
                drifted_count += 1

            test_result["ewma"] = round(self.ewma_values[metric_name], 4)
            details["tests"][metric_name] = test_result

            if test_result["drift"]:
                details["drifted_metrics"].append(metric_name)

        # Overall drift: if >= 2 metrics show drift
        details["drift_detected"] = drifted_count >= 2
        details["drifted_count"] = drifted_count
        details["total_metrics"] = len(current_metrics)

        return details["drift_detected"], details

    def get_history(self, metric_name: str) -> List[float]:
        """Get historical values for a metric."""
        return self.metric_history.get(metric_name, [])

    def reset_history(self):
        """Reset metric history while keeping baseline."""
        for k in self.metric_history.keys():
            self.metric_history[k] = []


class PerformanceDriftMonitor:
    """
    Wrapper that tracks performance metrics over time and manages
    multiple drift detectors for sliding windows.
    """

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        window_size: int = 30,
        num_windows: int = 3,
        **detector_kwargs,
    ):
        """
        Args:
            baseline_metrics: Reference metrics
            window_size: Size of each sliding window
            num_windows: Number of detectors for different windows
            **detector_kwargs: Passed to PerformanceDriftDetector
        """
        self.baseline_metrics = baseline_metrics
        self.window_size = window_size
        self.detectors = [
            PerformanceDriftDetector(baseline_metrics, **detector_kwargs)
            for _ in range(num_windows)
        ]
        self.predictions_buffer = []
        self.labels_buffer = []

    def add_batch(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Add predictions and labels to buffer.

        Args:
            y_pred: Predictions
            y_true: True labels
        """
        self.predictions_buffer.extend(y_pred)
        self.labels_buffer.extend(y_true)

    def check_drift(self) -> Tuple[bool, List[Dict]]:
        """
        Check if current buffer shows performance drift.
        Uses rotating detectors on sliding windows.

        Returns:
            (any_drift_detected, [detector_results, ...])
        """
        if len(self.predictions_buffer) < self.window_size:
            return False, []

        # Use last window_size samples
        y_pred = np.array(self.predictions_buffer[-self.window_size :])
        y_true = np.array(self.labels_buffer[-self.window_size :])

        current_metrics = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)

        results = []
        any_drift = False

        for i, detector in enumerate(self.detectors):
            drift, details = detector.detect_drift(current_metrics)
            details["detector_id"] = i
            results.append(details)
            if drift:
                any_drift = True

        return any_drift, results

    def reset_buffer(self):
        """Clear buffers after checking."""
        self.predictions_buffer = []
        self.labels_buffer = []
