"""
test_performance_drift.py
──────────────────────────
Unit tests for PerformanceDriftDetector module.
Tests drift detection logic, statistical tests, and EWMA.
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from performance_drift_detector import (
    PerformanceDriftDetector,
    PerformanceDriftMonitor,
)


@pytest.fixture
def baseline_metrics():
    """Baseline metrics from a well-trained model."""
    return {
        "accuracy": 0.96,
        "precision": 0.96,
        "recall": 0.96,
        "f1": 0.96,
    }


@pytest.fixture
def stable_model_predictions():
    """Generate stable predictions (no drift)."""
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, y_test


class TestPerformanceDriftDetectorBasics:
    """Basic functionality tests."""

    def test_calculate_metrics(self, stable_model_predictions):
        """Test metric calculation."""
        y_pred, y_true = stable_model_predictions

        metrics = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_detector_initialization(self, baseline_metrics):
        """Test detector can be initialized."""
        detector = PerformanceDriftDetector(baseline_metrics)

        assert detector.baseline_metrics == baseline_metrics
        assert detector.p_value_threshold == 0.05

    def test_no_drift_on_stable_data(self, baseline_metrics, stable_model_predictions):
        """Test that stable predictions don't trigger drift."""
        y_pred, y_true = stable_model_predictions

        detector = PerformanceDriftDetector(baseline_metrics)

        # Calculate metrics close to baseline
        current_metrics = {
            "accuracy": 0.95,
            "precision": 0.95,
            "recall": 0.95,
            "f1": 0.95,
        }

        drift, details = detector.detect_drift(current_metrics)

        assert drift is False, f"False positive drift: {details['drifted_metrics']}"


class TestDriftDetectionLogic:
    """Tests for drift detection mechanisms."""

    def test_drift_on_large_metric_drop(self, baseline_metrics):
        """Test drift detection when metrics drop significantly."""
        detector = PerformanceDriftDetector(
            baseline_metrics, effect_size_threshold=0.05
        )

        # Large drop in accuracy
        current_metrics = {
            "accuracy": 0.80,  # 16% drop from 0.96
            "precision": 0.85,
            "recall": 0.85,
            "f1": 0.85,
        }

        drift, details = detector.detect_drift(current_metrics)

        assert drift is True, "Should detect drift on large accuracy drop"
        assert "accuracy" in details["drifted_metrics"]

    def test_no_drift_on_small_metric_change(self, baseline_metrics):
        """Test that small metric changes don't trigger drift."""
        detector = PerformanceDriftDetector(
            baseline_metrics, effect_size_threshold=0.05
        )

        # Small change in metrics
        current_metrics = {
            "accuracy": 0.94,  # 2% drop
            "precision": 0.95,
            "recall": 0.95,
            "f1": 0.95,
        }

        drift, details = detector.detect_drift(current_metrics)

        assert drift is False, "Should not detect drift on small changes"

    def test_t_test_drift_detection(self, baseline_metrics):
        """Test that t-test detects sustained metric degradation."""
        detector = PerformanceDriftDetector(
            baseline_metrics, effect_size_threshold=0.10
        )

        # Add 5 observations that are consistently below baseline
        for _ in range(5):
            current_metrics = {
                "accuracy": 0.88,  # Consistent drop, but < 10%
                "precision": 0.88,
                "recall": 0.88,
                "f1": 0.88,
            }
            drift, _ = detector.detect_drift(current_metrics)

        # After 5 observations, t-test should trigger (p < 0.05)
        assert drift is True, "t-test should detect sustained degradation"

    def test_ewma_trend_detection(self, baseline_metrics):
        """Test EWMA detects sustained trends."""
        detector = PerformanceDriftDetector(
            baseline_metrics, effect_size_threshold=0.10, ewma_alpha=0.3
        )

        # Gradual decline
        for i in range(8):
            current_metrics = {
                "accuracy": 0.96 - (i * 0.01),
                "precision": 0.96 - (i * 0.01),
                "recall": 0.96 - (i * 0.01),
                "f1": 0.96 - (i * 0.01),
            }
            drift, details = detector.detect_drift(current_metrics)

        # After sustained degradation, EWMA should detect drift
        assert drift is True, "EWMA should detect sustained trend"

    def test_drifted_count_logic(self, baseline_metrics):
        """Test that drift requires >= 2 metrics to be flagged."""
        detector = PerformanceDriftDetector(
            baseline_metrics, effect_size_threshold=0.05
        )

        # Only 1 metric drifts
        current_metrics = {
            "accuracy": 0.80,  # Drifted
            "precision": 0.96,
            "recall": 0.96,
            "f1": 0.96,
        }

        drift, details = detector.detect_drift(current_metrics)

        assert details["drifted_count"] == 1
        # Since only 1 metric drifted, overall drift should be False (needs >= 2)
        assert drift is False

    def test_multiple_metrics_drift(self, baseline_metrics):
        """Test drift detection when multiple metrics degrade."""
        detector = PerformanceDriftDetector(
            baseline_metrics, effect_size_threshold=0.05
        )

        current_metrics = {
            "accuracy": 0.80,  # Drifted
            "precision": 0.75,  # Drifted
            "recall": 0.96,
            "f1": 0.96,
        }

        drift, details = detector.detect_drift(current_metrics)

        assert details["drifted_count"] >= 2
        assert drift is True


class TestPerformanceDriftMonitor:
    """Tests for PerformanceDriftMonitor wrapper."""

    def test_monitor_initialization(self, baseline_metrics):
        """Test monitor can be initialized."""
        monitor = PerformanceDriftMonitor(baseline_metrics, window_size=30, num_windows=3)

        assert monitor.window_size == 30
        assert len(monitor.detectors) == 3

    def test_monitor_buffer_accumulation(self, baseline_metrics):
        """Test that monitor accumulates predictions."""
        monitor = PerformanceDriftMonitor(
            baseline_metrics, window_size=10, num_windows=1
        )

        y_pred = np.array([0, 1, 2, 0, 1])
        y_true = np.array([0, 1, 2, 0, 1])

        monitor.add_batch(y_pred, y_true)

        assert len(monitor.predictions_buffer) == 5
        assert len(monitor.labels_buffer) == 5

    def test_monitor_check_drift_insufficient_data(self, baseline_metrics):
        """Test that check_drift returns False when buffer < window_size."""
        monitor = PerformanceDriftMonitor(
            baseline_metrics, window_size=100, num_windows=1
        )

        y_pred = np.array([0, 1, 2] * 10)
        y_true = np.array([0, 1, 2] * 10)

        monitor.add_batch(y_pred, y_true)
        drift, results = monitor.check_drift()

        assert drift is False
        assert len(results) == 0

    def test_monitor_check_drift_with_data(self, baseline_metrics):
        """Test check_drift with sufficient data."""
        monitor = PerformanceDriftMonitor(
            baseline_metrics, window_size=30, num_windows=1
        )

        # Generate stable predictions
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test[: min(30, len(X_test))])
        y_true = y_test[: min(30, len(y_test))]

        monitor.add_batch(y_pred, y_true)
        drift, results = monitor.check_drift()

        assert isinstance(drift, bool)
        assert len(results) >= 1
        assert "metrics" in results[0]
        assert "drift_detected" in results[0]

    def test_monitor_reset_buffer(self, baseline_metrics):
        """Test that buffer can be reset."""
        monitor = PerformanceDriftMonitor(
            baseline_metrics, window_size=10, num_windows=1
        )

        y_pred = np.array([0, 1, 2] * 5)
        y_true = np.array([0, 1, 2] * 5)

        monitor.add_batch(y_pred, y_true)
        assert len(monitor.predictions_buffer) > 0

        monitor.reset_buffer()

        assert len(monitor.predictions_buffer) == 0
        assert len(monitor.labels_buffer) == 0


class TestEdgeCases:
    """Edge case tests."""

    def test_all_zeros_predictions(self, baseline_metrics):
        """Test with all predictions being same class."""
        detector = PerformanceDriftDetector(baseline_metrics)

        y_true = np.array([0, 1, 2] * 5)
        y_pred = np.array([0, 0, 0] * 5)  # All class 0

        current_metrics = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)

        drift, details = detector.detect_drift(current_metrics)

        # Should handle gracefully
        assert "accuracy" in details["metrics"]
        assert current_metrics["accuracy"] < baseline_metrics["accuracy"]

    def test_empty_predictions(self, baseline_metrics):
        """Test with empty arrays."""
        monitor = PerformanceDriftMonitor(baseline_metrics, window_size=10)

        y_pred = np.array([])
        y_true = np.array([])

        monitor.add_batch(y_pred, y_true)
        drift, results = monitor.check_drift()

        assert drift is False


class TestHistoryTracking:
    """Tests for metric history and EWMA tracking."""

    def test_metric_history_accumulation(self, baseline_metrics):
        """Test that metric history accumulates."""
        detector = PerformanceDriftDetector(baseline_metrics)

        for i in range(5):
            current_metrics = {
                "accuracy": 0.95 - (i * 0.01),
                "precision": 0.95 - (i * 0.01),
                "recall": 0.95 - (i * 0.01),
                "f1": 0.95 - (i * 0.01),
            }
            detector.detect_drift(current_metrics)

        history = detector.get_history("accuracy")
        assert len(history) == 5

    def test_ewma_smoothing(self, baseline_metrics):
        """Test EWMA values smooth over time."""
        detector = PerformanceDriftDetector(baseline_metrics, ewma_alpha=0.3)

        # Apply sudden drop
        current_metrics_drop = {
            "accuracy": 0.80,
            "precision": 0.80,
            "recall": 0.80,
            "f1": 0.80,
        }
        detector.detect_drift(current_metrics_drop)

        ewma_after_drop = detector.ewma_values["accuracy"]

        # Apply recovery
        current_metrics_recover = {
            "accuracy": 0.96,
            "precision": 0.96,
            "recall": 0.96,
            "f1": 0.96,
        }
        detector.detect_drift(current_metrics_recover)

        ewma_after_recover = detector.ewma_values["accuracy"]

        # EWMA should move towards recovery but not instantly
        assert ewma_after_recover > ewma_after_drop
        assert ewma_after_recover < 0.96  # Not fully recovered in 1 step


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
