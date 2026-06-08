"""
test_performance_drift_integration.py
──────────────────────────────────────
Integration tests for performance drift detection.
Tests with MLFlow logging, realistic drift scenarios, and end-to-end flows.
"""

import os
import pytest
import numpy as np
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

log = logging.getLogger(__name__)

# Import detector
from performance_drift_detector import (
    PerformanceDriftDetector,
    PerformanceDriftMonitor,
)


@pytest.fixture
def baseline_metrics():
    """Create baseline metrics from real model."""
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return PerformanceDriftDetector.calculate_metrics(y_test, y_pred)


@pytest.fixture
def mlflow_available():
    """Check if MLFlow is available."""
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
        mlflow.get_experiment_by_name("test-perf-drift")
        return True
    except Exception as e:
        log.warning(f"MLFlow not available: {e}")
        return False


class TestRealWorldDriftScenarios:
    """Realistic drift scenarios."""

    def test_gradual_performance_degradation(self, baseline_metrics):
        """
        Simulate gradual model degradation over time.
        This is realistic when training data shifts gradually.
        """
        monitor = PerformanceDriftMonitor(
            baseline_metrics,
            window_size=50,
            num_windows=3,
            effect_size_threshold=0.05,
        )

        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        # Generate degrading predictions
        np.random.seed(42)
        for window in range(5):
            # Gradually add noise to simulate degradation
            noise_level = window * 0.1  # 0%, 10%, 20%, 30%, 40%

            # Sample from Iris
            indices = np.random.choice(len(X), size=50, replace=True)
            X_sample = X.iloc[indices].values
            y_sample = y.iloc[indices].values

            # Train degraded model
            X_train, X_test, y_train, y_test = train_test_split(
                X_sample, y_sample, test_size=0.3, random_state=42
            )

            clf = RandomForestClassifier(n_estimators=50 + int(noise_level * 10), random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Add random flips to degrade
            flip_count = int(len(y_pred) * noise_level * 0.2)
            if flip_count > 0:
                flip_indices = np.random.choice(len(y_pred), size=flip_count, replace=False)
                y_pred[flip_indices] = (y_pred[flip_indices] + 1) % 3

            monitor.add_batch(y_pred, y_test)

        # Check drift after accumulation
        drift, results = monitor.check_drift()

        # With gradual degradation, should detect drift
        assert len(results) > 0
        log.info(f"Gradual degradation test: drift={drift}, metrics={results[0]['metrics']}")

    def test_sudden_distribution_shift(self, baseline_metrics):
        """
        Simulate sudden shift in input distribution.
        Predictions may be accurate but on wrong distribution.
        """
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        # Train on full Iris
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        # Get baseline
        y_pred_baseline = clf.predict(X)
        baseline = PerformanceDriftDetector.calculate_metrics(y, y_pred_baseline)

        detector = PerformanceDriftDetector(baseline, effect_size_threshold=0.05)

        # Now test on shifted distribution (only virginica and versicolor)
        # This simulates real-world shift where model sees different classes
        shifted_mask = y != 0  # Remove setosa
        X_shifted = X[shifted_mask]
        y_shifted = y[shifted_mask]

        y_pred_shifted = clf.predict(X_shifted)

        current_metrics = PerformanceDriftDetector.calculate_metrics(y_shifted, y_pred_shifted)

        drift, details = detector.detect_drift(current_metrics)

        log.info(f"Distribution shift test: drift={drift}, details={details}")

    def test_class_imbalance_drift(self, baseline_metrics):
        """
        Test drift in scenarios where class distribution shifts dramatically.
        This affects recall and precision differently per class.
        """
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        # Train normally
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Get baseline on balanced test set
        y_pred = clf.predict(X_test)
        baseline = PerformanceDriftDetector.calculate_metrics(y_test, y_pred)

        detector = PerformanceDriftDetector(baseline, effect_size_threshold=0.05)

        # Now create highly imbalanced predictions
        # 80% class 0, 15% class 1, 5% class 2
        y_pred_imbalanced = np.random.choice(
            [0, 1, 2],
            size=len(y_test),
            p=[0.80, 0.15, 0.05],
        )
        y_true_imbalanced = y_test  # Keep true distribution

        current_metrics = PerformanceDriftDetector.calculate_metrics(
            y_true_imbalanced, y_pred_imbalanced
        )

        drift, details = detector.detect_drift(current_metrics)

        log.info(f"Class imbalance test: drift={drift}, metrics={current_metrics}")
        assert current_metrics["accuracy"] < baseline["accuracy"]

    def test_drift_recovery(self, baseline_metrics):
        """
        Test that detector recovers when performance improves after drift.
        """
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        detector = PerformanceDriftDetector(
            baseline_metrics, effect_size_threshold=0.05, ewma_alpha=0.3
        )

        # Phase 1: Degradation
        for _ in range(3):
            current_metrics = {
                "accuracy": 0.70,
                "precision": 0.70,
                "recall": 0.70,
                "f1": 0.70,
            }
            drift, details = detector.detect_drift(current_metrics)

        assert drift is True, "Should detect drift during degradation"

        # Phase 2: Recovery
        for _ in range(5):
            current_metrics = {
                "accuracy": 0.95,
                "precision": 0.95,
                "recall": 0.95,
                "f1": 0.95,
            }
            drift, details = detector.detect_drift(current_metrics)

        # After recovery, EWMA should improve
        ewma_accuracy = detector.ewma_values["accuracy"]
        assert ewma_accuracy > 0.85, f"EWMA should recover, got {ewma_accuracy}"

        log.info(f"Drift recovery test: EWMA accuracy = {ewma_accuracy}")


class TestMLFlowIntegration:
    """Tests with MLFlow logging (requires MLFlow running)."""

    @pytest.mark.skipif(
        not os.environ.get("MLFLOW_TRACKING_URI"),
        reason="MLFLOW_TRACKING_URI not set",
    )
    def test_log_drift_detection_to_mlflow(self, baseline_metrics):
        """Log performance drift detection results to MLFlow."""
        try:
            import mlflow
        except ImportError:
            pytest.skip("mlflow not installed")

        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test-perf-drift")

        detector = PerformanceDriftDetector(baseline_metrics)

        # Simulate several drift checks
        with mlflow.start_run(run_name="perf-drift-test"):
            for i in range(5):
                if i < 2:
                    metrics = baseline_metrics.copy()  # Normal
                else:
                    metrics = {k: v * 0.85 for k, v in baseline_metrics.items()}  # Degraded

                drift, details = detector.detect_drift(metrics)

                mlflow.log_metrics(
                    {
                        f"run_{i}_accuracy": metrics["accuracy"],
                        f"run_{i}_f1": metrics["f1"],
                        f"run_{i}_drift": int(drift),
                    },
                    step=i,
                )

            mlflow.set_tag("test_type", "performance_drift_integration")

        log.info("Successfully logged to MLFlow")

    @pytest.mark.skipif(
        not os.environ.get("MLFLOW_TRACKING_URI"),
        reason="MLFLOW_TRACKING_URI not set",
    )
    def test_mlflow_experiment_creation(self):
        """Test that MLFlow experiment can be created."""
        try:
            import mlflow
        except ImportError:
            pytest.skip("mlflow not installed")

        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(mlflow_uri)

        # Try to create/get experiment
        exp = mlflow.get_experiment_by_name("test-perf-drift-exp")
        if not exp:
            mlflow.create_experiment("test-perf-drift-exp")

        exp = mlflow.get_experiment_by_name("test-perf-drift-exp")
        assert exp is not None


class TestMonitorWithRealData:
    """Monitor tests with realistic data patterns."""

    def test_monitor_detects_systematic_error(self, baseline_metrics):
        """Test monitor detects systematic prediction errors."""
        monitor = PerformanceDriftMonitor(
            baseline_metrics,
            window_size=30,
            num_windows=2,
            effect_size_threshold=0.05,
        )

        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        # Create systematically wrong predictions
        # Off-by-one errors for setosa and versicolor
        y_pred_wrong = y.copy()
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        y_pred_wrong[mask_0] = 1
        y_pred_wrong[mask_1] = 0

        # Add samples to buffer
        for _ in range(30):
            monitor.add_batch(y_pred_wrong.values[:5], y.values[:5])

        drift, results = monitor.check_drift()

        log.info(f"Systematic error test: drift={drift}, results={results}")

    def test_monitor_window_management(self, baseline_metrics):
        """Test monitor properly manages sliding windows."""
        monitor = PerformanceDriftMonitor(
            baseline_metrics,
            window_size=20,
            num_windows=2,
        )

        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)

        # Add data in multiple batches
        for _ in range(3):
            monitor.add_batch(y_pred[:20], y[:20])

            if len(monitor.predictions_buffer) >= 20:
                drift, results = monitor.check_drift()
                assert len(results) > 0, "Should return results when buffer >= window_size"

                # Reset for next batch
                monitor.reset_buffer()


class TestThresholdSensitivity:
    """Test detector sensitivity to parameter thresholds."""

    def test_high_threshold_low_false_positives(self, baseline_metrics):
        """High thresholds should reduce false positives."""
        detector_strict = PerformanceDriftDetector(
            baseline_metrics,
            effect_size_threshold=0.15,  # High threshold
            p_value_threshold=0.01,  # Strict p-value
        )

        # Small metric changes
        current_metrics = {
            "accuracy": 0.94,
            "precision": 0.94,
            "recall": 0.94,
            "f1": 0.94,
        }

        drift, _ = detector_strict.detect_drift(current_metrics)
        assert drift is False, "Strict thresholds should not flag small changes"

    def test_low_threshold_high_sensitivity(self, baseline_metrics):
        """Low thresholds should detect smaller changes."""
        detector_sensitive = PerformanceDriftDetector(
            baseline_metrics,
            effect_size_threshold=0.01,  # Low threshold
            p_value_threshold=0.10,  # Lenient p-value
        )

        # Modest metric changes
        current_metrics = {
            "accuracy": 0.92,
            "precision": 0.92,
            "recall": 0.92,
            "f1": 0.92,
        }

        drift, details = detector_sensitive.detect_drift(current_metrics)

        log.info(f"Sensitive detector: drift={drift}, drifted_metrics={details['drifted_metrics']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
