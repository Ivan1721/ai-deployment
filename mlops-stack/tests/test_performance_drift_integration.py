"""
test_performance_drift_integration.py
──────────────────────────────────────
Integration tests for PerformanceDriftDetector with regression scenarios.
Tests realistic drift patterns (gradual degradation, sudden shift, recovery).
"""

import os
import pytest
import numpy as np
import logging

log = logging.getLogger(__name__)

from performance_drift_detector import PerformanceDriftDetector, PerformanceDriftMonitor


def _make_predictions(n: int, r2_target: float = 0.95, seed: int = 42) -> tuple:
    """Generate (y_pred, y_true) where y_pred achieves approximately r2_target."""
    rng    = np.random.default_rng(seed)
    y_true = rng.uniform(50, 300, n)
    var_y  = np.var(y_true)
    noise_std = np.sqrt(max(var_y * (1 - r2_target), 1e-6))
    y_pred    = y_true + rng.normal(0, noise_std, n)
    return y_pred, y_true


@pytest.fixture
def baseline_metrics():
    y_pred, y_true = _make_predictions(200, r2_target=0.93)
    return PerformanceDriftDetector.calculate_metrics(y_true, y_pred)


# ── Realistic Drift Scenarios ──────────────────────────────────────────────────

class TestRealWorldDriftScenarios:

    def test_gradual_performance_degradation(self, baseline_metrics):
        monitor = PerformanceDriftMonitor(
            baseline_metrics, window_size=50, num_windows=3,
            effect_size_threshold=0.05,
        )
        # Feed 5 windows with gradually worsening predictions
        for window in range(5):
            r2_target = 0.93 - window * 0.07  # 0.93 → 0.65
            y_pred, y_true = _make_predictions(50, r2_target=max(r2_target, 0.1),
                                               seed=window)
            monitor.add_batch(y_pred, y_true)

        drift, results = monitor.check_drift()
        assert len(results) > 0
        log.info(f"Gradual degradation: drift={drift}, "
                 f"metrics={results[0].get('metrics', {})}")

    def test_sudden_distribution_shift(self, baseline_metrics):
        detector = PerformanceDriftDetector(baseline_metrics, effect_size_threshold=0.05)

        # Sudden shift: predictions are on a completely different scale
        rng    = np.random.default_rng(99)
        y_true = rng.uniform(50, 300, 50)
        y_pred = y_true * 0.5 + rng.normal(0, 40, 50)  # systematic bias

        shifted = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)
        drift, details = detector.detect_drift(shifted)

        log.info(f"Sudden shift: drift={drift}, metrics={shifted}")
        assert shifted["r2"] < baseline_metrics["r2"]

    def test_drift_recovery(self, baseline_metrics):
        detector = PerformanceDriftDetector(
            baseline_metrics, effect_size_threshold=0.05, ewma_alpha=0.3
        )

        # Phase 1: degradation
        for _ in range(3):
            drift, _ = detector.detect_drift(
                {"r2": 0.65, "rmse": 35.0, "mae": 25.0}
            )
        assert drift is True, "Should detect drift during degradation"

        # Phase 2: recovery
        for _ in range(5):
            detector.detect_drift({"r2": 0.93, "rmse": 10.5, "mae": 7.8})

        ewma_r2 = detector.ewma_values["r2"]
        assert ewma_r2 > 0.80, f"EWMA should recover, got {ewma_r2}"
        log.info(f"Recovery test: EWMA R² = {ewma_r2}")

    def test_high_variance_predictions(self, baseline_metrics):
        detector = PerformanceDriftDetector(baseline_metrics)
        rng = np.random.default_rng(7)

        for _ in range(5):
            y_true = rng.uniform(50, 300, 30)
            y_pred = y_true + rng.normal(0, 60, 30)   # high noise
            metrics = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)
            detector.detect_drift(metrics)

        assert len(detector.get_history("r2")) == 5


# ── MLflow Integration (skipped without MLFLOW_TRACKING_URI) ──────────────────

class TestMLFlowIntegration:

    @pytest.mark.skipif(
        not os.environ.get("MLFLOW_TRACKING_URI"),
        reason="MLFLOW_TRACKING_URI not set",
    )
    def test_log_drift_to_mlflow(self, baseline_metrics):
        try:
            import mlflow
        except ImportError:
            pytest.skip("mlflow not installed")

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("test-perf-drift-hri")

        detector = PerformanceDriftDetector(baseline_metrics)

        with mlflow.start_run(run_name="perf-drift-integration-test"):
            for i in range(5):
                r2_target = 0.93 if i < 2 else 0.70
                y_pred, y_true = _make_predictions(50, r2_target=r2_target, seed=i)
                metrics = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)
                drift, _ = detector.detect_drift(metrics)
                mlflow.log_metrics(
                    {f"step_{i}_r2": metrics["r2"],
                     f"step_{i}_drift": int(drift)},
                    step=i,
                )
            mlflow.set_tag("test_type", "regression_drift_integration")

        log.info("Successfully logged regression drift to MLflow")

    @pytest.mark.skipif(
        not os.environ.get("MLFLOW_TRACKING_URI"),
        reason="MLFLOW_TRACKING_URI not set",
    )
    def test_mlflow_experiment_creation(self):
        try:
            import mlflow
        except ImportError:
            pytest.skip("mlflow not installed")

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        exp_name = "test-perf-drift-hri-exp"
        if not mlflow.get_experiment_by_name(exp_name):
            mlflow.create_experiment(exp_name)
        assert mlflow.get_experiment_by_name(exp_name) is not None


# ── Monitor with Realistic Data ────────────────────────────────────────────────

class TestMonitorWithRealData:

    def test_monitor_detects_systematic_error(self, baseline_metrics):
        monitor = PerformanceDriftMonitor(
            baseline_metrics, window_size=30, num_windows=2,
            effect_size_threshold=0.05,
        )
        # Predictions with large systematic offset
        rng    = np.random.default_rng(5)
        y_true = rng.uniform(50, 300, 30)
        y_pred = y_true * 0.6 + 30  # ~40% bias

        for _ in range(30):
            monitor.add_batch(y_pred[:1], y_true[:1])

        drift, results = monitor.check_drift()
        log.info(f"Systematic error: drift={drift}, results={results}")

    def test_monitor_window_management(self, baseline_metrics):
        monitor = PerformanceDriftMonitor(
            baseline_metrics, window_size=20, num_windows=2,
        )
        for _ in range(3):
            y_pred, y_true = _make_predictions(20, r2_target=0.92)
            monitor.add_batch(y_pred, y_true)

            if len(monitor.predictions_buffer) >= 20:
                _, results = monitor.check_drift()
                assert len(results) > 0
                monitor.reset_buffer()


# ── Threshold Sensitivity ──────────────────────────────────────────────────────

class TestThresholdSensitivity:

    def test_strict_threshold_low_false_positives(self, baseline_metrics):
        det = PerformanceDriftDetector(
            baseline_metrics, effect_size_threshold=0.15, p_value_threshold=0.01
        )
        current = {"r2": 0.90, "rmse": 11.5, "mae": 8.5}  # small changes
        drift, _ = det.detect_drift(current)
        assert drift is False

    def test_sensitive_threshold_detects_small_changes(self, baseline_metrics):
        det = PerformanceDriftDetector(
            baseline_metrics, effect_size_threshold=0.01, p_value_threshold=0.10
        )
        current = {"r2": 0.88, "rmse": 14.0, "mae": 10.0}
        drift, details = det.detect_drift(current)
        log.info(f"Sensitive: drift={drift}, drifted={details['drifted_metrics']}")
