"""
test_performance_drift.py
──────────────────────────
Unit tests for PerformanceDriftDetector with regression metrics (R², RMSE, MAE, sMAPE, Max Error).
"""

import pytest
import numpy as np

from performance_drift_detector import PerformanceDriftDetector, PerformanceDriftMonitor


@pytest.fixture
def baseline_metrics():
    return {"r2": 0.92, "rmse": 10.5, "mae": 7.8, "smape": 4.5, "max_error": 28.0}


@pytest.fixture
def stable_predictions():
    """Continuous regression predictions with small noise → high R²."""
    np.random.seed(42)
    y_true = np.random.uniform(50, 200, 100)
    y_pred = y_true + np.random.randn(100) * 3.0
    return y_pred, y_true


# ── Basics ─────────────────────────────────────────────────────────────────────

class TestPerformanceDriftDetectorBasics:

    def test_calculate_metrics_keys(self, stable_predictions):
        y_pred, y_true = stable_predictions
        metrics = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {"r2", "rmse", "mae", "smape", "max_error"}

    def test_calculate_metrics_ranges(self, stable_predictions):
        y_pred, y_true = stable_predictions
        metrics = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)
        assert metrics["r2"]        <= 1.0
        assert metrics["rmse"]      >= 0.0
        assert metrics["mae"]       >= 0.0
        assert 0.0 <= metrics["smape"] <= 200.0
        assert metrics["max_error"] >= metrics["mae"]   # max >= mean always

    def test_detector_initialization(self, baseline_metrics):
        det = PerformanceDriftDetector(baseline_metrics)
        assert det.baseline_metrics == baseline_metrics
        assert det.p_value_threshold == 0.05

    def test_no_drift_on_stable_data(self, baseline_metrics):
        det     = PerformanceDriftDetector(baseline_metrics)
        current = {"r2": 0.91, "rmse": 11.0, "mae": 8.0}  # small changes
        drift, details = det.detect_drift(current)
        assert drift is False, f"False positive: {details['drifted_metrics']}"


# ── Drift Detection Logic ──────────────────────────────────────────────────────

class TestDriftDetectionLogic:

    def test_drift_on_large_r2_drop(self, baseline_metrics):
        det     = PerformanceDriftDetector(baseline_metrics, effect_size_threshold=0.05)
        current = {"r2": 0.70, "rmse": 25.0, "mae": 18.0}  # large degradation
        drift, details = det.detect_drift(current)
        assert drift is True, "Should detect drift on large R² drop + RMSE increase"

    def test_no_drift_on_small_change(self, baseline_metrics):
        det     = PerformanceDriftDetector(baseline_metrics, effect_size_threshold=0.05)
        current = {"r2": 0.91, "rmse": 10.8, "mae": 7.9}
        drift, _ = det.detect_drift(current)
        assert drift is False

    def test_t_test_sustained_degradation(self, baseline_metrics):
        det = PerformanceDriftDetector(baseline_metrics, effect_size_threshold=0.10)
        for _ in range(6):
            current = {"r2": 0.85, "rmse": 16.0, "mae": 12.0}
            drift, _ = det.detect_drift(current)
        assert drift is True, "t-test should detect sustained degradation"

    def test_ewma_trend_detection(self, baseline_metrics):
        det = PerformanceDriftDetector(baseline_metrics, effect_size_threshold=0.10,
                                       ewma_alpha=0.3)
        for i in range(8):
            current = {
                "r2":   0.92 - i * 0.015,
                "rmse": 10.5 + i * 1.5,
                "mae":  7.8  + i * 1.0,
            }
            drift, _ = det.detect_drift(current)
        assert drift is True, "EWMA should detect sustained degradation trend"

    def test_requires_two_drifted_metrics(self, baseline_metrics):
        det = PerformanceDriftDetector(baseline_metrics, effect_size_threshold=0.05)
        # Only R² drifts
        current = {"r2": 0.80, "rmse": 10.6, "mae": 7.9}
        drift, details = det.detect_drift(current)
        assert details["drifted_count"] == 1
        assert drift is False

    def test_multiple_metrics_drift(self, baseline_metrics):
        det = PerformanceDriftDetector(baseline_metrics, effect_size_threshold=0.05)
        current = {"r2": 0.75, "rmse": 22.0, "mae": 16.0}
        drift, details = det.detect_drift(current)
        assert details["drifted_count"] >= 2
        assert drift is True


# ── Monitor ────────────────────────────────────────────────────────────────────

class TestPerformanceDriftMonitor:

    def test_monitor_initialization(self, baseline_metrics):
        mon = PerformanceDriftMonitor(baseline_metrics, window_size=30, num_windows=3)
        assert mon.window_size == 30
        assert len(mon.detectors) == 3

    def test_buffer_accumulation(self, baseline_metrics):
        mon    = PerformanceDriftMonitor(baseline_metrics, window_size=10, num_windows=1)
        y_pred = np.array([100.0, 110.0, 120.0])
        y_true = np.array([102.0, 108.0, 119.0])
        mon.add_batch(y_pred, y_true)
        assert len(mon.predictions_buffer) == 3

    def test_insufficient_data_returns_no_drift(self, baseline_metrics):
        mon    = PerformanceDriftMonitor(baseline_metrics, window_size=100, num_windows=1)
        y_pred = np.random.uniform(50, 200, 20)
        y_true = y_pred + np.random.randn(20) * 5
        mon.add_batch(y_pred, y_true)
        drift, results = mon.check_drift()
        assert drift is False
        assert len(results) == 0

    def test_check_drift_with_sufficient_data(self, baseline_metrics):
        mon    = PerformanceDriftMonitor(baseline_metrics, window_size=30, num_windows=1)
        np.random.seed(0)
        y_true = np.random.uniform(50, 200, 30)
        y_pred = y_true + np.random.randn(30) * 3.0  # stable predictions
        mon.add_batch(y_pred, y_true)
        drift, results = mon.check_drift()
        assert isinstance(drift, bool)
        assert len(results) >= 1
        assert "metrics" in results[0]

    def test_reset_buffer(self, baseline_metrics):
        mon    = PerformanceDriftMonitor(baseline_metrics, window_size=10, num_windows=1)
        y_pred = np.random.uniform(50, 200, 10)
        y_true = y_pred + np.random.randn(10) * 3
        mon.add_batch(y_pred, y_true)
        mon.reset_buffer()
        assert len(mon.predictions_buffer) == 0
        assert len(mon.labels_buffer) == 0


# ── Edge Cases ─────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_constant_predictions(self, baseline_metrics):
        det    = PerformanceDriftDetector(baseline_metrics)
        y_true = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        y_pred = np.full_like(y_true, 120.0)  # all same prediction
        metrics = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)
        assert "r2" in metrics  # should not crash

    def test_perfect_predictions(self, baseline_metrics):
        y_true  = np.random.uniform(50, 200, 50)
        y_pred  = y_true.copy()
        metrics = PerformanceDriftDetector.calculate_metrics(y_true, y_pred)
        assert metrics["r2"]        == pytest.approx(1.0, abs=1e-6)
        assert metrics["rmse"]      == pytest.approx(0.0, abs=1e-6)
        assert metrics["mae"]       == pytest.approx(0.0, abs=1e-6)
        assert metrics["smape"]     == pytest.approx(0.0, abs=1e-6)
        assert metrics["max_error"] == pytest.approx(0.0, abs=1e-6)


# ── History & EWMA ─────────────────────────────────────────────────────────────

class TestHistoryTracking:

    def test_history_accumulation(self, baseline_metrics):
        det = PerformanceDriftDetector(baseline_metrics)
        for i in range(5):
            det.detect_drift({"r2": 0.92 - i * 0.01, "rmse": 10.5, "mae": 7.8})
        assert len(det.get_history("r2")) == 5

    def test_ewma_smoothing(self, baseline_metrics):
        det = PerformanceDriftDetector(baseline_metrics, ewma_alpha=0.3)
        det.detect_drift({"r2": 0.60, "rmse": 30.0, "mae": 22.0})  # big drop
        ewma_after_drop = det.ewma_values["r2"]

        det.detect_drift({"r2": 0.92, "rmse": 10.5, "mae": 7.8})   # recovery
        ewma_after_recover = det.ewma_values["r2"]

        assert ewma_after_recover > ewma_after_drop
        assert ewma_after_recover < 0.92  # not fully recovered in 1 step
