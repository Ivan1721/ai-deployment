"""
detector.py  -  Drift Detection Service
"""

import os
import sys
import time
import logging
import traceback

# ── test de imports al inicio ──────────────────────────────────────────────
print("=== DRIFT DETECTOR STARTING ===", flush=True)
print(f"Python: {sys.version}", flush=True)

try:
    import numpy as np
    print(f"numpy {np.__version__} OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR numpy: {e}", flush=True)
    sys.exit(1)

try:
    import pandas as pd
    print(f"pandas {pd.__version__} OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR pandas: {e}", flush=True)
    sys.exit(1)

try:
    from scipy import stats
    print("scipy OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR scipy: {e}", flush=True)
    sys.exit(1)

try:
    import mlflow
    from mlflow import MlflowClient
    print(f"mlflow {mlflow.__version__} OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR mlflow: {e}", flush=True)
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    print("sklearn OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR sklearn: {e}", flush=True)
    sys.exit(1)

try:
    from mlops_common import load_config
    from mlops_common.data_sources import get_data_source
    from mlops_common.models import get_model_strategy
    from mlops_common.problem_types import get_problem_type
    print("mlops_common OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR mlops_common: {e}", flush=True)
    sys.exit(1)

try:
    import retrain_trigger
    print("retrain_trigger OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR retrain_trigger: {e}", flush=True)
    sys.exit(1)

try:
    from performance_drift_detector import PerformanceDriftDetector, PerformanceDriftMonitor
    print("performance_drift_detector OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR performance_drift_detector: {e}", flush=True)
    sys.exit(1)

print("=== ALL IMPORTS OK ===", flush=True)

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DRIFT] %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI",      "http://mlflow:5001")
INFER_URL    = os.environ.get("INFERENCE_API_URL",         "http://inference-api:8000")
KS_THR       = float(os.environ.get("KS_P_VALUE_THRESHOLD",   "0.05"))
CONCEPT_THR  = float(os.environ.get("CONCEPT_DRIFT_P_VALUE_THRESHOLD",  "0.05"))
CONSEC_NEED  = int(os.environ.get("CONSECUTIVE_DRIFT_WINDOWS", "2"))
WIN_SIZE     = int(os.environ.get("WINDOW_SIZE",               "30"))
INTERVAL_S   = int(os.environ.get("CHECK_INTERVAL_S",          "30"))

# Performance Drift Detection Config
ENABLE_PERF_DRIFT   = os.environ.get("ENABLE_PERFORMANCE_DRIFT", "true").lower() == "true"
PERF_DRIFT_THR      = float(os.environ.get("PERF_DRIFT_EFFECT_SIZE", "0.05"))
PERF_DRIFT_P_VALUE  = float(os.environ.get("PERF_DRIFT_P_VALUE", "0.05"))
PERF_CONSEC_NEED    = int(os.environ.get("PERF_DRIFT_CONSECUTIVE", "2"))


def wait_for(url, label, retries=30, delay=4):
    import urllib.request
    log.info(f"Waiting for {label} at {url} ...")
    for i in range(retries):
        try:
            urllib.request.urlopen(url, timeout=3)
            log.info(f"{label} ready ✓")
            return True
        except Exception as e:
            log.info(f"  {label} not ready ({i+1}/{retries}): {e}")
            time.sleep(delay)
    log.error(f"{label} never became ready — continuing anyway")
    return False


def build_reference(bundle, cfg):
    """Distribución de referencia: features (para KS) + salida del modelo (para concept drift)."""
    ref = {f: bundle.X[f].values for f in bundle.feature_names}

    if bundle.target_names is not None:
        n_classes = len(bundle.target_names)
        counts = np.bincount(bundle.y.values, minlength=n_classes).astype(float)
        ref["class_dist"] = counts / counts.sum()
        log.info(f"Reference: {len(bundle.X)} samples, class_dist={ref['class_dist'].round(3)}")
    else:
        ref["y_values"] = bundle.y.values
        log.info(f"Reference: {len(bundle.X)} samples, y_mean={ref['y_values'].mean():.3f}")

    return ref


class ProductionBuffer:
    """
    Simulador de tráfico de producción. A partir del ciclo 3 desplaza las
    últimas 2 columnas numéricas (en general las más sensibles al drift) y
    sesga las predicciones simuladas, para forzar la detección de drift.
    """

    def __init__(self, bundle, problem_type: str):
        self.bundle = bundle
        self.problem_type = problem_type
        self.rows  = []
        self.preds = []
        self.cycle = 0

    def fill(self):
        import random

        X = self.bundle.X
        feature_names = self.bundle.feature_names
        shift_cols = range(max(0, len(feature_names) - 2), len(feature_names))

        if self.problem_type == "classification":
            n_classes = len(self.bundle.target_names)
        else:
            y_values = self.bundle.y.values

        self.rows  = []
        self.preds = []
        for _ in range(WIN_SIZE):
            row = X.sample(1).values[0].copy()
            drift_phase = self.cycle >= 3
            if drift_phase:
                mag = (self.cycle - 2) * 0.9
                for c in shift_cols:
                    row[c] += mag

            if self.problem_type == "classification":
                last_class = n_classes - 1
                if drift_phase:
                    label = last_class if random.random() < 0.85 else random.randint(0, n_classes - 1)
                else:
                    label = random.randint(0, n_classes - 1)
            else:
                base = float(random.choice(y_values))
                label = base * (1 + mag * 0.1) if drift_phase else base

            self.rows.append(row)
            self.preds.append(label)

    def df(self):
        return pd.DataFrame(self.rows, columns=self.bundle.feature_names)

    def predictions(self):
        if self.problem_type == "classification":
            return np.array(self.preds, dtype=int)
        return np.array(self.preds, dtype=float)


def run_cycle(cycle, ref, consec, bundle, problem, cfg):
    log.info(f"--- Cycle {cycle}  (drift_phase={cycle >= 3}) ---")

    buf = ProductionBuffer(bundle, cfg.problem_type)
    buf.cycle = cycle
    buf.fill()

    df    = buf.df()
    preds = buf.predictions()

    # KS test por feature (data drift)
    data_drifted = False
    drifted_feats = []
    for feat in bundle.feature_names:
        stat, pval = stats.ks_2samp(ref[feat], df[feat].values)
        drifted = pval < KS_THR
        if drifted:
            data_drifted = True
            drifted_feats.append(feat)
        log.info(f"  KS [{feat}]: stat={stat:.4f} pval={pval:.4f} drift={drifted}")

    # Concept drift: delegado al problem_type (chi2 para clasificación, KS para regresión)
    if bundle.target_names is not None:
        n_classes = len(bundle.target_names)
        observed = np.bincount(preds, minlength=n_classes).astype(float)
        concept_drifted, concept_details = problem.concept_drift(ref["class_dist"], observed, CONCEPT_THR)
        log.info(f"  Pred dist: obs={(observed / observed.sum()).round(3).tolist()}")
    else:
        concept_drifted, concept_details = problem.concept_drift(ref["y_values"], preds, CONCEPT_THR)

    log.info(
        f"  Concept drift [{concept_details['method']}]: "
        f"stat={concept_details['statistic']:.4f} pval={concept_details['p_value']:.4f} "
        f"drift={concept_drifted}"
    )

    any_drift = data_drifted or concept_drifted
    if any_drift:
        consec += 1
    else:
        consec  = 0

    status = "DRIFT ⚠" if any_drift else "OK ✓"
    log.info(f"  Status: {status}  consecutive={consec}/{CONSEC_NEED}")

    # Registrar en MLFlow
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("drift-monitoring")
        with mlflow.start_run(run_name=f"window-{cycle:03d}"):
            for feat in bundle.feature_names:
                s = feat.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                stat_v, pval_v = stats.ks_2samp(ref[feat], df[feat].values)
                mlflow.log_metric(f"ks_stat_{s}", float(stat_v))
                mlflow.log_metric(f"ks_pval_{s}", float(pval_v))
            mlflow.log_metric("concept_drift_pval", concept_details["p_value"])
            mlflow.log_metric("data_drift",     int(data_drifted))
            mlflow.log_metric("concept_drift",  int(concept_drifted))
            mlflow.log_metric("consec_drifts",  consec)
            mlflow.log_params({
                "cycle": cycle, "window_size": WIN_SIZE,
                "ks_thr": KS_THR, "concept_drift_thr": CONCEPT_THR,
            })
        log.info("  MLFlow run logged ✓")
    except Exception:
        log.error(f"  MLFlow logging failed:\n{traceback.format_exc()}")

    # Reentrenar si se cumple el umbral
    if consec >= CONSEC_NEED:
        log.warning(f"  RETRAINING TRIGGERED (consecutive={consec})")
        try:
            retrain_trigger.trigger(
                reason="data_drift" if data_drifted else "concept_drift",
                consecutive_windows=consec,
                mlflow_uri=MLFLOW_URI,
            )
            consec = 0
        except Exception:
            log.error(f"  Retrain failed:\n{traceback.format_exc()}")

    return consec


def build_baseline_metrics(bundle, strategy, problem, cfg):
    """Entrena con los mismos hiperparámetros/split que el trainer para obtener métricas baseline."""
    split_kwargs = {"test_size": cfg.training.test_size, "random_state": 42}
    if cfg.training.stratify:
        split_kwargs["stratify"] = bundle.y
    X_train, X_test, y_train, y_test = train_test_split(bundle.X, bundle.y, **split_kwargs)

    model = strategy.build(cfg.model.params)
    model = strategy.fit(model, X_train, y_train)
    y_pred = strategy.predict(model, X_test)
    y_proba = strategy.predict_proba(model, X_test) if strategy.supports_proba else None

    return problem.compute_metrics(y_test, y_pred, y_proba)


def run_performance_drift_cycle(cycle, perf_consec, baseline_metrics, bundle, problem, cfg):
    """
    Check performance metrics for drift.
    Uses ProductionBuffer to simulate real predictions and ground truth.
    """
    log.info(f"--- Performance Drift Cycle {cycle} ---")

    if not ENABLE_PERF_DRIFT:
        return perf_consec

    try:
        detector = PerformanceDriftDetector(
            baseline_metrics,
            p_value_threshold=PERF_DRIFT_P_VALUE,
            effect_size_threshold=PERF_DRIFT_THR,
        )

        # Simulate production buffer (in real scenario, this comes from inference logs)
        buf = ProductionBuffer(bundle, cfg.problem_type)
        buf.cycle = cycle
        buf.fill()

        y_pred = buf.predictions()

        if bundle.target_names is not None:
            n_classes = len(bundle.target_names)
            y_true = np.resize(np.arange(n_classes), len(y_pred))
        else:
            y_true = bundle.y.sample(n=len(y_pred), replace=True).values

        # Calculate current metrics
        current_metrics = problem.compute_metrics(y_true, y_pred)
        log.info(f"Current metrics: {current_metrics}")

        # Detect drift
        drift, details = detector.detect_drift(current_metrics)

        if drift:
            perf_consec += 1
        else:
            perf_consec = 0

        status = "DRIFT ⚠" if drift else "OK ✓"
        log.info(f"  Performance Status: {status}  consecutive={perf_consec}/{PERF_CONSEC_NEED}")

        # Log to MLFlow
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("performance-drift-monitoring")
            with mlflow.start_run(run_name=f"perf-window-{cycle:03d}"):
                for metric_name, value in current_metrics.items():
                    mlflow.log_metric(f"current_{metric_name}", float(value))

                if "tests" in details:
                    for metric_name, test_result in details["tests"].items():
                        mlflow.log_metric(f"baseline_{metric_name}",
                                        float(test_result["baseline"]))
                        mlflow.log_metric(f"change_pct_{metric_name}",
                                        float(test_result["change_pct"]))
                        if "p_value" in test_result:
                            mlflow.log_metric(f"ttest_pval_{metric_name}",
                                            float(test_result["p_value"]))

                mlflow.log_metric("perf_drift_detected", int(drift))
                mlflow.log_metric("perf_consec_drifts", perf_consec)
                mlflow.log_params({
                    "cycle": cycle,
                    "effect_size_thr": PERF_DRIFT_THR,
                    "p_value_thr": PERF_DRIFT_P_VALUE,
                })

            log.info("  Performance metrics logged to MLFlow ✓")
        except Exception:
            log.warning(f"  MLFlow performance logging skipped: {traceback.format_exc()}")

        # Trigger retraining if consecutive drifts threshold met
        if perf_consec >= PERF_CONSEC_NEED:
            log.warning(f"  PERFORMANCE DRIFT RETRAINING TRIGGERED (consecutive={perf_consec})")
            try:
                retrain_trigger.trigger(
                    reason="performance_drift",
                    consecutive_windows=perf_consec,
                    mlflow_uri=MLFLOW_URI,
                )
                perf_consec = 0
            except Exception:
                log.error(f"  Retrain failed: {traceback.format_exc()}")

        return perf_consec

    except Exception:
        log.error(f"  Unhandled error in perf drift cycle: {traceback.format_exc()}")
        return perf_consec


def main():
    cfg = load_config()
    bundle = get_data_source(cfg.data_source).load()
    strategy = get_model_strategy(cfg.model.family)
    problem = get_problem_type(cfg.problem_type)

    log.info("Drift Detector ready")
    log.info(f"  MLFLOW_URI={MLFLOW_URI}")
    log.info(f"  config={cfg.name}  problem_type={cfg.problem_type}  model_family={cfg.model.family}")
    log.info(f"  WIN_SIZE={WIN_SIZE}  INTERVAL={INTERVAL_S}s  CONSEC_NEED={CONSEC_NEED}")
    if ENABLE_PERF_DRIFT:
        log.info(f"  PERFORMANCE DRIFT DETECTION ENABLED")
        log.info(f"    Effect size threshold: {PERF_DRIFT_THR}")
        log.info(f"    P-value threshold: {PERF_DRIFT_P_VALUE}")
        log.info(f"    Consecutive threshold: {PERF_CONSEC_NEED}")

    wait_for(f"{MLFLOW_URI}/", "MLFlow")
    wait_for(f"{INFER_URL}/health", "InferenceAPI", retries=10, delay=3)

    ref = build_reference(bundle, cfg)
    baseline_metrics = build_baseline_metrics(bundle, strategy, problem, cfg) if ENABLE_PERF_DRIFT else None
    if baseline_metrics is not None:
        log.info(f"Baseline metrics: {baseline_metrics}")

    consec = 0
    perf_consec = 0
    cycle  = 0

    while True:
        try:
            consec = run_cycle(cycle, ref, consec, bundle, problem, cfg)

            if ENABLE_PERF_DRIFT:
                perf_consec = run_performance_drift_cycle(cycle, perf_consec, baseline_metrics, bundle, problem, cfg)
        except Exception:
            log.error(f"Unhandled error in cycle {cycle}:\n{traceback.format_exc()}")
        cycle += 1
        log.info(f"  Next cycle in {INTERVAL_S}s ...")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
