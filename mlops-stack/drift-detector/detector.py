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
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    print("sklearn OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR sklearn: {e}", flush=True)
    sys.exit(1)

try:
    import retrain_trigger
    print("retrain_trigger OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR retrain_trigger: {e}", flush=True)
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
CHI2_THR     = float(os.environ.get("CHI2_P_VALUE_THRESHOLD",  "0.05"))
CONSEC_NEED  = int(os.environ.get("CONSECUTIVE_DRIFT_WINDOWS", "2"))
WIN_SIZE     = int(os.environ.get("WINDOW_SIZE",               "30"))
INTERVAL_S   = int(os.environ.get("CHECK_INTERVAL_S",          "30"))
MODEL_NAME   = os.environ.get("MODEL_NAME", "iris-classifier")

FEATURES = [
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)",
]


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


def build_reference():
    iris = load_iris(as_frame=True)
    X    = iris.data
    ref  = {f: X[f].values for f in FEATURES}
    counts = np.bincount(iris.target.values, minlength=3).astype(float)
    ref["class_dist"] = counts / counts.sum()
    log.info(f"Reference: {len(X)} samples, class_dist={ref['class_dist'].round(3)}")
    return ref


class ProductionBuffer:
    def __init__(self):
        self.rows  = []
        self.preds = []
        self.cycle = 0

    def fill(self):
        import random
        iris = load_iris(as_frame=True)
        X    = iris.data
        self.rows  = []
        self.preds = []
        for _ in range(WIN_SIZE):
            row = X.sample(1).values[0].copy()
            if self.cycle >= 3:
                mag    = (self.cycle - 2) * 0.9
                row[2] += mag
                row[3] += mag
                label  = 2 if random.random() < 0.85 else random.randint(0, 1)
            else:
                label = random.randint(0, 2)
            self.rows.append(row)
            self.preds.append(label)

    def df(self):
        return pd.DataFrame(self.rows, columns=FEATURES)

    def predictions(self):
        return np.array(self.preds)


def run_cycle(cycle, ref, consec):
    log.info(f"--- Cycle {cycle}  (drift_phase={cycle >= 3}) ---")

    buf = ProductionBuffer()
    buf.cycle = cycle
    buf.fill()

    df    = buf.df()
    preds = buf.predictions()

    # KS test por feature
    data_drifted = False
    drifted_feats = []
    for feat in FEATURES:
        stat, pval = stats.ks_2samp(ref[feat], df[feat].values)
        drifted    = pval < KS_THR
        if drifted:
            data_drifted = True
            drifted_feats.append(feat)
        log.info(f"  KS [{feat}]: stat={stat:.4f} pval={pval:.4f} drift={drifted}")

    # Chi2 sobre predicciones
    observed = np.bincount(preds, minlength=3).astype(float)
    expected = np.maximum(ref["class_dist"] * len(preds), 1e-6)
    chi2_stat, chi2_pval = stats.chisquare(observed, f_exp=expected)
    concept_drifted = chi2_pval < CHI2_THR
    log.info(f"  Chi2: stat={chi2_stat:.4f} pval={chi2_pval:.4f} drift={concept_drifted}")
    log.info(f"  Pred dist: obs={( observed/observed.sum() ).round(3).tolist()}")

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
            for feat in FEATURES:
                s = feat.replace(" ","_").replace("(","").replace(")","").replace("/","_")
                stat_v, pval_v = stats.ks_2samp(ref[feat], df[feat].values)
                mlflow.log_metric(f"ks_stat_{s}", float(stat_v))
                mlflow.log_metric(f"ks_pval_{s}", float(pval_v))
            mlflow.log_metric("chi2_pval",      float(chi2_pval))
            mlflow.log_metric("data_drift",     int(data_drifted))
            mlflow.log_metric("concept_drift",  int(concept_drifted))
            mlflow.log_metric("consec_drifts",  consec)
            mlflow.log_params({
                "cycle": cycle, "window_size": WIN_SIZE,
                "ks_thr": KS_THR, "chi2_thr": CHI2_THR,
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


def main():
    log.info("Drift Detector ready")
    log.info(f"  MLFLOW_URI={MLFLOW_URI}")
    log.info(f"  WIN_SIZE={WIN_SIZE}  INTERVAL={INTERVAL_S}s  CONSEC_NEED={CONSEC_NEED}")

    wait_for(f"{MLFLOW_URI}/", "MLFlow")
    wait_for(f"{INFER_URL}/health", "InferenceAPI", retries=10, delay=3)

    ref    = build_reference()
    consec = 0
    cycle  = 0

    while True:
        try:
            consec = run_cycle(cycle, ref, consec)
        except Exception:
            log.error(f"Unhandled error in cycle {cycle}:\n{traceback.format_exc()}")
        cycle += 1
        log.info(f"  Next cycle in {INTERVAL_S}s ...")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
