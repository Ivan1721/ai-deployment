"""
detector.py  —  Drift Detection Service (HRI Regression)

Monitors two types of drift every CHECK_INTERVAL_S seconds:
  • Data drift:    KS test on each input feature vs. training distribution.
  • Concept drift: KS test on prediction values vs. reference predictions.
  • Perf drift:    PerformanceDriftDetector on R²/RMSE/MAE (requires ground truth).

When CONSECUTIVE_DRIFT_WINDOWS consecutive windows flag drift, triggers
champion/challenger retraining via retrain_trigger.trigger().
"""

import os, sys, time, logging, traceback
print("=== DRIFT DETECTOR STARTING ===", flush=True)

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    import mlflow
    from mlflow import MlflowClient
    import retrain_trigger
    from performance_drift_detector import PerformanceDriftDetector, PerformanceDriftMonitor
    print("All imports OK", flush=True)
except Exception as e:
    print(f"IMPORT ERROR: {e}", flush=True)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DRIFT] %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI",       "http://mlflow:5001")
INFER_URL    = os.environ.get("INFERENCE_API_URL",          "http://inference-api:8000")
DATASET_PATH = os.environ.get("DATASET_PATH",               "/data/simulation_all.csv")
KS_THR       = float(os.environ.get("KS_P_VALUE_THRESHOLD",    "0.05"))
CONSEC_NEED  = int(os.environ.get("CONSECUTIVE_DRIFT_WINDOWS", "2"))
WIN_SIZE     = int(os.environ.get("WINDOW_SIZE",                "30"))
INTERVAL_S   = int(os.environ.get("CHECK_INTERVAL_S",           "30"))

ENABLE_PERF_DRIFT  = os.environ.get("ENABLE_PERFORMANCE_DRIFT", "true").lower() == "true"
PERF_DRIFT_THR     = float(os.environ.get("PERF_DRIFT_EFFECT_SIZE", "0.05"))
PERF_DRIFT_P_VALUE = float(os.environ.get("PERF_DRIFT_P_VALUE",     "0.05"))
PERF_CONSEC_NEED   = int(os.environ.get("PERF_DRIFT_CONSECUTIVE",   "2"))

SCENARIOS     = {0: "HumanOnly", 1: "WithRobot"}
FEATURE_NAMES = ["Humans", "ROW_N", "RandomPosition", "Act_Ladder", "Act_Mixed", "Act_Picker"]
TARGETS = {
    "TotalRecollected": "TotalRecollectedCrops_crop_units",
    "CargoZoneProd":    "TotalProductionCargoZone_crop_units",
    "TotalWorkload":    "TotalHumanWorkload_kcal",
    "AvgProduction":    "AverageHumanProduction_crop_units",
}

# Valid discrete values per feature (for synthetic data generation)
VALID_HUMANS    = [1, 3, 6, 8, 10, 12]
VALID_ROWS      = [1, 2, 3]
VALID_ACTIVITIES = ["harv_ground", "harv_ladder", "harv_mixed", "harv_picker"]


# ── helpers ───────────────────────────────────────────────────────────────────

def wait_for(url: str, label: str, retries: int = 30, delay: int = 4) -> bool:
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


def _load_preprocessed() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    dummies = pd.get_dummies(df["MainActivity"], prefix="Act", drop_first=True)
    dummies = dummies.rename(columns={
        "Act_harv_ladder": "Act_Ladder",
        "Act_harv_mixed":  "Act_Mixed",
        "Act_harv_picker": "Act_Picker",
    })
    for col in ["Act_Ladder", "Act_Mixed", "Act_Picker"]:
        if col not in dummies.columns:
            dummies[col] = 0
    return pd.concat([df, dummies[["Act_Ladder", "Act_Mixed", "Act_Picker"]]], axis=1)


def _predict_api(scenario: int, workers: int, crop_row: int,
                 rand_pos: int, activity: str) -> dict | None:
    import urllib.request, json as _json
    body = _json.dumps({
        "scenario": scenario, "workers": workers,
        "crop_row": crop_row, "rand_pos": rand_pos, "activity": activity,
    }).encode()
    req = urllib.request.Request(
        f"{INFER_URL}/predict", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return _json.loads(r.read())
    except Exception:
        return None


def _activity_from_row(row: pd.Series) -> str:
    if row.get("Act_Ladder", 0) > 0.5: return "harv_ladder"
    if row.get("Act_Mixed",  0) > 0.5: return "harv_mixed"
    if row.get("Act_Picker", 0) > 0.5: return "harv_picker"
    return "harv_ground"


# ── reference building ────────────────────────────────────────────────────────

def build_reference(df: pd.DataFrame) -> dict:
    """
    Builds reference distributions from the training dataset.
    Also queries the inference API for a reference prediction distribution.
    """
    ref = {feat: df[feat].values for feat in FEATURE_NAMES}

    # Build reference prediction distribution (TotalRecollected, scenario 0)
    ref_preds = []
    sample = df[df["Scenario"] == 0].sample(min(WIN_SIZE, len(df)), replace=False)
    for _, row in sample.iterrows():
        resp = _predict_api(
            scenario=0,
            workers=int(row["Humans"]),
            crop_row=int(row["ROW_N"]),
            rand_pos=int(row["RandomPosition"]),
            activity=_activity_from_row(row),
        )
        if resp:
            ref_preds.append(resp.get("total_recollected", 0.0))

    ref["pred_dist"] = np.array(ref_preds) if ref_preds else None
    log.info(f"Reference: {len(df)} rows | API reference predictions: {len(ref_preds)}")
    return ref


def build_baseline_metrics(df: pd.DataFrame) -> dict:
    """
    Returns baseline R²/RMSE/MAE from MLflow production model.
    Falls back to a conservative default if MLflow is unavailable.
    """
    try:
        import mlflow.sklearn
        from sklearn.model_selection import train_test_split
        mlflow.set_tracking_uri(MLFLOW_URI)
        model = mlflow.sklearn.load_model(
            "models:/hri-HumanOnly-TotalRecollected/Production"
        )
        df_sc = df[df["Scenario"] == 0]
        X = df_sc[FEATURE_NAMES].values
        y = df_sc["TotalRecollectedCrops_crop_units"].values
        _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        return PerformanceDriftDetector.calculate_metrics(y_te, model.predict(X_te))
    except Exception as e:
        log.warning(f"Could not load model for baseline: {e} — using defaults")
        return {"r2": 0.90, "rmse": 20.0, "mae": 15.0}


# ── production buffer ─────────────────────────────────────────────────────────

class ProductionBuffer:
    """
    Simulates a production data window by sampling from the training dataset.
    In later cycles (cycle >= 3), injects synthetic drift to test detection.
    Calls the inference API for real predictions; falls back to dataset values.
    """

    def __init__(self, df: pd.DataFrame, cycle: int = 0):
        self.df    = df
        self.cycle = cycle
        self.feat_rows: list = []
        self.pred_vals: list = []

    def fill(self):
        sample = self.df.sample(WIN_SIZE, replace=True).reset_index(drop=True)
        self.feat_rows = []
        self.pred_vals = []

        for _, row in sample.iterrows():
            feat = row[FEATURE_NAMES].values.astype(float).copy()

            # Inject drift: shift Humans and ROW_N in later cycles
            if self.cycle >= 3:
                mag      = (self.cycle - 2) * 0.5
                feat[0]  = float(np.clip(feat[0] + np.random.randn() * mag, 1, 12))
                feat[1]  = float(np.clip(feat[1] + np.random.randn() * 0.3, 1, 3))

            self.feat_rows.append(feat)

            # Try real API prediction
            resp = _predict_api(
                scenario=int(row.get("Scenario", 0)),
                workers=int(np.clip(round(feat[0]), 1, 12)),
                crop_row=int(np.clip(round(feat[1]), 1, 3)),
                rand_pos=int(round(feat[2])),
                activity=_activity_from_row(row),
            )
            pred = resp.get("total_recollected", 0.0) if resp else \
                   float(row.get("TotalRecollectedCrops_crop_units", 0.0))
            self.pred_vals.append(pred)

    def features_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.feat_rows, columns=FEATURE_NAMES)

    def predictions(self) -> np.ndarray:
        return np.array(self.pred_vals)

    def ground_truth(self, target_col: str = "TotalRecollectedCrops_crop_units") -> np.ndarray:
        sample = self.df.sample(len(self.feat_rows), replace=True)
        return sample[target_col].values


# ── drift cycles ──────────────────────────────────────────────────────────────

def run_cycle(cycle: int, ref: dict, df: pd.DataFrame, consec: int) -> int:
    log.info(f"--- Cycle {cycle}  (drift_phase={cycle >= 3}) ---")

    buf = ProductionBuffer(df, cycle)
    buf.fill()
    feat_df = buf.features_df()
    preds   = buf.predictions()

    # ── Data drift: KS per feature ────────────────────────────────────────────
    data_drifted  = False
    drifted_feats = []
    for feat in FEATURE_NAMES:
        stat, pval = stats.ks_2samp(ref[feat], feat_df[feat].values)
        if pval < KS_THR:
            data_drifted = True
            drifted_feats.append(feat)
        log.info(f"  KS [{feat:18s}]: stat={stat:.4f}  pval={pval:.4f}  "
                 f"drift={'YES' if pval < KS_THR else 'no'}")

    # ── Concept drift: KS on predictions ─────────────────────────────────────
    concept_drifted = False
    pred_ks_pval    = 1.0
    ref_preds = ref.get("pred_dist")
    if ref_preds is not None and len(ref_preds) >= 5 and len(preds) >= 5:
        stat, pred_ks_pval = stats.ks_2samp(ref_preds, preds)
        concept_drifted    = pred_ks_pval < KS_THR
        log.info(f"  KS [predictions  ]: stat={stat:.4f}  pval={pred_ks_pval:.4f}  "
                 f"drift={'YES' if concept_drifted else 'no'}")

    any_drift = data_drifted or concept_drifted
    consec    = consec + 1 if any_drift else 0

    log.info(f"  Status: {'DRIFT ⚠' if any_drift else 'OK ✓'}  "
             f"consecutive={consec}/{CONSEC_NEED}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("drift-monitoring")
        with mlflow.start_run(run_name=f"window-{cycle:03d}"):
            for feat in FEATURE_NAMES:
                s = feat.replace(" ", "_")
                stat_v, pval_v = stats.ks_2samp(ref[feat], feat_df[feat].values)
                mlflow.log_metric(f"ks_stat_{s}", float(stat_v))
                mlflow.log_metric(f"ks_pval_{s}", float(pval_v))
            mlflow.log_metric("pred_ks_pval",  float(pred_ks_pval))
            mlflow.log_metric("data_drift",    int(data_drifted))
            mlflow.log_metric("concept_drift", int(concept_drifted))
            mlflow.log_metric("consec_drifts", consec)
            mlflow.log_params({
                "cycle": cycle, "window_size": WIN_SIZE, "ks_thr": KS_THR,
            })
        log.info("  MLflow run logged ✓")
    except Exception:
        log.error(f"  MLflow logging failed:\n{traceback.format_exc()}")

    # ── Trigger retrain ───────────────────────────────────────────────────────
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


def run_performance_drift_cycle(
    cycle: int,
    perf_monitor: PerformanceDriftMonitor,
    df: pd.DataFrame,
) -> int:
    if not ENABLE_PERF_DRIFT:
        return 0

    log.info(f"--- Performance Drift Cycle {cycle} ---")
    try:
        df_sc  = df[df["Scenario"] == 0].sample(min(WIN_SIZE, len(df)), replace=True)
        y_true = df_sc["TotalRecollectedCrops_crop_units"].values.astype(float)

        # Simulate predictions: stable early, degraded later
        if cycle < 3:
            y_pred = y_true + np.random.randn(len(y_true)) * 5.0
        else:
            bias   = (cycle - 2) * 10.0
            y_pred = y_true * 0.75 + np.random.randn(len(y_true)) * 20.0 + bias

        perf_monitor.add_batch(y_pred, y_true)
        drift, results = perf_monitor.check_drift()

        if results:
            metrics = results[0].get("metrics", {})
            log.info(f"  Perf drift={drift}  R²={metrics.get('r2', 'N/A'):.4f}  "
                     f"RMSE={metrics.get('rmse', 'N/A'):.2f}")

            try:
                mlflow.set_tracking_uri(MLFLOW_URI)
                mlflow.set_experiment("performance-drift-monitoring")
                with mlflow.start_run(run_name=f"perf-window-{cycle:03d}"):
                    for k, v in metrics.items():
                        mlflow.log_metric(f"current_{k}", float(v))
                    mlflow.log_metric("perf_drift_detected", int(drift))
                    mlflow.log_param("cycle", cycle)
            except Exception:
                log.warning("MLflow performance logging failed")

        return int(drift)
    except Exception:
        log.error(f"  Perf drift error:\n{traceback.format_exc()}")
        return 0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Drift Detector — HRI Regression mode")
    log.info(f"  MLFLOW_URI={MLFLOW_URI}  WIN_SIZE={WIN_SIZE}  INTERVAL={INTERVAL_S}s")

    wait_for(f"{MLFLOW_URI}/",     "MLflow")
    wait_for(f"{INFER_URL}/health", "InferenceAPI", retries=10, delay=3)

    df  = _load_preprocessed()
    ref = build_reference(df)

    baseline_metrics = build_baseline_metrics(df)
    log.info(f"Baseline metrics: {baseline_metrics}")

    perf_monitor = PerformanceDriftMonitor(
        baseline_metrics,
        window_size=WIN_SIZE,
        num_windows=3,
        effect_size_threshold=PERF_DRIFT_THR,
        p_value_threshold=PERF_DRIFT_P_VALUE,
    )

    consec      = 0
    perf_consec = 0
    cycle       = 0

    while True:
        try:
            consec      = run_cycle(cycle, ref, df, consec)
            perf_consec += run_performance_drift_cycle(cycle, perf_monitor, df)
            if perf_consec >= PERF_CONSEC_NEED:
                log.warning(f"  PERFORMANCE DRIFT RETRAINING TRIGGERED")
                try:
                    retrain_trigger.trigger(
                        reason="performance_drift",
                        consecutive_windows=perf_consec,
                        mlflow_uri=MLFLOW_URI,
                    )
                    perf_consec = 0
                    perf_monitor.reset_buffer()
                except Exception:
                    log.error(traceback.format_exc())
        except Exception:
            log.error(f"Unhandled error in cycle {cycle}:\n{traceback.format_exc()}")

        cycle += 1
        log.info(f"  Next cycle in {INTERVAL_S}s ...")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
