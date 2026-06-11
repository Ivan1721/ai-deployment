"""
retrain_trigger.py  ─  Champion/Challenger para modelos HRI de regresión.

Cuando el detector detecta drift, este módulo:
  1. Loguea el evento en MLflow.
  2. Entrena un challenger con GradientBoosting para los 8 slots.
  3. Por cada slot, compara el R² del challenger vs el champion.
  4. Promueve el challenger si mejora el R².
  5. Recarga la inference-api vía POST /reload.
"""

import os, time, logging, traceback
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

log = logging.getLogger(__name__)

DATASET_PATH  = os.environ.get("DATASET_PATH",      "/data/simulation_all.csv")
INFERENCE_URL = os.environ.get("INFERENCE_API_URL",  "http://inference-api:8000")
MIN_DELTA     = float(os.environ.get("CHALLENGER_MIN_IMPROVEMENT", "0.0"))

SCENARIOS = {0: "HumanOnly", 1: "WithRobot"}
TARGETS   = {
    "TotalRecollected": "TotalRecollectedCrops_crop_units",
    "CargoZoneProd":    "TotalProductionCargoZone_crop_units",
    "TotalWorkload":    "TotalHumanWorkload_kcal",
    "AvgProduction":    "AverageHumanProduction_crop_units",
}
FEATURE_NAMES = ["Humans", "ROW_N", "RandomPosition", "Act_Ladder", "Act_Mixed", "Act_Picker"]

CHALLENGER_PARAMS = {
    "n_estimators": 200, "max_depth": 4,
    "learning_rate": 0.05, "random_state": 42,
}


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _champion_r2(client: MlflowClient, model_name: str) -> float | None:
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return float(run.data.metrics.get("ho_r2", 0.0))
    except Exception:
        return None


def _challenger_r2(client: MlflowClient, model_name: str) -> tuple:
    """Returns (version_str, r2) for newest undeployed version."""
    try:
        versions   = client.search_model_versions(f"name='{model_name}'")
        candidates = [v for v in versions if v.current_stage == "None"]
        if not candidates:
            return None, None
        newest = sorted(candidates, key=lambda v: int(v.version))[-1]
        run    = client.get_run(newest.run_id)
        r2     = float(run.data.metrics.get("ho_r2", 0.0))
        return newest.version, r2
    except Exception:
        log.error(traceback.format_exc())
        return None, None


def _reload_api():
    try:
        import urllib.request
        req  = urllib.request.Request(
            f"{INFERENCE_URL}/reload", data=b"", method="POST",
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        log.info(f"API reload: {resp.status}")
    except Exception as e:
        log.warning(f"Could not reload API: {e}")


def _train_challengers(mlflow_uri: str) -> bool:
    """Trains a GradientBoosting challenger for every scenario × target slot."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("hri-harvesting")

    df = _load_preprocessed()

    for scenario_id, scenario_label in SCENARIOS.items():
        df_sc = df[df["Scenario"] == scenario_id].copy()
        X     = df_sc[FEATURE_NAMES].values

        for target_alias, target_col in TARGETS.items():
            y          = df_sc[target_col].values
            model_name = f"hri-{scenario_label}-{target_alias}"
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=int(time.time()) % 9999
            )

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  GradientBoostingRegressor(**CHALLENGER_PARAMS)),
            ])
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)

            metrics = {
                "ho_r2":   float(r2_score(y_te, y_pred)),
                "ho_rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
                "ho_mae":  float(mean_absolute_error(y_te, y_pred)),
            }
            log.info(f"Challenger {scenario_label}-{target_alias}: "
                     f"R²={metrics['ho_r2']:.4f}")

            with mlflow.start_run(
                run_name=f"{scenario_label}-{target_alias}-challenger",
                tags={"phase": "challenger", "triggered_by": "drift_detector"},
            ):
                mlflow.log_params({
                    **CHALLENGER_PARAMS,
                    "algorithm":  "GradientBoosting",
                    "scenario":   scenario_label,
                    "target":     target_alias,
                })
                mlflow.log_metrics(metrics)
                sig = infer_signature(X_tr, pipe.predict(X_tr))
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    artifact_path="model",
                    registered_model_name=model_name,
                    signature=sig,
                )
    return True


# ── main entry point ──────────────────────────────────────────────────────────

def trigger(reason: str, consecutive_windows: int, mlflow_uri: str):
    log.info("━" * 55)
    log.info(f"  RETRAIN TRIGGER  reason={reason}  consec={consecutive_windows}")
    log.info("━" * 55)

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(mlflow_uri)

    # 1. Log trigger event
    mlflow.set_experiment("retraining-events")
    with mlflow.start_run(run_name=f"trigger-{reason}"):
        mlflow.log_params({"reason": reason, "consecutive_windows": consecutive_windows})
        mlflow.set_tag("event_type", "retrain_trigger")
    log.info("Trigger event logged ✓")

    # 2. Train challengers for all 8 slots
    try:
        _train_challengers(mlflow_uri)
    except Exception:
        log.error(f"Challenger training failed:\n{traceback.format_exc()}")
        return

    time.sleep(3)  # allow Registry to settle

    # 3. Compare and promote per slot
    for scenario_id, scenario_label in SCENARIOS.items():
        for target_alias in TARGETS:
            model_name  = f"hri-{scenario_label}-{target_alias}"
            champ_r2    = _champion_r2(client, model_name)
            chal_ver, chal_r2 = _challenger_r2(client, model_name)

            if chal_ver is None:
                log.error(f"No challenger found for {model_name}")
                continue

            log.info(f"{model_name}: champion R²={champ_r2}  challenger v{chal_ver} R²={chal_r2:.4f}")

            promote = (champ_r2 is None) or (chal_r2 is not None and
                                              chal_r2 >= (champ_r2 or 0.0) + MIN_DELTA)

            mlflow.set_experiment("retraining-events")
            with mlflow.start_run(run_name=f"promotion-{model_name}-v{chal_ver}"):
                mlflow.log_params({"model": model_name, "challenger_version": chal_ver})
                mlflow.log_metric("challenger_r2", chal_r2 or 0.0)
                mlflow.log_metric("champion_r2",   champ_r2 or 0.0)
                mlflow.log_metric("promoted",      int(promote))

            if promote:
                log.info(f"✅  Promoting {model_name} v{chal_ver} → Production")
                client.transition_model_version_stage(
                    name=model_name, version=chal_ver,
                    stage="Production", archive_existing_versions=True,
                )
            else:
                log.info(f"⏭   {model_name} challenger does not improve — archiving")
                client.transition_model_version_stage(
                    name=model_name, version=chal_ver, stage="Archived",
                )

    _reload_api()
    log.info("Retrain trigger complete ✓")
