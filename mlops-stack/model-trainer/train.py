"""
train.py
────────
Trains RandomForestRegressor models on the HRI Agricultural Harvesting Dataset.
2 scenarios x 4 targets = 8 models, each registered in MLflow Model Registry.
Dataset: Vasconez & Auat Cheein (2022), Biosystems Engineering Vol. 223.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
ARTIFACT_ROOT = os.environ.get("MLFLOW_ARTIFACT_ROOT", None)
DATASET_PATH  = os.environ.get("DATASET_PATH", "/data/simulation_all.csv")
EXPERIMENT    = "hri-harvesting"

TARGETS = {
    "TotalRecollected": "TotalRecollectedCrops_crop_units",
    "CargoZoneProd":    "TotalProductionCargoZone_crop_units",
    "TotalWorkload":    "TotalHumanWorkload_kcal",
    "AvgProduction":    "AverageHumanProduction_crop_units",
}

SCENARIOS = {0: "HumanOnly", 1: "WithRobot"}

FEATURE_NAMES = ["Humans", "ROW_N", "RandomPosition", "Act_Ladder", "Act_Mixed", "Act_Picker"]

PARAMS = {
    "n_estimators":      200,
    "max_depth":         8,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "random_state":      42,
}


def wait_for_mlflow(uri: str, retries: int = 15, delay: int = 4) -> None:
    import urllib.request
    for i in range(retries):
        try:
            urllib.request.urlopen(f"{uri}/", timeout=3)
            log.info("MLFlow server is up")
            return
        except Exception:
            log.info(f"Waiting for MLFlow... attempt {i+1}/{retries}")
            time.sleep(delay)
    raise RuntimeError("MLFlow server did not respond in time.")


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

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


def run_quality_gate(pipe, X_test: np.ndarray, y_test: np.ndarray,
                     min_r2: float = 0.70) -> bool:
    y_pred   = pipe.predict(X_test)
    r2       = r2_score(y_test, y_pred)
    passed   = r2 >= min_r2
    log.info(f"Quality Gate -- R2={r2:.4f} (min={min_r2})  {'PASSED' if passed else 'FAILED'}")
    return passed


def main():
    wait_for_mlflow(TRACKING_URI)
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(TRACKING_URI)
    if ARTIFACT_ROOT:
        try:
            exp = client.get_experiment_by_name(EXPERIMENT)
            if exp is None:
                client.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)
        except Exception as e:
            log.warning(f"Could not set artifact location: {e}")

    mlflow.set_experiment(EXPERIMENT)

    df = load_and_preprocess(DATASET_PATH)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for scenario_id, scenario_label in SCENARIOS.items():
        df_sc = df[df["Scenario"] == scenario_id].copy()
        log.info(f"\n{'='*55}")
        log.info(f"Scenario {scenario_id}: {scenario_label}  ({len(df_sc)} rows)")

        X = df_sc[FEATURE_NAMES].values

        for target_alias, target_col in TARGETS.items():
            y          = df_sc[target_col].values
            model_name = f"hri-{scenario_label}-{target_alias}"
            run_name   = f"{scenario_label}-{target_alias}"
            log.info(f"\n  Training: {run_name}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("rf",     RandomForestRegressor(**PARAMS)),
            ])

            cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            metrics = {
                "cv_r2_mean": float(cv_r2.mean()),
                "cv_r2_std":  float(cv_r2.std()),
                "ho_r2":      float(r2_score(y_test, y_pred)),
                "ho_rmse":    float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "ho_mae":     float(mean_absolute_error(y_test, y_pred)),
            }
            log.info(f"  cv_r2={metrics['cv_r2_mean']:.4f}+-{metrics['cv_r2_std']:.4f}"
                     f"  ho_r2={metrics['ho_r2']:.4f}  rmse={metrics['ho_rmse']:.4f}")

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    **PARAMS,
                    "scenario":   scenario_label,
                    "target":     target_alias,
                    "feature_set": "A",
                    "n_features": len(FEATURE_NAMES),
                    "n_train":    len(X_train),
                    "n_test":     len(X_test),
                    "dataset":    DATASET_PATH,
                })
                mlflow.log_metrics(metrics)

                sig = infer_signature(X_train, pipe.predict(X_train))
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    artifact_path="model",
                    registered_model_name=model_name,
                    signature=sig,
                    input_example=X_test[:3],
                )

            if not run_quality_gate(pipe, X_test, y_test):
                log.warning(f"  Skipping promotion for {model_name} -- quality gate failed")
                continue

            versions = client.search_model_versions(f"name='{model_name}'")
            latest   = sorted(versions, key=lambda v: int(v.version))[-1]
            client.transition_model_version_stage(
                name=model_name,
                version=latest.version,
                stage="Production",
                archive_existing_versions=True,
            )
            log.info(f"  '{model_name}' v{latest.version} -> Production")

    log.info("\nTraining complete -- up to 8 models registered.")


if __name__ == "__main__":
    main()
