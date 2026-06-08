"""
train.py
────────
Multi-model comparison for the HRI Agricultural Harvesting Dataset.
2 scenarios × 4 targets = 8 model slots.
Each slot: all candidates evaluated with 5-fold CV → best R² wins → registered.
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
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
SCENARIOS     = {0: "HumanOnly", 1: "WithRobot"}
FEATURE_NAMES = ["Humans", "ROW_N", "RandomPosition", "Act_Ladder", "Act_Mixed", "Act_Picker"]
QUALITY_GATE  = 0.70


# ── Optional heavy packages ───────────────────────────────────────────────────

try:
    from xgboost import XGBRegressor as _XGB
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor as _LGB
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostRegressor as _CB
    _HAS_CB = True
except ImportError:
    _HAS_CB = False


# ── Candidate factory ─────────────────────────────────────────────────────────

def build_candidates() -> dict[str, Pipeline]:
    """Returns fresh (unfitted) pipeline instances for every candidate."""
    c: dict[str, Pipeline] = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=0.1, max_iter=2000)),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)),
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVR(C=10.0, epsilon=0.1, kernel="rbf")),
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  ExtraTreesRegressor(n_estimators=200, max_depth=8,
                                           random_state=42, n_jobs=-1)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(n_estimators=200, max_depth=8,
                                             min_samples_split=4, min_samples_leaf=2,
                                             random_state=42, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                  learning_rate=0.05, random_state=42)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                    random_state=42)),
        ]),
    }
    if _HAS_XGB:
        c["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  _XGB(n_estimators=200, max_depth=4, learning_rate=0.05,
                            random_state=42, verbosity=0)),
        ])
    if _HAS_LGB:
        c["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  _LGB(n_estimators=200, max_depth=4, learning_rate=0.05,
                            random_state=42, verbose=-1)),
        ])
    if _HAS_CB:
        c["CatBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  _CB(iterations=200, depth=4, learning_rate=0.05,
                           random_seed=42, verbose=0)),
        ])
    return c


# ── Helpers ───────────────────────────────────────────────────────────────────

def wait_for_mlflow(uri: str, retries: int = 15, delay: int = 4) -> None:
    import urllib.request
    for i in range(retries):
        try:
            urllib.request.urlopen(f"{uri}/", timeout=3)
            log.info("MLflow server is up")
            return
        except Exception:
            log.info(f"Waiting for MLflow... attempt {i+1}/{retries}")
            time.sleep(delay)
    raise RuntimeError("MLflow server did not respond in time.")


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    wait_for_mlflow(TRACKING_URI)
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(TRACKING_URI)
    if ARTIFACT_ROOT:
        try:
            if client.get_experiment_by_name(EXPERIMENT) is None:
                client.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)
        except Exception as e:
            log.warning(f"Could not set artifact location: {e}")

    mlflow.set_experiment(EXPERIMENT)
    df  = load_and_preprocess(DATASET_PATH)
    cv  = KFold(n_splits=5, shuffle=True, random_state=42)

    for scenario_id, scenario_label in SCENARIOS.items():
        df_sc = df[df["Scenario"] == scenario_id].copy()
        X     = df_sc[FEATURE_NAMES].values
        log.info(f"\n{'='*60}")
        log.info(f"Scenario {scenario_id}: {scenario_label}  ({len(df_sc)} rows)")

        for target_alias, target_col in TARGETS.items():
            y = df_sc[target_col].values
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=42
            )

            log.info(f"\n  Target: {target_alias}")

            # ── Phase 1: evaluate all candidates ─────────────────
            candidates  = build_candidates()
            cv_results: dict[str, dict] = {}
            best_name   = None
            best_cv_r2  = -np.inf

            for cand_name, pipe in candidates.items():
                scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2")
                cv_results[cand_name] = {
                    "cv_r2_mean": float(scores.mean()),
                    "cv_r2_std":  float(scores.std()),
                }
                run_name = f"{scenario_label}-{target_alias}-{cand_name}"
                with mlflow.start_run(run_name=run_name,
                                      tags={"phase": "comparison",
                                            "scenario": scenario_label,
                                            "target": target_alias}):
                    mlflow.log_params({
                        "algorithm": cand_name,
                        "scenario":  scenario_label,
                        "target":    target_alias,
                    })
                    mlflow.log_metrics(cv_results[cand_name])

                log.info(f"    {cand_name:20s}  cv_r2={scores.mean():.4f} ± {scores.std():.4f}")

                if scores.mean() > best_cv_r2:
                    best_cv_r2 = scores.mean()
                    best_name  = cand_name

            log.info(f"  >>> Winner: {best_name}  (cv_r2={best_cv_r2:.4f})")

            # ── Phase 2: fit winner on full train set, register ───
            winner    = candidates[best_name]
            winner.fit(X_tr, y_tr)
            y_pred    = winner.predict(X_te)
            model_reg = f"hri-{scenario_label}-{target_alias}"

            metrics = {
                "cv_r2_mean": best_cv_r2,
                "cv_r2_std":  cv_results[best_name]["cv_r2_std"],
                "ho_r2":      float(r2_score(y_te, y_pred)),
                "ho_rmse":    float(np.sqrt(mean_squared_error(y_te, y_pred))),
                "ho_mae":     float(mean_absolute_error(y_te, y_pred)),
            }
            log.info(f"    ho_r2={metrics['ho_r2']:.4f}  "
                     f"rmse={metrics['ho_rmse']:.4f}  mae={metrics['ho_mae']:.4f}")

            with mlflow.start_run(run_name=f"{scenario_label}-{target_alias}",
                                  tags={"phase": "winner"}):
                mlflow.log_params({
                    "algorithm":   best_name,
                    "scenario":    scenario_label,
                    "target":      target_alias,
                    "feature_set": "A",
                    "n_features":  len(FEATURE_NAMES),
                    "n_train":     len(X_tr),
                    "n_test":      len(X_te),
                    "dataset":     DATASET_PATH,
                })
                mlflow.log_metrics(metrics)
                # log all candidates for easy paper comparison
                for cname, cres in cv_results.items():
                    mlflow.log_metric(f"cmp_{cname}_cv_r2",
                                      float(cres["cv_r2_mean"]))

                sig = infer_signature(X_tr, winner.predict(X_tr))
                mlflow.sklearn.log_model(
                    sk_model=winner,
                    artifact_path="model",
                    registered_model_name=model_reg,
                    signature=sig,
                    input_example=X_te[:3],
                )

            # ── Phase 3: quality gate + promote ──────────────────
            r2_ho = metrics["ho_r2"]
            if r2_ho < QUALITY_GATE:
                log.warning(f"  Skipping promotion: R2={r2_ho:.4f} < {QUALITY_GATE}")
                continue

            versions = client.search_model_versions(f"name='{model_reg}'")
            latest   = sorted(versions, key=lambda v: int(v.version))[-1]
            client.transition_model_version_stage(
                name=model_reg,
                version=latest.version,
                stage="Production",
                archive_existing_versions=True,
            )
            log.info(f"  '{model_reg}' v{latest.version} ({best_name}) -> Production")

    log.info("\nTraining complete — up to 8 models registered.")


if __name__ == "__main__":
    main()
