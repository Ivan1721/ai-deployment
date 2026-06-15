"""
train.py
────────
Multi-model tournament trainer for the MLOps stack.

Reads all configuration from MODEL_CONFIG (YAML via mlops_common).
When cfg.multi_slot is set, runs a 2-scenarios × 4-targets tournament
(HRI mode); otherwise trains a single model (generic mode for Iris-style cases).

Dataset: Vasconez & Auat Cheein (2022), Biosystems Engineering Vol. 223.
"""

import os
import time
import logging
import json
import numpy as np
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
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             max_error as sklearn_max_error)
from sklearn.inspection import permutation_importance

from mlops_common import load_config
from mlops_common.data_sources import get_data_source
from mlops_common.problem_types import get_problem_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
ARTIFACT_ROOT = os.environ.get("MLFLOW_ARTIFACT_ROOT", None)


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


# ── Metric helpers ────────────────────────────────────────────────────────────

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (0–100%). Robust to near-zero targets."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.where(denom == 0, 0.0, np.abs(y_pred - y_true) / denom)) * 100)


def feature_ranking(pipeline, X_te: np.ndarray, y_te: np.ndarray,
                    feature_names: list) -> dict:
    """Permutation importance — works for any sklearn-compatible pipeline."""
    result = permutation_importance(pipeline, X_te, y_te,
                                    n_repeats=10, random_state=42, scoring="r2")
    ranks  = np.argsort(np.argsort(-result.importances_mean)) + 1
    return {
        name: {
            "importance_mean": round(float(result.importances_mean[i]), 6),
            "importance_std":  round(float(result.importances_std[i]),  6),
            "rank":            int(ranks[i]),
        }
        for i, name in enumerate(feature_names)
    }


# ── Candidate factory ─────────────────────────────────────────────────────────

_ALL_CANDIDATES = {
    "linear_regression": lambda: Pipeline([
        ("scaler", StandardScaler()), ("model", LinearRegression()),
    ]),
    "ridge": lambda: Pipeline([
        ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0)),
    ]),
    "lasso": lambda: Pipeline([
        ("scaler", StandardScaler()), ("model", Lasso(alpha=0.1, max_iter=2000)),
    ]),
    "elastic_net": lambda: Pipeline([
        ("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)),
    ]),
    "svr": lambda: Pipeline([
        ("scaler", StandardScaler()), ("model", SVR(C=10.0, epsilon=0.1, kernel="rbf")),
    ]),
    "extra_trees_regressor": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
    ]),
    "random_forest_regressor": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=200, max_depth=8,
                                        min_samples_split=4, min_samples_leaf=2,
                                        random_state=42, n_jobs=-1)),
    ]),
    "gradient_boosting_regressor": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                             learning_rate=0.05, random_state=42)),
    ]),
    "mlp_regressor": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
    ]),
}

if _HAS_XGB:
    _ALL_CANDIDATES["xgboost_regressor"] = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", _XGB(n_estimators=200, max_depth=4, learning_rate=0.05,
                       random_state=42, verbosity=0)),
    ])
if _HAS_LGB:
    _ALL_CANDIDATES["lightgbm_regressor"] = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", _LGB(n_estimators=200, max_depth=4, learning_rate=0.05,
                       random_state=42, verbose=-1)),
    ])
if _HAS_CB:
    _ALL_CANDIDATES["catboost_regressor"] = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", _CB(iterations=200, depth=4, learning_rate=0.05,
                      random_seed=42, verbose=0)),
    ])


def build_candidates(names: list) -> dict:
    """Returns fresh (unfitted) pipelines for the requested candidate names."""
    result = {}
    for name in names:
        if name in _ALL_CANDIDATES:
            result[name] = _ALL_CANDIDATES[name]()
        else:
            log.warning(f"Unknown candidate '{name}' — skipping")
    return result


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    wait_for_mlflow(TRACKING_URI)
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(TRACKING_URI)
    if ARTIFACT_ROOT:
        try:
            if client.get_experiment_by_name(cfg.mlflow.experiment_name) is None:
                client.create_experiment(cfg.mlflow.experiment_name,
                                         artifact_location=ARTIFACT_ROOT)
        except Exception as e:
            log.warning(f"Could not set artifact location: {e}")

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    bundle = get_data_source(cfg.data_source).load()
    problem = get_problem_type(cfg.problem_type)

    # ── Multi-slot mode (HRI: 2 scenarios × 4 targets) ───────────────────────
    if cfg.multi_slot is not None:
        ms            = cfg.multi_slot
        SCENARIOS     = ms.scenarios
        TARGETS       = ms.targets
        FEATURE_NAMES = ms.feature_names
        QUALITY_GATE  = cfg.quality_gates.get("min_r2", 0.70)
        df            = bundle.extra["df"]
        cv            = KFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=42)

        for scenario_id, scenario_label in SCENARIOS.items():
            df_sc = df[df["Scenario"] == scenario_id].copy()
            X     = df_sc[FEATURE_NAMES].values
            log.info(f"\n{'='*60}")
            log.info(f"Scenario {scenario_id}: {scenario_label}  ({len(df_sc)} rows)")

            for target_alias, target_col in TARGETS.items():
                y = df_sc[target_col].values
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=cfg.training.test_size, random_state=42
                )
                log.info(f"\n  Target: {target_alias}")

                # Phase 1 — Tournament
                candidates  = build_candidates(ms.tournament_candidates)
                cv_results  = {}
                best_name   = None
                best_cv_r2  = -np.inf

                for cand_name, pipe in candidates.items():
                    scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2")
                    cv_results[cand_name] = {
                        "cv_r2_mean": float(scores.mean()),
                        "cv_r2_std":  float(scores.std()),
                    }
                    with mlflow.start_run(
                        run_name=f"{scenario_label}-{target_alias}-{cand_name}",
                        tags={"phase": "comparison",
                              "scenario": scenario_label,
                              "target": target_alias,
                              "config": cfg.name},
                    ):
                        mlflow.log_params({
                            "algorithm": cand_name,
                            "scenario":  scenario_label,
                            "target":    target_alias,
                        })
                        mlflow.log_metrics(cv_results[cand_name])

                    log.info(f"    {cand_name:30s}  cv_r2={scores.mean():.4f} ± {scores.std():.4f}")
                    if scores.mean() > best_cv_r2:
                        best_cv_r2 = scores.mean()
                        best_name  = cand_name

                log.info(f"  >>> Winner: {best_name}  (cv_r2={best_cv_r2:.4f})")

                # Phase 2 — Fit winner + register
                winner    = candidates[best_name]
                winner.fit(X_tr, y_tr)
                y_pred    = winner.predict(X_te)
                model_reg = f"hri-{scenario_label}-{target_alias}"

                feat_rank = feature_ranking(winner, X_te, y_te, FEATURE_NAMES)
                ho_metrics = problem.compute_metrics(y_te, y_pred)
                metrics = {
                    "cv_r2_mean":   best_cv_r2,
                    "cv_r2_std":    cv_results[best_name]["cv_r2_std"],
                    "ho_r2":        ho_metrics["r2"],
                    "ho_rmse":      ho_metrics["rmse"],
                    "ho_mae":       ho_metrics["mae"],
                    "ho_smape":     ho_metrics["smape"],
                    "ho_max_error": ho_metrics["max_error"],
                }
                log.info(
                    f"    ho_r2={metrics['ho_r2']:.4f}  rmse={metrics['ho_rmse']:.4f}"
                    f"  mae={metrics['ho_mae']:.4f}  smape={metrics['ho_smape']:.2f}%"
                )

                with mlflow.start_run(
                    run_name=f"{scenario_label}-{target_alias}",
                    tags={"phase": "winner", "config": cfg.name},
                ):
                    mlflow.log_params({
                        "algorithm":   best_name,
                        "scenario":    scenario_label,
                        "target":      target_alias,
                        "n_features":  len(FEATURE_NAMES),
                        "n_train":     len(X_tr),
                        "n_test":      len(X_te),
                        "config":      cfg.name,
                    })
                    mlflow.set_tag("problem_type", cfg.problem_type)
                    mlflow.set_tag("feature_names", json.dumps(FEATURE_NAMES))
                    mlflow.log_metrics(metrics)
                    mlflow.log_dict(feat_rank, "feature_ranking.json")
                    for cname, cres in cv_results.items():
                        mlflow.log_metric(f"cmp_{cname}_cv_r2", float(cres["cv_r2_mean"]))

                    sig = infer_signature(X_tr, winner.predict(X_tr))
                    mlflow.sklearn.log_model(
                        sk_model=winner,
                        artifact_path="model",
                        registered_model_name=model_reg,
                        signature=sig,
                        input_example=X_te[:3],
                    )

                # Phase 3 — Quality gate + promote
                ok, failures = problem.quality_gate(ho_metrics, cfg.quality_gates)
                if not ok:
                    log.warning(f"  Skipping promotion: {failures}")
                    continue

                versions = client.search_model_versions(f"name='{model_reg}'")
                latest   = sorted(versions, key=lambda v: int(v.version))[-1]
                client.transition_model_version_stage(
                    name=model_reg,
                    version=latest.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                log.info(f"  '{model_reg}' v{latest.version} ({best_name}) → Production ✓")

        log.info("\nTraining complete — up to 8 models registered.")

    else:
        # ── Single-model mode (generic: Iris, Wine, CSV, ...) ────────────────
        log.info(f"Single-model mode: {cfg.name}  ({cfg.model.family})")
        from mlops_common.models import get_model_strategy

        X, y = bundle.X, bundle.y
        split_kwargs = {"test_size": cfg.training.test_size, "random_state": 42}
        if cfg.training.stratify:
            split_kwargs["stratify"] = y
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, **split_kwargs)

        strategy = get_model_strategy(cfg.model.family)
        model    = strategy.build(cfg.model.params)
        model    = strategy.fit(model, X_tr, y_tr)
        y_pred   = strategy.predict(model, X_te)
        y_proba  = strategy.predict_proba(model, X_te) if strategy.supports_proba else None

        metrics  = problem.compute_metrics(y_te, y_pred, y_proba)
        cv_scores = strategy.cv_score(model, X, y,
                                       cv=cfg.training.cv_folds,
                                       scoring=problem.primary_metric)
        metrics["cv_mean"] = float(cv_scores.mean())
        metrics["cv_std"]  = float(cv_scores.std())

        with mlflow.start_run(run_name=f"{cfg.name}-v1",
                              tags={"config": cfg.name}):
            mlflow.log_params(cfg.model.params)
            mlflow.log_param("dataset", f"{cfg.data_source.type}:{cfg.data_source.name or cfg.data_source.path}")
            mlflow.set_tag("problem_type", cfg.problem_type)
            mlflow.set_tag("feature_names", json.dumps(bundle.feature_names))
            if bundle.target_names is not None:
                mlflow.set_tag("target_names", json.dumps(bundle.target_names))
            if bundle.target_name is not None:
                mlflow.set_tag("target_name", bundle.target_name)
            mlflow.log_metrics(metrics)
            log.info("  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            sig = infer_signature(X_tr, model.predict(X_tr))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=cfg.mlflow.registered_model_name,
                signature=sig,
                input_example=X_te.iloc[:3] if hasattr(X_te, "iloc") else X_te[:3],
            )

        ok, failures = problem.quality_gate(metrics, cfg.quality_gates)
        if not ok:
            log.warning(f"Quality gate FAILED: {failures}")
            return

        versions = client.search_model_versions(f"name='{cfg.mlflow.registered_model_name}'")
        latest   = sorted(versions, key=lambda v: int(v.version))[-1]
        client.transition_model_version_stage(
            name=cfg.mlflow.registered_model_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        log.info(f"'{cfg.mlflow.registered_model_name}' v{latest.version} → Production ✓")


if __name__ == "__main__":
    main()
