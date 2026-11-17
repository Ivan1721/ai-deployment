"""
retrain_trigger.py  ─  Champion/Challenger para modelos HRI de regresión.

Cuando el detector detecta drift, este módulo:
  1. Loguea el evento en MLflow.
  2. Entrena un challenger con GradientBoosting para cada slot.
  3. Evalúa champion y challenger sobre EL MISMO hold-out (el split de
     train.py, random_state=42) — comparar R² medidos en particiones distintas
     de ~28 filas es comparar ruido.
  4. Promueve el challenger solo si pasa los quality gates del config y
     mejora estrictamente el R² del champion.
  5. Recarga la inference-api vía POST /reload si hubo alguna promoción.
"""

import os, time, json, logging, traceback
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_common import load_config
from mlops_common.data_sources import get_data_source
from mlops_common.problem_types import get_problem_type

log = logging.getLogger(__name__)

INFERENCE_URL = os.environ.get("INFERENCE_API_URL", "http://inference-api:8000")
API_KEY       = os.environ.get("API_KEY", "")
# Same split as model-trainer/train.py and tests/test_model.py.
SPLIT_SEED    = 42

CHALLENGER_PARAMS = {
    "n_estimators": 200, "max_depth": 4,
    "learning_rate": 0.05, "random_state": 42,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _min_delta(cfg) -> float:
    env = os.environ.get("CHALLENGER_MIN_IMPROVEMENT")
    if env is not None:
        return float(env)
    return float(getattr(cfg.retrain, "challenger_min_improvement", 0.0))


def _champion_metrics(problem, model_name: str, X_te, y_te) -> dict | None:
    """Metrics of the current Production model on the shared hold-out."""
    try:
        champion = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    except Exception as e:
        log.info(f"{model_name}: no loadable champion ({type(e).__name__})")
        return None
    return problem.compute_metrics(y_te, champion.predict(X_te))


def _feature_ranking(pipeline, X_te, y_te, feature_names: list) -> dict:
    """Permutation importance — same artifact layout as train.py."""
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


def _version_for_run(client: MlflowClient, model_name: str, run_id: str) -> str | None:
    for v in client.search_model_versions(f"name='{model_name}'"):
        if v.run_id == run_id:
            return v.version
    return None


def _reload_api():
    try:
        import urllib.request
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        req  = urllib.request.Request(
            f"{INFERENCE_URL}/reload", data=b"", method="POST",
            headers=headers,
        )
        resp = urllib.request.urlopen(req, timeout=30)
        log.info(f"API reload: {resp.status}")
    except Exception as e:
        log.warning(f"Could not reload API: {e}")


def _train_challenger(cfg, problem, df_sc, scenario_label: str,
                      target_alias: str, target_col: str) -> dict:
    """
    Trains and registers a challenger for one slot. Returns
    {"model_name", "run_id", "metrics", "champion_metrics"}.
    """
    ms         = cfg.multi_slot
    model_name = f"hri-{scenario_label}-{target_alias}"
    X          = df_sc[ms.feature_names].values
    y          = df_sc[target_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.training.test_size, random_state=SPLIT_SEED,
    )

    champion_metrics = _champion_metrics(problem, model_name, X_te, y_te)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(**CHALLENGER_PARAMS)),
    ])
    pipe.fit(X_tr, y_tr)
    ho      = problem.compute_metrics(y_te, pipe.predict(X_te))
    metrics = {f"ho_{k}": v for k, v in ho.items()}
    log.info(f"Challenger {scenario_label}-{target_alias}: R²={ho['r2']:.4f}")

    with mlflow.start_run(
        run_name=f"{scenario_label}-{target_alias}-challenger",
        tags={"phase": "challenger", "triggered_by": "drift_detector",
              "config": cfg.name},
    ) as run:
        mlflow.log_params({
            **CHALLENGER_PARAMS,
            "algorithm":  "GradientBoosting",
            "scenario":   scenario_label,
            "target":     target_alias,
            "n_train":    len(X_tr),
            "n_test":     len(X_te),
        })
        mlflow.set_tag("problem_type", cfg.problem_type)
        mlflow.set_tag("feature_names", json.dumps(ms.feature_names))
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            _feature_ranking(pipe, X_te, y_te, ms.feature_names),
            "feature_ranking.json",
        )
        sig = infer_signature(X_tr, pipe.predict(X_tr))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=model_name,
            signature=sig,
            input_example=X_te[:3],
        )
        run_id = run.info.run_id

    return {"model_name": model_name, "run_id": run_id,
            "metrics": ho, "champion_metrics": champion_metrics}


# ── main entry point ──────────────────────────────────────────────────────────

def trigger(reason: str, consecutive_windows: int, mlflow_uri: str):
    cfg = load_config()
    ms  = cfg.multi_slot
    if ms is None:
        log.warning(f"Retrain trigger ignored: config '{cfg.name}' is not multi-slot")
        return

    log.info("━" * 55)
    log.info(f"  RETRAIN TRIGGER  reason={reason}  consec={consecutive_windows}")
    log.info("━" * 55)

    mlflow.set_tracking_uri(mlflow_uri)
    client    = MlflowClient(mlflow_uri)
    problem   = get_problem_type(cfg.problem_type)
    min_delta = _min_delta(cfg)

    # 1. Log trigger event
    mlflow.set_experiment("retraining-events")
    with mlflow.start_run(run_name=f"trigger-{reason}"):
        mlflow.log_params({"reason": reason, "consecutive_windows": consecutive_windows})
        mlflow.set_tag("event_type", "retrain_trigger")
    log.info("Trigger event logged ✓")

    df       = get_data_source(cfg.data_source).load().extra["df"]
    promoted = 0

    # 2. Per slot: train challenger, compare on the shared hold-out, decide
    for scenario_id, scenario_label in ms.scenarios.items():
        df_sc = df[df["Scenario"] == scenario_id].copy()

        for target_alias, target_col in ms.targets.items():
            mlflow.set_experiment(cfg.mlflow.experiment_name)
            try:
                res = _train_challenger(cfg, problem, df_sc,
                                        scenario_label, target_alias, target_col)
            except Exception:
                log.error(f"Challenger training failed for {scenario_label}-{target_alias}:\n"
                          f"{traceback.format_exc()}")
                continue

            model_name = res["model_name"]
            chal       = res["metrics"]
            champ      = res["champion_metrics"]
            chal_ver   = _version_for_run(client, model_name, res["run_id"])
            if chal_ver is None:
                log.error(f"No registered version found for challenger of {model_name}")
                continue

            gate_ok, failures = problem.quality_gate(chal, cfg.quality_gates)
            champ_r2 = champ["r2"] if champ else None
            # Strictly better: the challenger is deterministic, so a tie means
            # "same model again" and must not churn Production.
            improves = champ_r2 is None or chal["r2"] > champ_r2 + min_delta
            promote  = gate_ok and improves

            log.info(f"{model_name}: champion R²={champ_r2}  challenger v{chal_ver} "
                     f"R²={chal['r2']:.4f}  gate_ok={gate_ok}  improves={improves}")
            if not gate_ok:
                log.warning(f"  Challenger fails quality gate: {failures}")

            mlflow.set_experiment("retraining-events")
            with mlflow.start_run(run_name=f"promotion-{model_name}-v{chal_ver}"):
                mlflow.log_params({"model": model_name, "challenger_version": chal_ver,
                                   "reason": reason})
                mlflow.log_metric("challenger_r2", chal["r2"])
                mlflow.log_metric("champion_r2",   champ_r2 if champ_r2 is not None else 0.0)
                mlflow.log_metric("gate_passed",   int(gate_ok))
                mlflow.log_metric("promoted",      int(promote))
                mlflow.set_tag("event_type", "promotion_decision")

            if promote:
                log.info(f"✅  Promoting {model_name} v{chal_ver} → Production")
                client.transition_model_version_stage(
                    name=model_name, version=chal_ver,
                    stage="Production", archive_existing_versions=True,
                )
                promoted += 1
            else:
                log.info(f"⏭   {model_name} challenger not promoted — archiving")
                client.transition_model_version_stage(
                    name=model_name, version=chal_ver, stage="Archived",
                )

    if promoted:
        _reload_api()
    log.info(f"Retrain trigger complete ✓  ({promoted} slot(s) promoted)")
