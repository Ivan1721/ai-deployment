"""
retrain_trigger.py  ─  Champion/Challenger automático
"""

import os
import time
import logging
import traceback

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from mlops_common import load_config
from mlops_common.data_sources import get_data_source
from mlops_common.models import get_model_strategy
from mlops_common.problem_types import get_problem_type

log = logging.getLogger(__name__)

INFERENCE_URL = os.environ.get("INFERENCE_API_URL", "http://inference-api:8000")
MIN_DELTA     = float(os.environ.get("CHALLENGER_MIN_IMPROVEMENT", "0.0"))


def _champion_metric(client: MlflowClient, cfg, problem):
    try:
        versions = client.get_latest_versions(cfg.mlflow.registered_model_name, stages=["Production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return float(run.data.metrics.get(problem.primary_metric, 0.0))
    except Exception:
        return None


def _challenger_version_metric(client: MlflowClient, cfg, problem) -> tuple:
    """Retorna (version_str, metric) de la versión más nueva en stage 'None'."""
    try:
        versions = client.search_model_versions(f"name='{cfg.mlflow.registered_model_name}'")
        candidates = [v for v in versions if v.current_stage == "None"]
        if not candidates:
            return None, None
        newest = sorted(candidates, key=lambda v: int(v.version))[-1]
        run    = client.get_run(newest.run_id)
        metric = float(run.data.metrics.get(problem.primary_metric, 0.0))
        return newest.version, metric
    except Exception:
        log.error(traceback.format_exc())
        return None, None


def _reload_api():
    try:
        import urllib.request
        req  = urllib.request.Request(
            f"{INFERENCE_URL}/reload",
            data=b"",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        log.info(f"API reload response: {resp.status}")
    except Exception as e:
        log.warning(f"Could not reload API: {e}")


def _train_challenger(mlflow_uri: str, cfg, bundle, strategy, problem) -> bool:
    """Entrena un nuevo modelo y lo registra como Challenger."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    split_kwargs = {"test_size": cfg.training.test_size, "random_state": int(time.time()) % 9999}
    if cfg.training.stratify:
        split_kwargs["stratify"] = bundle.y
    X_tr, X_te, y_tr, y_te = train_test_split(bundle.X, bundle.y, **split_kwargs)

    with mlflow.start_run(run_name="challenger-auto"):
        mlflow.log_params(cfg.model.params)
        mlflow.set_tag("triggered_by", "drift_detector")

        model = strategy.build(cfg.model.params)
        model = strategy.fit(model, X_tr, y_tr)

        y_pred = strategy.predict(model, X_te)
        y_proba = strategy.predict_proba(model, X_te) if strategy.supports_proba else None

        metrics = problem.compute_metrics(y_te, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        log.info("Challenger trained: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        sig = infer_signature(X_tr, model.predict(X_tr))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=cfg.mlflow.registered_model_name,
            signature=sig,
        )
    return True


def trigger(reason: str, consecutive_windows: int, mlflow_uri: str):
    cfg = load_config()
    bundle = get_data_source(cfg.data_source).load()
    strategy = get_model_strategy(cfg.model.family)
    problem = get_problem_type(cfg.problem_type)

    log.info("━" * 55)
    log.info(f"  RETRAIN TRIGGER  reason={reason}  consec={consecutive_windows}")
    log.info("━" * 55)

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(mlflow_uri)

    # 1. Registrar evento
    mlflow.set_experiment("retraining-events")
    with mlflow.start_run(run_name=f"trigger-{reason}"):
        mlflow.log_params({"reason": reason, "consecutive_windows": consecutive_windows})
        mlflow.set_tag("event_type", "retrain_trigger")
    log.info("Trigger event logged to MLFlow ✓")

    # 2. Guardar métrica del champion (problem.primary_metric: "mayor es mejor")
    champ_metric = _champion_metric(client, cfg, problem)
    log.info(f"Champion {problem.primary_metric} before retrain: {champ_metric}")

    # 3. Entrenar challenger
    try:
        _train_challenger(mlflow_uri, cfg, bundle, strategy, problem)
    except Exception:
        log.error(f"Challenger training failed:\n{traceback.format_exc()}")
        return

    time.sleep(3)  # dar tiempo al Registry

    # 4. Leer métricas del challenger
    chal_ver, chal_metric = _challenger_version_metric(client, cfg, problem)
    if chal_ver is None:
        log.error("Challenger not found in Registry — aborting promotion")
        return

    log.info(f"Challenger v{chal_ver} {problem.primary_metric}: {chal_metric:.4f}")
    log.info(f"Champion   {problem.primary_metric}:            {champ_metric}")

    # 5. Decisión de promoción
    promote = (champ_metric is None) or (chal_metric is not None and
                                          chal_metric >= champ_metric + MIN_DELTA)

    mlflow.set_experiment("retraining-events")
    with mlflow.start_run(run_name=f"promotion-v{chal_ver}"):
        mlflow.log_params({"challenger_version": chal_ver, "reason": reason})
        mlflow.log_metric(f"challenger_{problem.primary_metric}", chal_metric or 0.0)
        mlflow.log_metric(f"champion_{problem.primary_metric}",   champ_metric or 0.0)
        mlflow.log_metric("promoted",            int(promote))
        mlflow.set_tag("event_type", "promotion_decision")

    if promote:
        log.info(f"✅ Promoting Challenger v{chal_ver} → Production")
        client.transition_model_version_stage(
            name=cfg.mlflow.registered_model_name, version=chal_ver,
            stage="Production", archive_existing_versions=True,
        )
        _reload_api()
        log.info(f"Model v{chal_ver} now serving in Production ✓")
    else:
        log.info(f"⏭  Challenger v{chal_ver} does not improve — archiving")
        client.transition_model_version_stage(
            name=cfg.mlflow.registered_model_name, version=chal_ver, stage="Archived"
        )
