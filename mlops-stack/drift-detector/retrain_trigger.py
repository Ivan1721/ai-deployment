"""
retrain_trigger.py  ─  Champion/Challenger automático
"""

import os, time, logging, traceback
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature

log = logging.getLogger(__name__)

MODEL_NAME    = os.environ.get("MODEL_NAME", "iris-classifier")
INFERENCE_URL = os.environ.get("INFERENCE_API_URL", "http://inference-api:8000")
MIN_DELTA     = float(os.environ.get("CHALLENGER_MIN_IMPROVEMENT", "0.0"))


def _champion_accuracy(client: MlflowClient) -> float | None:
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return float(run.data.metrics.get("accuracy", 0.0))
    except Exception:
        return None


def _challenger_version_accuracy(client: MlflowClient) -> tuple:
    """Retorna (version_str, accuracy) de la versión más nueva en stage 'None'."""
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        candidates = [v for v in versions if v.current_stage == "None"]
        if not candidates:
            return None, None
        newest = sorted(candidates, key=lambda v: int(v.version))[-1]
        run    = client.get_run(newest.run_id)
        acc    = float(run.data.metrics.get("accuracy", 0.0))
        return newest.version, acc
    except Exception:
        log.error(traceback.format_exc())
        return None, None


def _reload_api():
    try:
        import urllib.request, json as _json
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


def _train_challenger(mlflow_uri: str) -> bool:
    """Entrena un nuevo modelo y lo registra como Challenger."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("iris-production")

    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=int(time.time()) % 9999
    )
    params = {"n_estimators": 200, "max_depth": 8, "min_samples_split": 4,
              "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 42}

    with mlflow.start_run(run_name="challenger-auto"):
        mlflow.log_params(params)
        mlflow.set_tag("triggered_by", "drift_detector")

        clf    = RandomForestClassifier(**params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        f1     = f1_score(y_te, y_pred, average="weighted")

        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})
        log.info(f"Challenger trained: accuracy={acc:.4f}  f1={f1:.4f}")

        sig = infer_signature(X_tr, clf.predict(X_tr))
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=sig,
        )
    return True


def trigger(reason: str, consecutive_windows: int, mlflow_uri: str):
    log.info("━"*55)
    log.info(f"  RETRAIN TRIGGER  reason={reason}  consec={consecutive_windows}")
    log.info("━"*55)

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(mlflow_uri)

    # 1. Registrar evento
    mlflow.set_experiment("retraining-events")
    with mlflow.start_run(run_name=f"trigger-{reason}"):
        mlflow.log_params({"reason": reason, "consecutive_windows": consecutive_windows})
        mlflow.set_tag("event_type", "retrain_trigger")
    log.info("Trigger event logged to MLFlow ✓")

    # 2. Guardar accuracy del champion
    champ_acc = _champion_accuracy(client)
    log.info(f"Champion accuracy before retrain: {champ_acc}")

    # 3. Entrenar challenger
    try:
        _train_challenger(mlflow_uri)
    except Exception:
        log.error(f"Challenger training failed:\n{traceback.format_exc()}")
        return

    time.sleep(3)  # dar tiempo al Registry

    # 4. Leer métricas del challenger
    chal_ver, chal_acc = _challenger_version_accuracy(client)
    if chal_ver is None:
        log.error("Challenger not found in Registry — aborting promotion")
        return

    log.info(f"Challenger v{chal_ver} accuracy: {chal_acc:.4f}")
    log.info(f"Champion   accuracy:            {champ_acc}")

    # 5. Decisión de promoción
    promote = (champ_acc is None) or (chal_acc is not None and
                                       chal_acc >= champ_acc + MIN_DELTA)

    mlflow.set_experiment("retraining-events")
    with mlflow.start_run(run_name=f"promotion-v{chal_ver}"):
        mlflow.log_params({"challenger_version": chal_ver, "reason": reason})
        mlflow.log_metric("challenger_accuracy", chal_acc or 0.0)
        mlflow.log_metric("champion_accuracy",   champ_acc or 0.0)
        mlflow.log_metric("promoted",            int(promote))
        mlflow.set_tag("event_type", "promotion_decision")

    if promote:
        log.info(f"✅ Promoting Challenger v{chal_ver} → Production")
        client.transition_model_version_stage(
            name=MODEL_NAME, version=chal_ver,
            stage="Production", archive_existing_versions=True,
        )
        _reload_api()
        log.info(f"Model v{chal_ver} now serving in Production ✓")
    else:
        log.info(f"⏭  Challenger v{chal_ver} does not improve — archiving")
        client.transition_model_version_stage(
            name=MODEL_NAME, version=chal_ver, stage="Archived"
        )
