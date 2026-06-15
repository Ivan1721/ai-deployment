"""
train.py
────────
Entrena el modelo descrito por la configuración (MODEL_CONFIG), registra
métricas/parámetros/tags en MLFlow Tracking Server, guarda el modelo como
artefacto y lo promueve a "Production" en el Model Registry.
"""

import os
import json
import time
import logging

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from mlops_common import load_config
from mlops_common.data_sources import get_data_source
from mlops_common.models import get_model_strategy
from mlops_common.problem_types import get_problem_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
ARTIFACT_ROOT = os.environ.get("MLFLOW_ARTIFACT_ROOT", None)   # /mlflow/artifacts si está montado


def wait_for_mlflow(uri: str, retries: int = 15, delay: int = 4) -> None:
    import urllib.request
    for i in range(retries):
        try:
            urllib.request.urlopen(f"{uri}/", timeout=3)
            log.info("MLFlow server is up ✓")
            return
        except Exception:
            log.info(f"Waiting for MLFlow… attempt {i+1}/{retries}")
            time.sleep(delay)
    raise RuntimeError("MLFlow server did not respond in time.")


def main():
    cfg = load_config()
    wait_for_mlflow(TRACKING_URI)

    mlflow.set_tracking_uri(TRACKING_URI)

    # Si tenemos acceso local al filesystem de artefactos, usarlo directamente
    if ARTIFACT_ROOT:
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        # Sobreescribir artifact location del experimento al crearlo
        client = MlflowClient(tracking_uri=TRACKING_URI)
        try:
            exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
            if exp is None:
                client.create_experiment(cfg.mlflow.experiment_name, artifact_location=ARTIFACT_ROOT)
        except Exception as e:
            log.warning(f"Could not set artifact location: {e}")

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    bundle = get_data_source(cfg.data_source).load()
    strategy = get_model_strategy(cfg.model.family)
    problem = get_problem_type(cfg.problem_type)

    X, y = bundle.X, bundle.y
    split_kwargs = {"test_size": cfg.training.test_size, "random_state": 42}
    if cfg.training.stratify:
        split_kwargs["stratify"] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    log.info(
        f"Dataset: {X.shape[0]} samples · {X.shape[1]} features · "
        f"problem_type={cfg.problem_type} · model_family={cfg.model.family}"
    )

    with mlflow.start_run(run_name="rf-production-v1") as run:
        run_id = run.info.run_id
        log.info(f"MLFlow run_id: {run_id}")

        mlflow.log_params(cfg.model.params)
        mlflow.log_param("dataset", f"{cfg.data_source.type}:{cfg.data_source.name or cfg.data_source.path}")
        mlflow.log_param("test_size", cfg.training.test_size)

        model = strategy.build(cfg.model.params)
        model = strategy.fit(model, X_train, y_train)

        y_pred = strategy.predict(model, X_test)
        y_proba = strategy.predict_proba(model, X_test) if strategy.supports_proba else None

        metrics = problem.compute_metrics(y_test, y_pred, y_proba)

        cv_scoring = problem.primary_metric
        cv_scores = strategy.cv_score(model, X, y, cv=cfg.training.cv_folds, scoring=cv_scoring)
        metrics["cv_mean"] = float(cv_scores.mean())
        metrics["cv_std"] = float(cv_scores.std())

        mlflow.log_metrics(metrics)

        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        log.info(metrics_str)

        if bundle.target_names is not None:
            from sklearn.metrics import classification_report
            log.info("\n" + classification_report(y_test, y_pred, target_names=bundle.target_names))

        # Tags de esquema: permiten que inference-api y drift-detector sean
        # agnósticos del dataset, leyendo la forma de X/y desde el run.
        mlflow.set_tag("problem_type", cfg.problem_type)
        mlflow.set_tag("model_family", cfg.model.family)
        mlflow.set_tag("config_name", cfg.name)
        mlflow.set_tag("feature_names", json.dumps(bundle.feature_names))
        if bundle.target_names is not None:
            mlflow.set_tag("target_names", json.dumps(bundle.target_names))
        if bundle.target_name is not None:
            mlflow.set_tag("target_name", bundle.target_name)

        signature = infer_signature(X_train, model.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=cfg.mlflow.registered_model_name,
            signature=signature,
            input_example=X_test.iloc[:3],
        )
        log.info(f"Model URI: {model_info.model_uri}")

    # Promover a Production
    client   = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{cfg.mlflow.registered_model_name}'")
    latest   = sorted(versions, key=lambda v: int(v.version))[-1]

    client.transition_model_version_stage(
        name=cfg.mlflow.registered_model_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    log.info(f"Model '{cfg.mlflow.registered_model_name}' v{latest.version} → Production ✓")


if __name__ == "__main__":
    main()
