"""
train.py
────────
Entrena un clasificador RandomForest sobre el dataset Iris,
registra métricas y parámetros en MLFlow Tracking Server,
guarda el modelo como artefacto y lo promueve a "Production"
en el Model Registry.
"""

import os
import time
import logging
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TRACKING_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
ARTIFACT_ROOT  = os.environ.get("MLFLOW_ARTIFACT_ROOT", None)   # /mlflow/artifacts si está montado
MODEL_NAME     = "iris-classifier"
EXPERIMENT     = "iris-production"

PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
}


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
    wait_for_mlflow(TRACKING_URI)

    mlflow.set_tracking_uri(TRACKING_URI)

    # Si tenemos acceso local al filesystem de artefactos, usarlo directamente
    if ARTIFACT_ROOT:
        mlflow.set_experiment(EXPERIMENT)
        # Sobreescribir artifact location del experimento al crearlo
        client = MlflowClient(tracking_uri=TRACKING_URI)
        try:
            exp = client.get_experiment_by_name(EXPERIMENT)
            if exp is None:
                client.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)
        except Exception as e:
            log.warning(f"Could not set artifact location: {e}")

    mlflow.set_experiment(EXPERIMENT)

    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    feature_names = list(iris.feature_names)
    target_names  = list(iris.target_names)

    log.info(f"Dataset: {X.shape[0]} samples · {X.shape[1]} features · {len(target_names)} classes")

    with mlflow.start_run(run_name="rf-production-v1") as run:
        run_id = run.info.run_id
        log.info(f"MLFlow run_id: {run_id}")

        mlflow.log_params(PARAMS)
        mlflow.log_param("dataset", "sklearn.datasets.load_iris")
        mlflow.log_param("test_size", 0.25)

        clf = RandomForestClassifier(**PARAMS)
        clf.fit(X_train, y_train)

        y_pred        = clf.predict(X_test)
        accuracy      = accuracy_score(y_test, y_pred)
        f1            = f1_score(y_test, y_pred, average="weighted")
        precision     = precision_score(y_test, y_pred, average="weighted")
        recall        = recall_score(y_test, y_pred, average="weighted")
        cv_scores     = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

        mlflow.log_metrics({
            "accuracy":    accuracy,
            "f1_weighted": f1,
            "precision":   precision,
            "recall":      recall,
            "cv_mean":     cv_scores.mean(),
            "cv_std":      cv_scores.std(),
        })

        log.info(f"accuracy={accuracy:.4f}  f1={f1:.4f}  cv_mean={cv_scores.mean():.4f}")
        log.info("\n" + classification_report(y_test, y_pred, target_names=target_names))

        from mlflow.models import infer_signature
        signature  = infer_signature(X_train, clf.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=X_test.iloc[:3],
        )
        log.info(f"Model URI: {model_info.model_uri}")

    # Promover a Production
    client   = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest   = sorted(versions, key=lambda v: int(v.version))[-1]

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    log.info(f"Model '{MODEL_NAME}' v{latest.version} → Production ✓")


if __name__ == "__main__":
    main()


def run_quality_gate(clf, X_test, y_test,
                     min_accuracy: float = 0.90,
                     min_f1: float = 0.90) -> bool:
    """
    Quality gate inline: evalúa el modelo antes de promoverlo.
    Retorna True si pasa, False si debe bloquearse la promoción.
    Fundamento: PS111, PS62 en Eken et al. (2025).
    """
    from sklearn.metrics import accuracy_score, f1_score
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")

    log.info(f"Quality Gate — accuracy={accuracy:.4f} (min={min_accuracy})  "
             f"f1={f1:.4f} (min={min_f1})")

    passed = accuracy >= min_accuracy and f1 >= min_f1
    if not passed:
        log.error("QUALITY GATE FAILED — model will NOT be promoted to Production")
    else:
        log.info("Quality Gate PASSED ✓")
    return passed
