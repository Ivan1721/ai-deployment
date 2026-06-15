"""
app.py  ─  Inference API
────────────────────────
Sirve en producción el modelo registrado en MLFlow Model Registry.
El esquema (features, clases/target, problem_type, model_family) se lee de
los tags del run de MLFlow que generó el modelo (ver model-trainer/train.py);
si el run no tiene esos tags (modelos legacy), se usa MODEL_CONFIG como
fallback.
"""

import os
import json
import time
import types
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mlops_common import load_config
from mlops_common.data_sources import get_data_source
from mlops_common.problem_types import get_problem_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME   = os.environ.get("MODEL_NAME",  "iris-classifier")
MODEL_STAGE  = os.environ.get("MODEL_STAGE", "Production")
# SECURITY FIX P0-2: CORS debe restringirse a dominios confiables
# Valores por defecto: solo localhost (desarrollo), cambiar en producción
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost,http://localhost:3000,http://localhost:8080").split(",")

cfg = load_config()

state: dict = {
    "model": None,
    "model_version": None,
    "loaded_at": None,
    "feature_names": None,
    "target_names": None,
    "target_name": None,
    "problem_type": None,
    "model_family": None,
}


def _fallback_schema() -> dict:
    """Esquema derivado de MODEL_CONFIG, para modelos sin tags (legacy)."""
    bundle = get_data_source(cfg.data_source).load()
    return {
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "target_name": bundle.target_name,
        "problem_type": cfg.problem_type,
        "model_family": cfg.model.family,
    }


def _schema_from_run(client: MlflowClient, version) -> dict:
    run = client.get_run(version.run_id)
    tags = run.data.tags

    if "feature_names" not in tags or "problem_type" not in tags:
        log.warning("Run has no schema tags — falling back to MODEL_CONFIG")
        return _fallback_schema()

    target_names = json.loads(tags["target_names"]) if "target_names" in tags else None

    return {
        "feature_names": json.loads(tags["feature_names"]),
        "target_names": target_names,
        "target_name": tags.get("target_name"),
        "problem_type": tags["problem_type"],
        "model_family": tags.get("model_family", cfg.model.family),
    }


def load_model(retries: int = 20, delay: int = 6):
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    for attempt in range(1, retries + 1):
        try:
            log.info(f"Loading '{model_uri}' … attempt {attempt}/{retries}")
            model = mlflow.sklearn.load_model(model_uri)

            client   = MlflowClient(TRACKING_URI)
            versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            version  = versions[0] if versions else None

            schema = _schema_from_run(client, version) if version else _fallback_schema()

            state["model"]         = model
            state["model_version"] = version.version if version else "?"
            state["loaded_at"]     = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            state["feature_names"] = schema["feature_names"]
            state["target_names"]  = schema["target_names"]
            state["target_name"]   = schema["target_name"]
            state["problem_type"]  = schema["problem_type"]
            state["model_family"]  = schema["model_family"]
            log.info(
                f"Model loaded ✓  version={state['model_version']}  "
                f"problem_type={state['problem_type']}"
            )
            return
        except Exception as e:
            log.warning(f"Could not load model: {e}")
            if attempt < retries:
                time.sleep(delay)

    log.error("Failed to load model after all retries.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="MLOps Inference API",
    description="Inference API backed by MLFlow Model Registry",
    version="1.0.0",
    lifespan=lifespan,
)

# SECURITY FIX P0-2: CORS restricción a dominios confiables
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],           # Solo métodos necesarios
    allow_headers=["Content-Type", "Accept"],
    max_age=3600,                             # Cache de preflight por 1 hora
    allow_credentials=True,
)


class PredictRequest(BaseModel):
    instances: List[List[float]] = Field(
        ..., example=[[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
    )


class Prediction(BaseModel):
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    probability: Optional[float] = None
    value: Optional[float] = None


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_name: str
    model_version: str
    model_stage: str
    problem_type: str


@app.get("/health")
def health():
    return {
        "status":        "ok" if state["model"] is not None else "degraded",
        "model_name":    MODEL_NAME,
        "model_version": state["model_version"],
        "model_stage":   MODEL_STAGE,
        "loaded_at":     state["loaded_at"],
    }

@app.get("/info")
def info():
    return {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_version": state["model_version"],
        "problem_type": state["problem_type"],
        "model_family": state["model_family"],
        "features": state["feature_names"],
        "classes": state["target_names"],
        "target": state["target_name"],
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if state["model"] is None:
        raise HTTPException(503, "Model not loaded yet.")
    if len(request.instances) == 0:
        raise HTTPException(422, "instances list must not be empty.")

    feature_names = state["feature_names"]
    for i, row in enumerate(request.instances):
        if len(row) != len(feature_names):
            raise HTTPException(422, f"Instance {i}: expected {len(feature_names)} features, got {len(row)}.")

    X = pd.DataFrame(request.instances, columns=feature_names)
    y_pred = state["model"].predict(X)

    y_proba = None
    if state["target_names"] is not None and hasattr(state["model"], "predict_proba"):
        y_proba = state["model"].predict_proba(X)

    problem = get_problem_type(state["problem_type"])
    bundle = types.SimpleNamespace(target_names=state["target_names"], target_name=state["target_name"])
    predictions = problem.build_predictions(y_pred, y_proba, bundle)

    return PredictResponse(
        predictions=[Prediction(**p) for p in predictions],
        model_name=MODEL_NAME,
        model_version=state["model_version"] or "?",
        model_stage=MODEL_STAGE,
        problem_type=state["problem_type"],
    )

@app.post("/reload")
def reload_model():
    load_model(retries=3, delay=2)
    if state["model"] is None:
        raise HTTPException(503, "Reload failed.")
    return {"status": "reloaded", "model_version": state["model_version"]}
