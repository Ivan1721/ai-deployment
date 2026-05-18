"""
app.py  ─  Inference API
────────────────────────
Sirve en producción el modelo registrado en MLFlow Model Registry.
Carga el modelo directamente desde el filesystem compartido.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME   = os.environ.get("MODEL_NAME",  "iris-classifier")
MODEL_STAGE  = os.environ.get("MODEL_STAGE", "Production")

TARGET_NAMES  = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = [
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)",
]

state: dict = {"model": None, "model_version": None, "loaded_at": None}


def load_model(retries: int = 20, delay: int = 6):
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    for attempt in range(1, retries + 1):
        try:
            log.info(f"Loading '{model_uri}' … attempt {attempt}/{retries}")
            model = mlflow.sklearn.load_model(model_uri)

            from mlflow import MlflowClient
            client   = MlflowClient(TRACKING_URI)
            versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            version  = versions[0].version if versions else "?"

            state["model"]         = model
            state["model_version"] = version
            state["loaded_at"]     = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            log.info(f"Model loaded ✓  version={version}")
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

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    instances: List[List[float]] = Field(
        ..., example=[[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
    )

class Prediction(BaseModel):
    class_id: int; class_name: str; probability: float

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_name: str; model_version: str; model_stage: str


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
    return {"model_name": MODEL_NAME, "model_stage": MODEL_STAGE,
            "model_version": state["model_version"],
            "features": FEATURE_NAMES, "classes": TARGET_NAMES}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if state["model"] is None:
        raise HTTPException(503, "Model not loaded yet.")
    if len(request.instances) == 0:
        raise HTTPException(422, "instances list must not be empty.")
    for i, row in enumerate(request.instances):
        if len(row) != 4:
            raise HTTPException(422, f"Instance {i}: expected 4 features, got {len(row)}.")
    X = pd.DataFrame(request.instances, columns=FEATURE_NAMES)
    class_ids     = state["model"].predict(X)
    probabilities = state["model"].predict_proba(X)
    predictions   = [
        Prediction(
            class_id=int(cid),
            class_name=TARGET_NAMES[int(cid)],
            probability=round(float(probabilities[i][int(cid)]), 4),
        )
        for i, cid in enumerate(class_ids)
    ]
    return PredictResponse(
        predictions=predictions,
        model_name=MODEL_NAME,
        model_version=state["model_version"] or "?",
        model_stage=MODEL_STAGE,
    )

@app.post("/reload")
def reload_model():
    load_model(retries=3, delay=2)
    if state["model"] is None:
        raise HTTPException(503, "Reload failed.")
    return {"status": "reloaded", "model_version": state["model_version"]}
