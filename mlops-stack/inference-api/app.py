"""
app.py  -  Inference API
------------------------
Two API surfaces:

  HRI-specific (backward compat):
    GET  /health  /info
    POST /predict           — takes HRI operational fields, returns 4 targets
    POST /reload            — hot-reloads the 8 HRI Production models

  Generic (Gap 1 + Gap 2 — multi-experiment):
    GET  /models            — list all models in Production across any experiment
    GET  /models/{name}     — metadata for one registered model
    POST /models/{name}/predict  — predict with any sklearn model in the registry;
                                   accepts a flat feature dict, caches model in memory
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

from mlops_common import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_STAGE  = os.environ.get("MODEL_STAGE", "Production")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost,http://localhost:3000").split(",")
API_KEY      = os.environ.get("API_KEY", "")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _check_api_key(key: str = Depends(_api_key_header)) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# Load HRI configuration
_cfg = load_config()
_ms  = _cfg.multi_slot

SCENARIOS       = _ms.scenarios       if _ms else {0: "HumanOnly", 1: "WithRobot"}
TARGETS         = _ms.targets         if _ms else {}
FEATURE_NAMES   = _ms.feature_names   if _ms else []
ACTIVITY_VALUES = ["harv_ground", "harv_ladder", "harv_mixed", "harv_picker"]

# HRI model state (8 pre-loaded models)
state: Dict = {"models": {}, "loaded_at": None}

# Generic model cache: "{name}/{stage}" → loaded sklearn model
_model_cache: Dict[str, Any] = {}

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total HTTP requests by endpoint and status code",
    ["endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "inference_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
MODELS_LOADED = Gauge(
    "inference_models_loaded",
    "Number of HRI models currently loaded (target: 8)",
)
PREDICTIONS_TOTAL = Counter(
    "inference_predictions_total",
    "HRI prediction requests by scenario and activity",
    ["scenario", "activity"],
)
GENERIC_PREDICTIONS_TOTAL = Counter(
    "inference_generic_predictions_total",
    "Generic model prediction requests by model name",
    ["model"],
)


# ── HRI helpers ───────────────────────────────────────────────────────────────

def _encode_activity(activity: str) -> Dict[str, int]:
    return {
        "Act_Ladder": int(activity == "harv_ladder"),
        "Act_Mixed":  int(activity == "harv_mixed"),
        "Act_Picker": int(activity == "harv_picker"),
    }


def load_all_models(retries: int = 20, delay: int = 6) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    loaded: Dict = {}
    for scenario_label in SCENARIOS.values():
        loaded[scenario_label] = {}
        for target_alias in TARGETS:
            model_name = f"hri-{scenario_label}-{target_alias}"
            model_uri  = f"models:/{model_name}/{MODEL_STAGE}"
            for attempt in range(1, retries + 1):
                try:
                    loaded[scenario_label][target_alias] = mlflow.sklearn.load_model(model_uri)
                    log.info(f"Loaded {model_name}")
                    break
                except Exception as e:
                    log.warning(f"  {model_name} attempt {attempt}/{retries}: {e}")
                    if attempt < retries:
                        time.sleep(delay)
            else:
                log.error(f"Failed to load {model_name} after {retries} attempts")

    state["models"]    = loaded
    state["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total = sum(len(v) for v in loaded.values())
    MODELS_LOADED.set(total)
    log.info(f"Models loaded: {total}/8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HRI Harvesting Inference API",
    description=(
        "Two API surfaces: HRI-specific endpoints (/predict, /reload) for the "
        "harvesting regression models, and generic endpoints (/models/*) for any "
        "sklearn model registered in MLflow."
    ),
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
    max_age=3600,
)

app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    if request.url.path in ("/metrics", "/metrics/"):
        return await call_next(request)
    start = time.time()
    response = await call_next(request)
    REQUEST_DURATION.labels(endpoint=request.url.path).observe(time.time() - start)
    REQUESTS_TOTAL.labels(endpoint=request.url.path, status=str(response.status_code)).inc()
    return response


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    scenario: int = Field(..., description="0=Human-Only, 1=Human-Robot", ge=0, le=1)
    workers:  int = Field(..., description="Number of human workers", ge=1, le=12)
    crop_row: int = Field(..., description="Crop row (1-3)", ge=1, le=3)
    rand_pos: int = Field(..., description="Random initial positions (0=Fixed, 1=Random)", ge=0, le=1)
    activity: str = Field(..., description="harv_ground | harv_ladder | harv_mixed | harv_picker")

    class Config:
        json_schema_extra = {
            "example": {"scenario": 0, "workers": 6, "crop_row": 2, "rand_pos": 0, "activity": "harv_ground"}
        }


class PredictResponse(BaseModel):
    scenario_label:      str
    total_recollected:   float
    cargo_zone_prod:     float
    total_workload_kcal: float
    avg_production:      float
    model_stage:         str
    loaded_at:           str


class GenericPredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Feature name → value mapping. Keys must match the model's training features.",
    )
    stage: str = Field("Production", description="MLflow model stage to use")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Humans": 6, "ROW_N": 2, "RandomPosition": 0,
                    "Act_Ladder": 0, "Act_Mixed": 0, "Act_Picker": 0
                },
                "stage": "Production",
            }
        }


class GenericPredictResponse(BaseModel):
    model:      str
    stage:      str
    prediction: Union[float, List[float]]


class ModelInfo(BaseModel):
    name:    str
    version: str
    stage:   str
    run_id:  str


# ── HRI-specific endpoints ────────────────────────────────────────────────────

@app.get("/health")
def health():
    total = sum(len(v) for v in state["models"].values())
    return {
        "status":        "ok" if total == 8 else "degraded",
        "models_loaded": total,
        "model_stage":   MODEL_STAGE,
        "loaded_at":     state["loaded_at"],
    }


@app.get("/info")
def info():
    return {
        "scenarios":   [{"id": k, "label": v} for k, v in SCENARIOS.items()],
        "targets":     list(TARGETS.keys()),
        "features":    FEATURE_NAMES,
        "activities":  ACTIVITY_VALUES,
        "model_stage": MODEL_STAGE,
    }


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(_check_api_key)])
def predict(req: PredictRequest):
    if req.activity not in ACTIVITY_VALUES:
        raise HTTPException(422, f"activity must be one of {ACTIVITY_VALUES}")

    scenario_label = SCENARIOS[req.scenario]
    models = state["models"].get(scenario_label, {})
    if not models:
        raise HTTPException(503, f"Models for scenario '{scenario_label}' not loaded yet.")

    enc = _encode_activity(req.activity)
    X   = pd.DataFrame([[
        req.workers, req.crop_row, req.rand_pos,
        enc["Act_Ladder"], enc["Act_Mixed"], enc["Act_Picker"],
    ]], columns=FEATURE_NAMES).values

    preds = {alias: round(float(model.predict(X)[0]), 4) for alias, model in models.items()}
    PREDICTIONS_TOTAL.labels(scenario=scenario_label, activity=req.activity).inc()

    return PredictResponse(
        scenario_label=scenario_label,
        total_recollected=preds.get("TotalRecollected", 0.0),
        cargo_zone_prod=preds.get("CargoZoneProd", 0.0),
        total_workload_kcal=preds.get("TotalWorkload", 0.0),
        avg_production=preds.get("AvgProduction", 0.0),
        model_stage=MODEL_STAGE,
        loaded_at=state["loaded_at"] or "",
    )


@app.post("/reload", dependencies=[Depends(_check_api_key)])
def reload_models():
    _model_cache.clear()
    load_all_models(retries=3, delay=2)
    total = sum(len(v) for v in state["models"].values())
    if total == 0:
        raise HTTPException(503, "Reload failed -- no models loaded.")
    return {"status": "reloaded", "models_loaded": total, "generic_cache_cleared": True}


# ── Generic model endpoints (Gap 1 + Gap 2) ───────────────────────────────────

@app.get("/models", response_model=List[ModelInfo], dependencies=[Depends(_check_api_key)])
def list_models():
    """List all models that have a Production version in the MLflow registry."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    result = []
    try:
        for rm in client.search_registered_models():
            versions = client.get_latest_versions(rm.name, stages=["Production"])
            if versions:
                v = versions[0]
                result.append(ModelInfo(
                    name=rm.name,
                    version=v.version,
                    stage=v.current_stage,
                    run_id=v.run_id,
                ))
    except Exception as e:
        raise HTTPException(503, f"MLflow unavailable: {e}")
    return result


@app.get("/models/{name}", response_model=ModelInfo, dependencies=[Depends(_check_api_key)])
def get_model(name: str, stage: str = "Production"):
    """Return metadata for a specific registered model."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(name, stages=[stage])
    except Exception as e:
        raise HTTPException(503, f"MLflow unavailable: {e}")
    if not versions:
        raise HTTPException(404, f"No '{stage}' version found for model '{name}'")
    v = versions[0]
    return ModelInfo(name=name, version=v.version, stage=v.current_stage, run_id=v.run_id)


@app.post(
    "/models/{name}/predict",
    response_model=GenericPredictResponse,
    dependencies=[Depends(_check_api_key)],
)
def generic_predict(name: str, req: GenericPredictRequest):
    """
    Predict with any sklearn model registered in MLflow.
    The model is loaded on first call and cached in memory for subsequent requests.
    Features must match the model's training schema exactly.
    """
    cache_key = f"{name}/{req.stage}"
    if cache_key not in _model_cache:
        mlflow.set_tracking_uri(TRACKING_URI)
        try:
            _model_cache[cache_key] = mlflow.sklearn.load_model(
                f"models:/{name}/{req.stage}"
            )
            log.info(f"Cached generic model {cache_key}")
        except Exception as e:
            raise HTTPException(503, f"Could not load model '{name}' (stage={req.stage}): {e}")

    model = _model_cache[cache_key]
    X = pd.DataFrame([req.features])
    try:
        raw = model.predict(X)
        prediction: Union[float, List[float]] = (
            float(raw[0]) if len(raw) == 1 else [float(v) for v in raw]
        )
    except Exception as e:
        raise HTTPException(422, f"Prediction failed for model '{name}': {e}")

    GENERIC_PREDICTIONS_TOTAL.labels(model=name).inc()
    return GenericPredictResponse(model=name, stage=req.stage, prediction=prediction)
