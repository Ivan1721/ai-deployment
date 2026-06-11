"""
app.py  -  Inference API
------------------------
Serves the 8 HRI harvesting models (2 scenarios x 4 targets) from MLflow.
Accepts operational configuration and returns all 4 regression targets.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_STAGE  = os.environ.get("MODEL_STAGE", "Production")
# SECURITY FIX P0-2: CORS debe restringirse a dominios confiables
# Valores por defecto: solo localhost (desarrollo), cambiar en producción
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost,http://localhost:3000,http://localhost:8080").split(",")

SCENARIOS = {0: "HumanOnly", 1: "WithRobot"}
TARGETS = {
    "TotalRecollected": "TotalRecollectedCrops_crop_units",
    "CargoZoneProd":    "TotalProductionCargoZone_crop_units",
    "TotalWorkload":    "TotalHumanWorkload_kcal",
    "AvgProduction":    "AverageHumanProduction_crop_units",
}
ACTIVITY_VALUES = ["harv_ground", "harv_ladder", "harv_mixed", "harv_picker"]
FEATURE_NAMES   = ["Humans", "ROW_N", "RandomPosition", "Act_Ladder", "Act_Mixed", "Act_Picker"]

state: Dict = {"models": {}, "loaded_at": None}


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
    log.info(f"Models loaded: {sum(len(v) for v in loaded.values())}/8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield


app = FastAPI(
    title="HRI Harvesting Inference API",
    description="Predicts harvesting productivity and workload for Human-Only and Human-Robot scenarios.",
    version="2.0.0",
    lifespan=lifespan,
)
# <<<<<<< develop

# SECURITY FIX P0-2: CORS restricción a dominios confiables
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],           # Solo métodos necesarios
    allow_headers=["Content-Type", "Accept"],
    max_age=3600,                             # Cache de preflight por 1 hora
    allow_credentials=True,
)
# =======
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# >>>>>>> feature/dataset-update


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


@app.post("/predict", response_model=PredictResponse)
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

    return PredictResponse(
        scenario_label=scenario_label,
        total_recollected=preds.get("TotalRecollected", 0.0),
        cargo_zone_prod=preds.get("CargoZoneProd", 0.0),
        total_workload_kcal=preds.get("TotalWorkload", 0.0),
        avg_production=preds.get("AvgProduction", 0.0),
        model_stage=MODEL_STAGE,
        loaded_at=state["loaded_at"] or "",
    )


@app.post("/reload")
def reload_models():
    load_all_models(retries=3, delay=2)
    total = sum(len(v) for v in state["models"].values())
    if total == 0:
        raise HTTPException(503, "Reload failed -- no models loaded.")
    return {"status": "reloaded", "models_loaded": total}
