"""
config.py
─────────
Carga y representa la configuración de un caso de uso ("experiment config")
desde un archivo YAML. Este archivo es el único punto de entrada que conecta
una fuente de datos, una familia de modelo y un tipo de problema con los
servicios del stack (trainer, inference-api, drift-detector, tests).

Ruta por defecto: variable de entorno MODEL_CONFIG, o
/app/configs/iris-classifier.yaml si no está definida.
"""

import os
import dataclasses
from typing import Any, Dict, Optional

import yaml


@dataclasses.dataclass
class DataSourceConfig:
    type: str
    name: Optional[str] = None
    path: Optional[str] = None
    target_column: Optional[str] = None


@dataclasses.dataclass
class ModelConfig:
    family: str
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrainingConfig:
    test_size: float = 0.25
    stratify: bool = True
    cv_folds: int = 5


@dataclasses.dataclass
class MlflowConfig:
    experiment_name: str
    registered_model_name: str


@dataclasses.dataclass
class PerformanceDriftConfig:
    enabled: bool = True
    effect_size_threshold: float = 0.05
    p_value_threshold: float = 0.05
    consecutive_windows: int = 2


@dataclasses.dataclass
class DriftConfig:
    data_drift_method: str = "ks"
    data_drift_p_value: float = 0.05
    concept_drift_p_value: float = 0.05
    performance: PerformanceDriftConfig = dataclasses.field(default_factory=PerformanceDriftConfig)


@dataclasses.dataclass
class RetrainConfig:
    challenger_min_improvement: float = 0.0


@dataclasses.dataclass
class ExperimentConfig:
    name: str
    problem_type: str  # "classification" | "regression"
    data_source: DataSourceConfig
    model: ModelConfig
    mlflow: MlflowConfig
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    quality_gates: Dict[str, Any] = dataclasses.field(default_factory=dict)
    drift: DriftConfig = dataclasses.field(default_factory=DriftConfig)
    retrain: RetrainConfig = dataclasses.field(default_factory=RetrainConfig)


DEFAULT_CONFIG_PATH = "/app/configs/iris-classifier.yaml"


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    path = path or os.environ.get("MODEL_CONFIG", DEFAULT_CONFIG_PATH)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ds_raw = raw["data_source"]
    data_source = DataSourceConfig(
        type=ds_raw["type"],
        name=ds_raw.get("name"),
        path=ds_raw.get("path"),
        target_column=ds_raw.get("target_column"),
    )

    model = ModelConfig(
        family=raw["model"]["family"],
        params=raw["model"].get("params", {}),
    )

    training_raw = raw.get("training", {})
    training = TrainingConfig(
        test_size=training_raw.get("test_size", 0.25),
        stratify=training_raw.get("stratify", True),
        cv_folds=training_raw.get("cv_folds", 5),
    )

    mlflow_cfg = MlflowConfig(
        experiment_name=raw["mlflow"]["experiment_name"],
        registered_model_name=raw["mlflow"]["registered_model_name"],
    )

    drift_raw = raw.get("drift", {})
    perf_raw = drift_raw.get("performance", {})
    drift = DriftConfig(
        data_drift_method=drift_raw.get("data_drift_method", "ks"),
        data_drift_p_value=drift_raw.get("data_drift_p_value", 0.05),
        concept_drift_p_value=drift_raw.get("concept_drift_p_value", 0.05),
        performance=PerformanceDriftConfig(
            enabled=perf_raw.get("enabled", True),
            effect_size_threshold=perf_raw.get("effect_size_threshold", 0.05),
            p_value_threshold=perf_raw.get("p_value_threshold", 0.05),
            consecutive_windows=perf_raw.get("consecutive_windows", 2),
        ),
    )

    retrain_raw = raw.get("retrain", {})
    retrain = RetrainConfig(
        challenger_min_improvement=retrain_raw.get("challenger_min_improvement", 0.0),
    )

    return ExperimentConfig(
        name=raw["name"],
        problem_type=raw["problem_type"],
        data_source=data_source,
        model=model,
        mlflow=mlflow_cfg,
        training=training,
        quality_gates=raw.get("quality_gates", {}),
        drift=drift,
        retrain=retrain,
    )
