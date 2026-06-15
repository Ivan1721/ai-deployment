"""
mlops_common
────────────
Paquete compartido por model-trainer, inference-api, drift-detector y tests.

Provee las abstracciones (patrón Strategy + Registry) que desacoplan el
stack de un dataset y un algoritmo concretos:

  - mlops_common.config         → carga el archivo de configuración (YAML)
                                   de un caso de uso (ExperimentConfig)
  - mlops_common.data_sources    → de dónde vienen X, y, feature/target names
  - mlops_common.models          → cómo se construye/entrena/evalúa el modelo
  - mlops_common.problem_types   → métricas, quality gates, forma de /predict
                                    y concept drift según clasificación/regresión
"""

from .config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
