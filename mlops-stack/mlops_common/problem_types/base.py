"""
base.py
───────
Contrato Strategy para un tipo de problema (clasificación, regresión):
qué métricas se calculan, cómo se evalúan los quality gates, qué forma
tiene la respuesta de /predict y cómo se detecta concept drift.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ProblemType(ABC):
    name: str
    primary_metric: str  # métrica "mayor-es-mejor" para comparar champion/challenger

    @abstractmethod
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        ...

    @abstractmethod
    def quality_gate(
        self, metrics: Dict[str, float], gates_cfg: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Retorna (ok, lista_de_fallos)."""

    @abstractmethod
    def build_predictions(
        self, y_pred: np.ndarray, y_proba: Optional[np.ndarray], bundle
    ) -> List[Dict[str, Any]]:
        """Forma de cada elemento de `predictions` en la respuesta de /predict."""

    @abstractmethod
    def concept_drift(
        self, reference: Any, current: Any, p_value_threshold: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Detecta drift en la distribución de salidas del modelo (predicciones)."""
