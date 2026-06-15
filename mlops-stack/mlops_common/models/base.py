"""
base.py
───────
Contrato Strategy para una familia de algoritmo (p.ej. Random Forest,
Logistic/Linear Regression): cómo se construye, entrena, predice y evalúa,
sin que el resto del stack conozca la clase concreta de sklearn.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd


class ModelStrategy(ABC):
    """Una familia de algoritmo para un tipo de problema (clasificación o regresión)."""

    @abstractmethod
    def build(self, params: Dict[str, Any]):
        """Crea un estimador sklearn sin entrenar."""

    def fit(self, model, X: pd.DataFrame, y: pd.Series):
        model.fit(X, y)
        return model

    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        return model.predict(X)

    def predict_proba(self, model, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} no soporta predict_proba")

    def cv_score(self, model, X: pd.DataFrame, y: pd.Series, cv: int, scoring: str) -> np.ndarray:
        from sklearn.model_selection import cross_val_score
        return cross_val_score(model, X, y, cv=cv, scoring=scoring)

    @property
    def supports_proba(self) -> bool:
        return False
