"""
classification.py
──────────────────
Estrategias de modelo para problemas de clasificación.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .base import ModelStrategy


class RandomForestClassifierStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return RandomForestClassifier(**params)

    def predict_proba(self, model, X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(X)

    @property
    def supports_proba(self) -> bool:
        return True


class LogisticRegressionStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return LogisticRegression(**params)

    def predict_proba(self, model, X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(X)

    @property
    def supports_proba(self) -> bool:
        return True
