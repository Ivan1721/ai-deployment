"""
regression.py
──────────────
Estrategias de modelo para problemas de regresión. Ninguna implementa
predict_proba — el comportamiento por defecto de ModelStrategy ya lanza
NotImplementedError, que es lo correcto para regresión.
"""

from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .base import ModelStrategy


class RandomForestRegressorStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return RandomForestRegressor(**params)


class LinearRegressionStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return LinearRegression(**params)
