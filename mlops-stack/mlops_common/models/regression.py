"""
regression.py
─────────────
Estrategias de modelo para problemas de regresión.
Incluye los 12 candidatos del torneo HRI + wrappers Pipeline con StandardScaler
(requerido para SVR y MLP).
"""

from typing import Any, Dict

from sklearn.ensemble import (GradientBoostingRegressor, ExtraTreesRegressor,
                               RandomForestRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .base import ModelStrategy


def _pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


class LinearRegressionStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return _pipeline(LinearRegression(**params))


class RidgeStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return _pipeline(Ridge(**params))


class LassoStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return _pipeline(Lasso(**params))


class ElasticNetStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return _pipeline(ElasticNet(**params))


class SvrStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return _pipeline(SVR(**params))


class ExtraTreesRegressorStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return _pipeline(ExtraTreesRegressor(random_state=42, **params))


class RandomForestRegressorStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return _pipeline(RandomForestRegressor(random_state=42, **params))


class GradientBoostingRegressorStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return _pipeline(GradientBoostingRegressor(random_state=42, **params))


class MlpRegressorStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        return _pipeline(MLPRegressor(max_iter=500, random_state=42, **params))


# ── Optional heavy packages ───────────────────────────────────────────────────

class XgboostRegressorStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise RuntimeError("xgboost is not installed")
        return _pipeline(XGBRegressor(random_state=42, verbosity=0, **params))


class LightgbmRegressorStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise RuntimeError("lightgbm is not installed")
        return _pipeline(LGBMRegressor(random_state=42, verbose=-1, **params))


class CatboostRegressorStrategy(ModelStrategy):
    def build(self, params: Dict[str, Any]):
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise RuntimeError("catboost is not installed")
        return _pipeline(CatBoostRegressor(random_state=42, verbose=0, **params))
