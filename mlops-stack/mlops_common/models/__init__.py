from .base import ModelStrategy
from .classification import LogisticRegressionStrategy, RandomForestClassifierStrategy
from .regression import (
    LinearRegressionStrategy, RidgeStrategy, LassoStrategy, ElasticNetStrategy,
    SvrStrategy, ExtraTreesRegressorStrategy, RandomForestRegressorStrategy,
    GradientBoostingRegressorStrategy, MlpRegressorStrategy,
    XgboostRegressorStrategy, LightgbmRegressorStrategy, CatboostRegressorStrategy,
)

MODEL_REGISTRY = {
    # Classification
    "random_forest_classifier": RandomForestClassifierStrategy,
    "logistic_regression":      LogisticRegressionStrategy,
    # Regression — single model
    "linear_regression":        LinearRegressionStrategy,
    "ridge":                    RidgeStrategy,
    "lasso":                    LassoStrategy,
    "elastic_net":              ElasticNetStrategy,
    "svr":                      SvrStrategy,
    "extra_trees_regressor":    ExtraTreesRegressorStrategy,
    "random_forest_regressor":  RandomForestRegressorStrategy,
    "gradient_boosting_regressor": GradientBoostingRegressorStrategy,
    "mlp_regressor":            MlpRegressorStrategy,
    "xgboost_regressor":        XgboostRegressorStrategy,
    "lightgbm_regressor":       LightgbmRegressorStrategy,
    "catboost_regressor":       CatboostRegressorStrategy,
}


def get_model_strategy(family: str) -> ModelStrategy:
    try:
        cls = MODEL_REGISTRY[family]
    except KeyError:
        raise ValueError(f"Unknown model family: {family!r}")
    return cls()


__all__ = ["ModelStrategy", "get_model_strategy", "MODEL_REGISTRY"]
