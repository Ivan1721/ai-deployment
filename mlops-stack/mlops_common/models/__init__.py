from .base import ModelStrategy
from .classification import LogisticRegressionStrategy, RandomForestClassifierStrategy
from .regression import (
    LinearRegressionStrategy,
    RandomForestRegressorStrategy,
    KernelWeightedRegressorStrategy,
)

MODEL_REGISTRY = {
    "random_forest_classifier":   RandomForestClassifierStrategy,
    "logistic_regression":        LogisticRegressionStrategy,
    "random_forest_regressor":    RandomForestRegressorStrategy,
    "linear_regression":          LinearRegressionStrategy,
    "kernel_weighted_regressor":  KernelWeightedRegressorStrategy,
}


def get_model_strategy(family: str) -> ModelStrategy:
    try:
        cls = MODEL_REGISTRY[family]
    except KeyError:
        raise ValueError(f"Unknown model family: {family!r}")
    return cls()


__all__ = ["ModelStrategy", "get_model_strategy", "MODEL_REGISTRY"]
