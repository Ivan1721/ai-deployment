from .base import ProblemType
from .classification import ClassificationProblem
from .regression import RegressionProblem

PROBLEM_TYPE_REGISTRY = {
    "classification": ClassificationProblem(),
    "regression": RegressionProblem(),
}


def get_problem_type(name: str) -> ProblemType:
    try:
        return PROBLEM_TYPE_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown problem type: {name!r}")


__all__ = ["ProblemType", "get_problem_type", "PROBLEM_TYPE_REGISTRY"]
