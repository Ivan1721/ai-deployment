"""
csv_source.py
─────────────
Fuente de datos genérica desde un archivo CSV: todas las columnas excepto
`target_column` son features; `target_column` es el target. Si el target es
numérico con muchos valores distintos se trata como regresión; en caso
contrario, como clasificación (sus valores únicos se usan como target_names).
"""

import pandas as pd

from .base import DataSource, DatasetBundle

_MAX_CLASSES_HEURISTIC = 20


class CsvDataSource(DataSource):
    def __init__(self, cfg):
        self.path = cfg.path
        self.target_column = cfg.target_column

    def load(self) -> DatasetBundle:
        df = pd.read_csv(self.path)
        if self.target_column not in df.columns:
            raise ValueError(f"target_column '{self.target_column}' not found in {self.path}")

        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])
        feature_names = list(X.columns)

        is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > _MAX_CLASSES_HEURISTIC
        if is_regression:
            return DatasetBundle(X=X, y=y, feature_names=feature_names, target_name=self.target_column)

        target_names = sorted(str(v) for v in y.unique())
        return DatasetBundle(X=X, y=y, feature_names=feature_names, target_names=target_names)
