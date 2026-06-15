"""
sklearn_dataset.py
──────────────────
Fuente de datos basada en sklearn.datasets.load_<name> (iris, wine, diabetes,
breast_cancer, ...). El tipo de problema se infiere del propio dataset:
si expone `target_names`, es clasificación; si no, es regresión.
"""

import sklearn.datasets

from .base import DataSource, DatasetBundle


class SklearnDatasetSource(DataSource):
    def __init__(self, cfg):
        self.name = cfg.name

    def load(self) -> DatasetBundle:
        loader = getattr(sklearn.datasets, f"load_{self.name}")
        bunch = loader(as_frame=True)

        X = bunch.data
        y = bunch.target
        feature_names = list(bunch.feature_names)

        target_names = getattr(bunch, "target_names", None)
        if target_names is not None:
            return DatasetBundle(
                X=X, y=y,
                feature_names=feature_names,
                target_names=[str(t) for t in target_names],
            )

        target_name = y.name if getattr(y, "name", None) else "target"
        return DatasetBundle(
            X=X, y=y,
            feature_names=feature_names,
            target_name=target_name,
        )
