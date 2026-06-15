"""
hri_csv.py
──────────
Fuente de datos para el HRI Agricultural Harvesting Dataset.
Realiza el preprocesamiento específico del dominio (one-hot de MainActivity)
y expone el DataFrame completo en bundle.extra["df"] para que el trainer
pueda iterar sobre escenarios y targets.
"""

import pandas as pd

from .base import DataSource, DatasetBundle

FEATURE_NAMES = ["Humans", "ROW_N", "RandomPosition", "Act_Ladder", "Act_Mixed", "Act_Picker"]


class HriCsvDataSource(DataSource):
    def __init__(self, cfg):
        self.path = cfg.path

    def load(self) -> DatasetBundle:
        df = pd.read_csv(self.path)

        dummies = pd.get_dummies(df["MainActivity"], prefix="Act", drop_first=True)
        dummies = dummies.rename(columns={
            "Act_harv_ladder": "Act_Ladder",
            "Act_harv_mixed":  "Act_Mixed",
            "Act_harv_picker": "Act_Picker",
        })
        for col in ["Act_Ladder", "Act_Mixed", "Act_Picker"]:
            if col not in dummies.columns:
                dummies[col] = 0

        df = pd.concat([df, dummies[["Act_Ladder", "Act_Mixed", "Act_Picker"]]], axis=1)

        X = df[FEATURE_NAMES]

        return DatasetBundle(
            X=X,
            y=None,
            feature_names=FEATURE_NAMES,
            target_name="multi-target-regression",
            extra={"df": df},
        )
