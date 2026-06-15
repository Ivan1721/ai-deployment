"""
base.py
───────
Contrato de fuente de datos: cada implementación concreta sabe cómo obtener
X, y, nombres de features y nombres de target/clases, y los expone con la
misma forma (DatasetBundle) sin importar el origen real de los datos.
"""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclasses.dataclass
class DatasetBundle:
    """Resultado uniforme de cargar un dataset, sin importar su origen."""

    X: pd.DataFrame
    y: pd.Series                                   # None para multi-target
    feature_names: List[str]
    target_names: Optional[List[str]] = None       # clasificación: nombres de clase
    target_name: Optional[str] = None              # regresión: nombre de la variable objetivo
    extra: Optional[Dict[str, Any]] = None         # metadata adicional (ej. DataFrame completo con escenarios)


class DataSource(ABC):
    """Una fuente de datos para entrenamiento, referencia de drift y tests."""

    @abstractmethod
    def load(self) -> DatasetBundle:
        ...
