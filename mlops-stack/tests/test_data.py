"""
test_data.py
────────────
Nivel 1: Data Tests

Valida el dataset de entrenamiento ANTES de que comience el training.
Si estos tests fallan, el pipeline no debe proceder.

Categorías (Eken et al., 2025 — Sección 3.2.8 Quality Assurance):
  - Completitud: no hay valores nulos ni filas vacías
  - Consistencia: tipos correctos, rangos válidos por feature
  - Distribución: balance de clases, estadísticas dentro de rangos esperados
  - Schema: columnas correctas, orden correcto
  - Integridad: sin duplicados exactos, tamaño mínimo

Fundamento:
  "Data quality involves assessing the scientific quality of data with
   domain experts, checking data completeness, checking consistency in
   data format and structure." (PS62, PS24, PS151)
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


# ── fixture compartido ─────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def iris_data():
    """Carga el dataset que usará el trainer. Fixture compartido por todos los tests."""
    iris = load_iris(as_frame=True)
    return iris.data, iris.target, iris.feature_names, iris.target_names


# ══════════════════════════════════════════════════════════════════
# COMPLETITUD
# ══════════════════════════════════════════════════════════════════

class TestCompleteness:

    def test_no_null_values_in_features(self, iris_data):
        """El dataset no debe contener valores nulos en ningún feature."""
        X, _, _, _ = iris_data
        null_counts = X.isnull().sum()
        assert null_counts.sum() == 0, (
            f"Se encontraron valores nulos:\n{null_counts[null_counts > 0]}"
        )

    def test_no_null_values_in_target(self, iris_data):
        """El target no debe contener valores nulos."""
        _, y, _, _ = iris_data
        assert y.isnull().sum() == 0, "Target contiene valores nulos"

    def test_minimum_sample_size(self, iris_data):
        """El dataset debe tener al menos 100 muestras para entrenar con cross-validation."""
        X, _, _, _ = iris_data
        assert len(X) >= 100, (
            f"Dataset demasiado pequeño: {len(X)} muestras (mínimo 100)"
        )

    def test_features_and_target_same_length(self, iris_data):
        """Features y target deben tener el mismo número de filas."""
        X, y, _, _ = iris_data
        assert len(X) == len(y), (
            f"Desalineación: X tiene {len(X)} filas pero y tiene {len(y)}"
        )


# ══════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════

class TestSchema:

    EXPECTED_FEATURES = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    def test_correct_number_of_features(self, iris_data):
        """El dataset debe tener exactamente 4 features."""
        X, _, _, _ = iris_data
        assert X.shape[1] == 4, (
            f"Se esperaban 4 features, se encontraron {X.shape[1]}"
        )

    def test_correct_feature_names(self, iris_data):
        """Los nombres de los features deben coincidir exactamente."""
        X, _, _, _ = iris_data
        assert list(X.columns) == self.EXPECTED_FEATURES, (
            f"Nombres incorrectos:\n  esperados: {self.EXPECTED_FEATURES}\n"
            f"  encontrados: {list(X.columns)}"
        )

    def test_feature_names_in_correct_order(self, iris_data):
        """El orden de los features importa para la inferencia — debe ser consistente."""
        X, _, _, _ = iris_data
        for i, (actual, expected) in enumerate(zip(X.columns, self.EXPECTED_FEATURES)):
            assert actual == expected, (
                f"Feature {i}: se esperaba '{expected}', se encontró '{actual}'"
            )

    def test_numeric_dtypes(self, iris_data):
        """Todos los features deben ser numéricos (float64)."""
        X, _, _, _ = iris_data
        for col in X.columns:
            assert np.issubdtype(X[col].dtype, np.number), (
                f"Feature '{col}' tiene tipo no numérico: {X[col].dtype}"
            )

    def test_target_is_integer(self, iris_data):
        """El target debe ser entero (clases discretas)."""
        _, y, _, _ = iris_data
        assert np.issubdtype(y.dtype, np.integer), (
            f"Target tiene tipo {y.dtype}, se esperaba entero"
        )

    def test_correct_number_of_classes(self, iris_data):
        """Debe haber exactamente 3 clases."""
        _, y, _, _ = iris_data
        n_classes = len(np.unique(y))
        assert n_classes == 3, (
            f"Se esperaban 3 clases, se encontraron {n_classes}: {np.unique(y)}"
        )

    def test_class_labels_are_0_1_2(self, iris_data):
        """Las etiquetas de clase deben ser 0, 1, 2 (sin gaps)."""
        _, y, _, _ = iris_data
        unique = sorted(np.unique(y).tolist())
        assert unique == [0, 1, 2], (
            f"Etiquetas de clase inesperadas: {unique}"
        )


# ══════════════════════════════════════════════════════════════════
# RANGOS Y CONSISTENCIA
# ══════════════════════════════════════════════════════════════════

class TestRangesAndConsistency:
    """
    Valida que los valores estén dentro de rangos biológicamente razonables para Iris.
    En un sistema real, estos rangos vendrían del conocimiento de dominio
    o de estadísticas históricas del dataset de referencia.
    """

    # Rangos esperados por feature (min, max) según el dominio del problema
    EXPECTED_RANGES = {
        "sepal length (cm)": (4.0, 8.0),
        "sepal width (cm)":  (1.5, 5.0),
        "petal length (cm)": (0.5, 7.5),
        "petal width (cm)":  (0.0, 3.0),
    }

    def test_sepal_length_range(self, iris_data):
        X, _, _, _ = iris_data
        col = "sepal length (cm)"
        mn, mx = self.EXPECTED_RANGES[col]
        assert X[col].min() >= mn and X[col].max() <= mx, (
            f"{col}: rango [{X[col].min():.2f}, {X[col].max():.2f}] "
            f"fuera de límites [{mn}, {mx}]"
        )

    def test_sepal_width_range(self, iris_data):
        X, _, _, _ = iris_data
        col = "sepal width (cm)"
        mn, mx = self.EXPECTED_RANGES[col]
        assert X[col].min() >= mn and X[col].max() <= mx, (
            f"{col}: rango [{X[col].min():.2f}, {X[col].max():.2f}] "
            f"fuera de límites [{mn}, {mx}]"
        )

    def test_petal_length_range(self, iris_data):
        X, _, _, _ = iris_data
        col = "petal length (cm)"
        mn, mx = self.EXPECTED_RANGES[col]
        assert X[col].min() >= mn and X[col].max() <= mx, (
            f"{col}: rango [{X[col].min():.2f}, {X[col].max():.2f}] "
            f"fuera de límites [{mn}, {mx}]"
        )

    def test_petal_width_range(self, iris_data):
        X, _, _, _ = iris_data
        col = "petal width (cm)"
        mn, mx = self.EXPECTED_RANGES[col]
        assert X[col].min() >= mn and X[col].max() <= mx, (
            f"{col}: rango [{X[col].min():.2f}, {X[col].max():.2f}] "
            f"fuera de límites [{mn}, {mx}]"
        )

    def test_no_negative_measurements(self, iris_data):
        """Las medidas biológicas no pueden ser negativas."""
        X, _, _, _ = iris_data
        for col in X.columns:
            assert (X[col] >= 0).all(), (
                f"Feature '{col}' contiene valores negativos: "
                f"{X[col][X[col] < 0].values}"
            )

    def test_no_infinite_values(self, iris_data):
        """No debe haber valores infinitos."""
        X, _, _, _ = iris_data
        assert not np.isinf(X.values).any(), "Se encontraron valores infinitos"

    def test_no_duplicate_rows(self, iris_data):
        """
        No deben existir filas completamente duplicadas.
        Algunos duplicados son aceptables en datasets reales,
        pero más del 10% indica un problema de recolección.
        """
        X, _, _, _ = iris_data
        dup_rate = X.duplicated().sum() / len(X)
        assert dup_rate < 0.10, (
            f"Tasa de duplicados muy alta: {dup_rate:.1%} ({X.duplicated().sum()} filas)"
        )


# ══════════════════════════════════════════════════════════════════
# DISTRIBUCIÓN Y BALANCE
# ══════════════════════════════════════════════════════════════════

class TestDistribution:

    def test_class_balance(self, iris_data):
        """
        Las clases deben estar razonablemente balanceadas.
        Si una clase tiene menos del 10% de las muestras, el modelo
        puede tener sesgo severo sin que accuracy lo refleje.
        """
        _, y, _, _ = iris_data
        counts = pd.Series(y).value_counts()
        total  = len(y)
        for cls, count in counts.items():
            pct = count / total
            assert pct >= 0.10, (
                f"Clase {cls} tiene solo {pct:.1%} de las muestras "
                f"({count}/{total}) — posible clase minoritaria problemática"
            )

    def test_feature_variance_not_zero(self, iris_data):
        """
        Ningún feature debe tener varianza cero (feature constante).
        Un feature constante no aporta información al modelo.
        """
        X, _, _, _ = iris_data
        for col in X.columns:
            assert X[col].var() > 0, (
                f"Feature '{col}' tiene varianza cero — es constante"
            )

    def test_features_not_perfectly_correlated(self, iris_data):
        """
        Detecta multicolinealidad perfecta (correlación = ±1.0).
        Features perfectamente correlacionados son redundantes.
        """
        X, _, _, _ = iris_data
        corr = X.corr().abs()
        # Ignorar la diagonal (cada feature correlaciona con sí mismo = 1.0)
        np.fill_diagonal(corr.values, 0)
        max_corr = corr.max().max()
        assert max_corr < 0.999, (
            f"Multicolinealidad perfecta detectada: correlación máxima = {max_corr:.4f}"
        )

    def test_train_test_split_stratification(self, iris_data):
        """
        Verifica que una partición estratificada produce proporciones similares.
        Esto asegura que el split del trainer es representativo.
        """
        from sklearn.model_selection import train_test_split
        _, y, _, _ = iris_data
        _, _, y_train, y_test = train_test_split(
            np.zeros(len(y)), y, test_size=0.25, stratify=y, random_state=42
        )
        # La proporción de cada clase en train y test no debe diferir más de 3%
        for cls in np.unique(y):
            train_pct = (y_train == cls).mean()
            test_pct  = (y_test  == cls).mean()
            assert abs(train_pct - test_pct) < 0.03, (
                f"Clase {cls}: train={train_pct:.3f} vs test={test_pct:.3f} "
                f"— estratificación deficiente"
            )
