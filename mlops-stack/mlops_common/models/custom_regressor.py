"""
custom_regressor.py
───────────────────
KernelWeightedRegressor: regresor no paramétrico de Nadaraya-Watson.

Predice como promedio ponderado de los targets de entrenamiento donde los pesos
son proporcionales a la similitud con el punto de consulta, medida por un kernel
Gaussiano sobre la distancia euclidiana.

    ŷ(x) = Σ_i K(x, x_i) * y_i  /  Σ_i K(x, x_i)
    K(x, x_i) = exp(-‖x - x_i‖² / (2h²))

No existe en scikit-learn ni en ninguna librería estándar de ML en Python.
Es un ejemplo representativo de algoritmo ad-hoc 100% en NumPy integrado al
stack vía la abstracción ModelStrategy.

Hereda BaseEstimator + RegressorMixin para compatibilidad con:
  - cross_val_score (usado por ModelStrategy.cv_score)
  - mlflow.sklearn.log_model
  - GridSearchCV / RandomizedSearchCV (opcional, futuro)
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class KernelWeightedRegressor(BaseEstimator, RegressorMixin):
    """Regresor Nadaraya-Watson con kernel Gaussiano.

    Parámetros
    ----------
    bandwidth : float
        Ancho de banda h del kernel. Mayor valor = más suavizado global.
        Sensible a la escala de los features; normalizar X antes de entrenar
        si las variables tienen escalas muy distintas.
    """

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # diff: (n_test, n_train, n_features)
        diff = X[:, None, :] - self.X_train_[None, :, :]
        sq_dist = (diff ** 2).sum(axis=2)                          # (n_test, n_train)
        weights = np.exp(-sq_dist / (2.0 * self.bandwidth ** 2))   # kernel Gaussiano
        denom = weights.sum(axis=1, keepdims=True)
        # Evitar división por cero cuando un punto está muy lejos de todos los de entrenamiento
        denom = np.where(denom == 0, 1e-10, denom)
        normalized = weights / denom                               # (n_test, n_train)
        return (normalized * self.y_train_[None, :]).sum(axis=1)   # (n_test,)
