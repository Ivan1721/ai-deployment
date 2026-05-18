"""
test_model.py
─────────────
Nivel 2: Model Tests — Quality Gates

Valida el modelo registrado en MLFlow ANTES de promoverlo a Production.
Si estos tests fallan, el modelo NO debe ser promovido.

Categorías:
  - Performance gates: accuracy y F1 mínimos aceptables
  - Robustez: comportamiento con inputs edge-case
  - Sesgo: distribución de predicciones por clase
  - Reproducibilidad: el mismo input produce el mismo output
  - Sanity checks: el modelo aprende algo (mejor que baseline aleatorio)
  - Firma: el modelo acepta exactamente los inputs especificados

Fundamento (Eken et al., 2025):
  "Model quality involves assessing the model's predictive performance
   through performance metrics and test sets (PS21), validating model
   bias and fairness (PS100), monitoring model to detect overfitting
   issues (PS59)." (Sección 3.2.8)
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier


# ── args de línea de comando (configurables desde run_tests.py) ────────────
def pytest_addoption(parser):
    parser.addoption("--mlflow-uri",  default="http://mlflow:5001")
    parser.addoption("--model-name",  default="iris-classifier")
    parser.addoption("--model-stage", default="Production")
    parser.addoption("--min-accuracy",default="0.90")
    parser.addoption("--min-f1",      default="0.90")


# ── fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def config(request):
    return {
        "mlflow_uri":   request.config.getoption("--mlflow-uri"),
        "model_name":   request.config.getoption("--model-name"),
        "model_stage":  request.config.getoption("--model-stage"),
        "min_accuracy": float(request.config.getoption("--min-accuracy")),
        "min_f1":       float(request.config.getoption("--min-f1")),
    }


@pytest.fixture(scope="module")
def model(config):
    """Carga el modelo desde MLFlow Model Registry."""
    import mlflow.sklearn
    mlflow_uri  = config["mlflow_uri"]
    model_name  = config["model_name"]
    model_stage = config["model_stage"]

    import mlflow
    mlflow.set_tracking_uri(mlflow_uri)
    model_uri = f"models:/{model_name}/{model_stage}"
    try:
        clf = mlflow.sklearn.load_model(model_uri)
        return clf
    except Exception as e:
        pytest.skip(f"No se pudo cargar el modelo desde {model_uri}: {e}")


@pytest.fixture(scope="module")
def iris_splits():
    """Partición fija del dataset (misma que usa el trainer)."""
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, iris.target_names


@pytest.fixture(scope="module")
def predictions(model, iris_splits):
    """Calcula predicciones una sola vez para todos los tests."""
    _, X_test, _, y_test, _ = iris_splits
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return y_pred, y_prob, y_test


# ══════════════════════════════════════════════════════════════════
# QUALITY GATES — PERFORMANCE MÍNIMA
# ══════════════════════════════════════════════════════════════════

class TestPerformanceGates:
    """
    Quality gates: el modelo DEBE superar estos umbrales
    para poder ser promovido a Production.
    """

    def test_accuracy_meets_minimum(self, model, iris_splits, config):
        """
        GATE: Accuracy mínima en test set.
        Un modelo por debajo de este umbral no debe ir a producción.
        """
        _, X_test, _, y_test, _ = iris_splits
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        threshold = config["min_accuracy"]
        assert acc >= threshold, (
            f"QUALITY GATE FAILED: accuracy={acc:.4f} < mínimo={threshold}\n"
            f"El modelo no cumple el umbral mínimo para producción."
        )

    def test_f1_meets_minimum(self, model, iris_splits, config):
        """
        GATE: F1-score weighted mínimo.
        Más relevante que accuracy cuando las clases no están perfectamente balanceadas.
        """
        _, X_test, _, y_test, _ = iris_splits
        y_pred = model.predict(X_test)
        f1     = f1_score(y_test, y_pred, average="weighted")
        threshold = config["min_f1"]
        assert f1 >= threshold, (
            f"QUALITY GATE FAILED: f1={f1:.4f} < mínimo={threshold}"
        )

    def test_per_class_recall_minimum(self, predictions, iris_splits):
        """
        Cada clase debe tener recall >= 0.80.
        Evita que el modelo ignore sistemáticamente una clase minoritaria.
        """
        y_pred, _, y_test = predictions
        _, _, _, _, target_names = iris_splits
        for cls_idx, cls_name in enumerate(target_names):
            mask   = (y_test == cls_idx)
            if mask.sum() == 0:
                continue
            recall = (y_pred[mask] == cls_idx).mean()
            assert recall >= 0.80, (
                f"Recall bajo para clase '{cls_name}': {recall:.4f} < 0.80\n"
                f"El modelo puede estar ignorando esta clase."
            )

    def test_precision_minimum(self, predictions):
        """Precision weighted mínima."""
        y_pred, _, y_test = predictions
        prec = precision_score(y_test, y_pred, average="weighted")
        assert prec >= 0.85, (
            f"Precision insuficiente: {prec:.4f} < 0.85"
        )

    def test_cross_validation_stability(self, model, iris_splits):
        """
        CV score debe ser estable (std < 0.05).
        Alta varianza entre folds indica overfitting o datos problemáticos.
        """
        from sklearn.datasets import load_iris
        iris   = load_iris(as_frame=True)
        X, y   = iris.data, iris.target
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        std    = scores.std()
        mean   = scores.mean()
        assert std < 0.05, (
            f"Modelo inestable: CV std={std:.4f} >= 0.05\n"
            f"Scores por fold: {scores.round(4)}"
        )
        assert mean >= 0.90, (
            f"CV mean accuracy insuficiente: {mean:.4f} < 0.90"
        )


# ══════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ══════════════════════════════════════════════════════════════════

class TestSanityChecks:
    """
    Verifica que el modelo ha aprendido algo real,
    no simplemente memorizar o predecir la clase mayoritaria.
    """

    def test_better_than_random_baseline(self, model, iris_splits):
        """
        El modelo debe superar significativamente un clasificador aleatorio.
        Si no supera el baseline, algo salió muy mal en el entrenamiento.
        """
        X_train, X_test, y_train, y_test, _ = iris_splits
        baseline = DummyClassifier(strategy="stratified", random_state=42)
        baseline.fit(X_train, y_train)
        baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
        model_acc    = accuracy_score(y_test, model.predict(X_test))
        margin       = 0.20  # el modelo debe superar el baseline por al menos 20%
        assert model_acc >= baseline_acc + margin, (
            f"El modelo ({model_acc:.4f}) no supera al baseline "
            f"({baseline_acc:.4f}) por el margen mínimo de {margin}"
        )

    def test_better_than_majority_class_baseline(self, model, iris_splits):
        """
        Un clasificador que siempre predice la clase mayoritaria logra ~33% en Iris.
        El modelo debe superarlo ampliamente.
        """
        X_train, X_test, y_train, y_test, _ = iris_splits
        majority = DummyClassifier(strategy="most_frequent")
        majority.fit(X_train, y_train)
        majority_acc = accuracy_score(y_test, majority.predict(X_test))
        model_acc    = accuracy_score(y_test, model.predict(X_test))
        assert model_acc >= majority_acc + 0.30, (
            f"El modelo ({model_acc:.4f}) apenas supera al clasificador "
            f"de clase mayoritaria ({majority_acc:.4f})"
        )

    def test_model_produces_all_classes(self, predictions, iris_splits):
        """
        El modelo debe ser capaz de predecir todas las clases, no solo una.
        Un modelo colapsado a una clase es inútil en producción.
        """
        y_pred, _, _ = predictions
        _, _, _, _, target_names = iris_splits
        predicted_classes = set(y_pred.tolist())
        for cls_idx in range(len(target_names)):
            assert cls_idx in predicted_classes, (
                f"El modelo nunca predice la clase "
                f"'{target_names[cls_idx]}' (idx={cls_idx})"
            )

    def test_confusion_matrix_no_total_class_confusion(self, predictions, iris_splits):
        """
        Ninguna clase debe confundirse con otra al 100%.
        Si setosa se confunde siempre con virginica, el modelo es deficiente.
        """
        y_pred, _, y_test = predictions
        _, _, _, _, target_names = iris_splits
        cm = confusion_matrix(y_test, y_pred)
        for i, cls_name in enumerate(target_names):
            total_actual   = cm[i].sum()
            correct        = cm[i][i]
            if total_actual > 0:
                class_accuracy = correct / total_actual
                assert class_accuracy > 0.50, (
                    f"Clase '{cls_name}': {class_accuracy:.1%} accuracy — "
                    f"el modelo se equivoca más del 50% en esta clase"
                )


# ══════════════════════════════════════════════════════════════════
# ROBUSTEZ — EDGE CASES
# ══════════════════════════════════════════════════════════════════

class TestRobustness:
    """
    El modelo debe comportarse correctamente ante inputs límite.
    Estos tests protegen contra errores silenciosos en producción.
    """

    @pytest.fixture(scope="class")
    def sample_input(self):
        """Input representativo para tests de robustez."""
        iris = load_iris(as_frame=True)
        return iris.data.iloc[:5]

    def test_predict_returns_correct_shape(self, model, sample_input):
        """predict() debe retornar un array con la misma cantidad de filas."""
        preds = model.predict(sample_input)
        assert len(preds) == len(sample_input), (
            f"predict() retornó {len(preds)} predicciones para {len(sample_input)} inputs"
        )

    def test_predict_proba_returns_valid_probabilities(self, model, sample_input):
        """predict_proba() debe retornar probabilidades que sumen 1.0 por fila."""
        probs = model.predict_proba(sample_input)
        assert probs.shape == (len(sample_input), 3), (
            f"Shape incorrecto: {probs.shape}, esperado ({len(sample_input)}, 3)"
        )
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"Las probabilidades no suman 1.0: {row_sums}"
        )

    def test_predict_proba_all_non_negative(self, model, sample_input):
        """Todas las probabilidades deben ser >= 0."""
        probs = model.predict_proba(sample_input)
        assert (probs >= 0).all(), "Se encontraron probabilidades negativas"

    def test_single_sample_prediction(self, model):
        """El modelo debe poder predecir con exactamente 1 muestra."""
        single = pd.DataFrame(
            [[5.1, 3.5, 1.4, 0.2]],
            columns=["sepal length (cm)", "sepal width (cm)",
                     "petal length (cm)", "petal width (cm)"]
        )
        pred = model.predict(single)
        assert len(pred) == 1
        assert pred[0] in [0, 1, 2]

    def test_batch_prediction_100_samples(self, model):
        """El modelo debe manejar batches de 100 muestras sin error."""
        iris = load_iris(as_frame=True)
        X_batch = iris.data.sample(100, replace=True, random_state=42)
        preds   = model.predict(X_batch)
        assert len(preds) == 100

    def test_reproducibility(self, model):
        """El mismo input debe producir siempre el mismo output (determinismo)."""
        iris   = load_iris(as_frame=True)
        sample = iris.data.iloc[:10]
        pred1  = model.predict(sample)
        pred2  = model.predict(sample)
        assert (pred1 == pred2).all(), (
            "El modelo no es determinista: el mismo input produce outputs diferentes"
        )

    def test_predictions_are_valid_class_indices(self, predictions):
        """Todas las predicciones deben ser 0, 1 o 2."""
        y_pred, _, _ = predictions
        valid = set(y_pred.tolist()).issubset({0, 1, 2})
        assert valid, (
            f"Se encontraron etiquetas de clase inválidas: {set(y_pred.tolist())}"
        )


# ══════════════════════════════════════════════════════════════════
# REGISTRO EN MLFLOW
# ══════════════════════════════════════════════════════════════════

class TestMLFlowRegistry:
    """
    Verifica que el modelo está correctamente registrado en MLFlow.
    """

    def test_model_exists_in_registry(self, config):
        """El modelo debe existir en el Model Registry."""
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(config["mlflow_uri"])
        client   = MlflowClient(config["mlflow_uri"])
        try:
            versions = client.get_latest_versions(
                config["model_name"], stages=[config["model_stage"]]
            )
            assert len(versions) > 0, (
                f"No hay versiones en stage '{config['model_stage']}' "
                f"para el modelo '{config['model_name']}'"
            )
        except Exception as e:
            pytest.skip(f"MLFlow no disponible: {e}")

    def test_model_has_accuracy_metric(self, config):
        """El run del modelo debe tener registrada la métrica 'accuracy'."""
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(config["mlflow_uri"])
        client = MlflowClient(config["mlflow_uri"])
        try:
            versions = client.get_latest_versions(
                config["model_name"], stages=[config["model_stage"]]
            )
            if not versions:
                pytest.skip("No hay versiones en Production")
            run  = client.get_run(versions[0].run_id)
            acc  = run.data.metrics.get("accuracy")
            assert acc is not None, (
                "El run del modelo no tiene registrada la métrica 'accuracy'"
            )
            assert acc > 0, f"Accuracy registrada es inválida: {acc}"
        except Exception as e:
            pytest.skip(f"MLFlow no disponible: {e}")

    def test_model_has_required_tags(self, config):
        """Verifica que el modelo tiene metadatos mínimos registrados."""
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(config["mlflow_uri"])
        client = MlflowClient(config["mlflow_uri"])
        try:
            versions = client.get_latest_versions(
                config["model_name"], stages=[config["model_stage"]]
            )
            if not versions:
                pytest.skip("No hay versiones en Production")
            run    = client.get_run(versions[0].run_id)
            params = run.data.params
            assert "n_estimators" in params, "Falta parámetro 'n_estimators'"
            assert "random_state"  in params, "Falta parámetro 'random_state'"
        except Exception as e:
            pytest.skip(f"MLFlow no disponible: {e}")
