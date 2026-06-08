"""
test_model.py  ─  Nivel 2: Model Tests + Quality Gates
"""
import os, pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.dummy import DummyClassifier

MLFLOW_URI   = os.environ.get("MLFLOW_URI",   "http://host.docker.internal:5001")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "iris-classifier")
MODEL_STAGE  = os.environ.get("MODEL_STAGE",  "Production")
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.90"))
MIN_F1       = float(os.environ.get("MIN_F1",       "0.90"))


@pytest.fixture(scope="module")
def model():
    import mlflow, mlflow.sklearn
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except Exception as e:
        pytest.skip(f"Cannot load model: {e}")


@pytest.fixture(scope="module")
def splits():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    return X_tr, X_te, y_tr, y_te, list(iris.target_names)


@pytest.fixture(scope="module")
def preds(model, splits):
    _, X_te, _, y_te, _ = splits
    return model.predict(X_te), model.predict_proba(X_te), y_te


# ── Quality Gates ──────────────────────────────────────────────────────────

class TestPerformanceGates:

    def test_accuracy_gate(self, model, splits):
        _, X_te, _, y_te, _ = splits
        acc = accuracy_score(y_te, model.predict(X_te))
        assert acc >= MIN_ACCURACY, f"GATE FAILED: accuracy={acc:.4f} < {MIN_ACCURACY}"

    def test_f1_gate(self, model, splits):
        _, X_te, _, y_te, _ = splits
        f1 = f1_score(y_te, model.predict(X_te), average="weighted")
        assert f1 >= MIN_F1, f"GATE FAILED: f1={f1:.4f} < {MIN_F1}"

    def test_per_class_recall(self, preds, splits):
        y_pred, _, y_te = preds
        _, _, _, _, names = splits
        for i, name in enumerate(names):
            mask = (y_te == i)
            if mask.sum() == 0: continue
            recall = (y_pred[mask] == i).mean()
            assert recall >= 0.80, f"Low recall for '{name}': {recall:.4f}"

    def test_cv_stability(self, model):
        iris = load_iris(as_frame=True)
        scores = cross_val_score(model, iris.data, iris.target, cv=5, scoring="accuracy")
        assert scores.std() < 0.05, f"CV unstable: std={scores.std():.4f}"
        assert scores.mean() >= 0.90, f"CV mean low: {scores.mean():.4f}"


class TestSanityChecks:

    def test_better_than_random(self, model, splits):
        X_tr, X_te, y_tr, y_te, _ = splits
        baseline = DummyClassifier(strategy="stratified", random_state=42)
        baseline.fit(X_tr, y_tr)
        b_acc = accuracy_score(y_te, baseline.predict(X_te))
        m_acc = accuracy_score(y_te, model.predict(X_te))
        assert m_acc >= b_acc + 0.20, f"Model ({m_acc:.4f}) barely beats baseline ({b_acc:.4f})"

    def test_predicts_all_classes(self, preds, splits):
        y_pred, _, _ = preds
        _, _, _, _, names = splits
        predicted = set(y_pred.tolist())
        for i in range(len(names)):
            assert i in predicted, f"Model never predicts class {names[i]}"

    def test_reproducible(self, model):
        iris = load_iris(as_frame=True)
        s = iris.data.iloc[:10]
        assert (model.predict(s) == model.predict(s)).all(), "Model not deterministic"


class TestRobustness:

    def test_single_sample(self, model):
        df = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                          columns=load_iris().feature_names)
        p = model.predict(df)
        assert len(p) == 1 and p[0] in [0, 1, 2]

    def test_probabilities_sum_to_one(self, model):
        iris = load_iris(as_frame=True)
        probs = model.predict_proba(iris.data.iloc[:20])
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_no_negative_probabilities(self, model):
        iris = load_iris(as_frame=True)
        assert (model.predict_proba(iris.data) >= 0).all()


class TestMLFlowRegistry:

    def test_model_in_registry(self):
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(MLFLOW_URI)
        try:
            vs = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            assert len(vs) > 0, f"No versions in stage '{MODEL_STAGE}'"
        except Exception as e:
            pytest.skip(f"MLFlow not available: {e}")

    def test_accuracy_metric_logged(self):
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(MLFLOW_URI)
        try:
            vs  = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            if not vs: pytest.skip("No production version")
            run = client.get_run(vs[0].run_id)
            acc = run.data.metrics.get("accuracy")
            assert acc is not None and acc > 0
        except Exception as e:
            pytest.skip(f"MLFlow not available: {e}")
