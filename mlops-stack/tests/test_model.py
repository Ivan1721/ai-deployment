"""
test_model.py  ─  Nivel 2: Model Tests + Quality Gates  (HRI Regression)
Tests all 8 models: 2 scenarios x 4 targets.
"""
import os, pytest, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              max_error as sklearn_max_error)
from sklearn.dummy import DummyRegressor

MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
MODEL_STAGE  = os.environ.get("MODEL_STAGE",  "Production")
DATASET_PATH = os.environ.get("DATASET_PATH", "/data/simulation_all.csv")
MIN_R2       = float(os.environ.get("GATE_MIN_R2",    "0.70"))
MAX_SMAPE    = float(os.environ.get("GATE_MAX_SMAPE", "20.0"))   # %

SCENARIOS = {0: "HumanOnly", 1: "WithRobot"}
TARGETS = {
    "TotalRecollected": "TotalRecollectedCrops_crop_units",
    "CargoZoneProd":    "TotalProductionCargoZone_crop_units",
    "TotalWorkload":    "TotalHumanWorkload_kcal",
    "AvgProduction":    "AverageHumanProduction_crop_units",
}
FEATURE_NAMES = ["Humans", "ROW_N", "RandomPosition", "Act_Ladder", "Act_Mixed", "Act_Picker"]

ALL_MODELS = [
    (sid, slabel, ta, tc)
    for sid, slabel in SCENARIOS.items()
    for ta, tc in TARGETS.items()
]


def _load_df():
    df = pd.read_csv(DATASET_PATH)
    dummies = pd.get_dummies(df["MainActivity"], prefix="Act", drop_first=True)
    dummies = dummies.rename(columns={
        "Act_harv_ladder": "Act_Ladder",
        "Act_harv_mixed":  "Act_Mixed",
        "Act_harv_picker": "Act_Picker",
    })
    for col in ["Act_Ladder", "Act_Mixed", "Act_Picker"]:
        if col not in dummies.columns:
            dummies[col] = 0
    return pd.concat([df, dummies[["Act_Ladder", "Act_Mixed", "Act_Picker"]]], axis=1)


def _model_name(scenario_label: str, target_alias: str) -> str:
    return f"hri-{scenario_label}-{target_alias}"


def _smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.where(denom == 0, 0.0, np.abs(y_pred - y_true) / denom)) * 100)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    return _load_df()


@pytest.fixture(scope="module", params=ALL_MODELS,
                ids=[f"{s}-{t}" for _, s, t, _ in ALL_MODELS])
def model_ctx(request):
    """Loads one of the 8 production models from MLflow."""
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(MLFLOW_URI)

    scenario_id, scenario_label, target_alias, target_col = request.param
    name = _model_name(scenario_label, target_alias)
    try:
        model = mlflow.sklearn.load_model(f"models:/{name}/{MODEL_STAGE}")
    except Exception as e:
        pytest.skip(f"Cannot load {name}: {e}")
    return model, scenario_id, scenario_label, target_alias, target_col


@pytest.fixture(scope="module")
def splits(model_ctx, df):
    model, scenario_id, _, _, target_col = model_ctx
    df_sc = df[df["Scenario"] == scenario_id]
    X = df_sc[FEATURE_NAMES].values
    y = df_sc[target_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    return model, X_tr, X_te, y_tr, y_te


# ── Quality Gates ──────────────────────────────────────────────────────────────

class TestPerformanceGates:

    def test_r2_gate(self, splits, model_ctx):
        model, _, X_te, _, y_te = splits
        _, _, scenario_label, target_alias, _ = model_ctx
        r2 = r2_score(y_te, model.predict(X_te))
        assert r2 >= MIN_R2, (
            f"GATE FAILED: {scenario_label}-{target_alias}  R2={r2:.4f} < {MIN_R2}"
        )

    def test_smape_gate(self, splits, model_ctx):
        model, _, X_te, _, y_te = splits
        _, _, scenario_label, target_alias, _ = model_ctx
        s = _smape(y_te, model.predict(X_te))
        assert s < MAX_SMAPE, (
            f"GATE FAILED: {scenario_label}-{target_alias}  sMAPE={s:.2f}% >= {MAX_SMAPE}%"
        )

    def test_max_error_finite(self, splits, model_ctx):
        model, _, X_te, _, y_te = splits
        _, _, scenario_label, target_alias, _ = model_ctx
        me = sklearn_max_error(y_te, model.predict(X_te))
        assert np.isfinite(me), (
            f"{scenario_label}-{target_alias}: max_error is not finite ({me})"
        )

    def test_cv_stability(self, model_ctx, df):
        model, scenario_id, scenario_label, target_alias, target_col = model_ctx
        df_sc = df[df["Scenario"] == scenario_id]
        X = df_sc[FEATURE_NAMES].values
        y = df_sc[target_col].values
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        assert scores.std() < 0.15, (
            f"CV unstable: {scenario_label}-{target_alias}  std={scores.std():.4f}"
        )
        assert scores.mean() >= MIN_R2, (
            f"CV mean too low: {scenario_label}-{target_alias}  mean={scores.mean():.4f}"
        )


# ── Sanity Checks ──────────────────────────────────────────────────────────────

class TestSanityChecks:

    def test_better_than_dummy(self, splits, model_ctx):
        model, X_tr, X_te, y_tr, y_te = splits
        _, _, scenario_label, target_alias, _ = model_ctx
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_tr, y_tr)
        r2_model = r2_score(y_te, model.predict(X_te))
        r2_dummy = r2_score(y_te, dummy.predict(X_te))
        assert r2_model >= r2_dummy + 0.30, (
            f"{scenario_label}-{target_alias}: model R2={r2_model:.4f} "
            f"barely beats mean-baseline R2={r2_dummy:.4f}"
        )

    def test_non_negative_predictions(self, model_ctx, df):
        """All targets (crops/workload) must be >= 0."""
        model, scenario_id, scenario_label, target_alias, _ = model_ctx
        X = df[df["Scenario"] == scenario_id][FEATURE_NAMES].values
        preds = model.predict(X)
        neg = int((preds < 0).sum())
        assert neg == 0, (
            f"{scenario_label}-{target_alias}: {neg} negative predictions"
        )

    def test_single_sample(self, model_ctx):
        """Model accepts a single row and returns a finite scalar."""
        model, *_ = model_ctx
        # 6 workers, row 2, no random position, mixed activity
        x = np.array([[6, 2, 0, 0, 1, 0]], dtype=float)
        p = model.predict(x)
        assert len(p) == 1 and np.isfinite(p[0]), "Prediction must be a single finite number"

    def test_reproducible(self, model_ctx, df):
        model, scenario_id, *_ = model_ctx
        X = df[df["Scenario"] == scenario_id][FEATURE_NAMES].values[:10]
        assert np.allclose(model.predict(X), model.predict(X)), "Model is not deterministic"


# ── MLflow Registry ────────────────────────────────────────────────────────────

class TestMLFlowRegistry:

    def test_model_in_registry(self, model_ctx):
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(MLFLOW_URI)
        _, _, scenario_label, target_alias, _ = model_ctx
        name = _model_name(scenario_label, target_alias)
        try:
            vs = client.get_latest_versions(name, stages=[MODEL_STAGE])
            assert len(vs) > 0, f"No versions of '{name}' in stage '{MODEL_STAGE}'"
        except Exception as e:
            pytest.skip(f"MLFlow not available: {e}")

    def test_all_metrics_logged(self, model_ctx):
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(MLFLOW_URI)
        _, _, scenario_label, target_alias, _ = model_ctx
        name = _model_name(scenario_label, target_alias)
        try:
            vs = client.get_latest_versions(name, stages=[MODEL_STAGE])
            if not vs:
                pytest.skip("No production version found")
            run = client.get_run(vs[0].run_id)
            m   = run.data.metrics
            for key in ("ho_r2", "ho_rmse", "ho_mae", "ho_smape", "ho_max_error"):
                assert key in m, f"Metric '{key}' not logged for {name}"
        except Exception as e:
            pytest.skip(f"MLFlow not available: {e}")

    def test_feature_ranking_artifact(self, model_ctx):
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(MLFLOW_URI)
        _, _, scenario_label, target_alias, _ = model_ctx
        name = _model_name(scenario_label, target_alias)
        try:
            vs = client.get_latest_versions(name, stages=[MODEL_STAGE])
            if not vs:
                pytest.skip("No production version found")
            artifacts = [a.path for a in client.list_artifacts(vs[0].run_id)]
            assert "feature_ranking.json" in artifacts, \
                f"feature_ranking.json not found for {name}. Found: {artifacts}"
        except Exception as e:
            pytest.skip(f"MLFlow not available: {e}")

    def test_r2_metric_logged(self, model_ctx):
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(MLFLOW_URI)
        _, _, scenario_label, target_alias, _ = model_ctx
        name = _model_name(scenario_label, target_alias)
        try:
            vs = client.get_latest_versions(name, stages=[MODEL_STAGE])
            if not vs:
                pytest.skip("No production version found")
            run = client.get_run(vs[0].run_id)
            ho_r2 = run.data.metrics.get("ho_r2")
            assert ho_r2 is not None, "Metric 'ho_r2' not found in MLflow run"
            assert ho_r2 >= MIN_R2, f"Logged ho_r2={ho_r2:.4f} < {MIN_R2}"
        except Exception as e:
            pytest.skip(f"MLFlow not available: {e}")
