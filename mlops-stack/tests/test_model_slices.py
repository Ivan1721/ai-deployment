"""
test_model_slices.py — Nivel 5: Model Slice Tests

Validates that performance is consistent across data sub-groups.
Catches hidden bias where the aggregate R² passes but a specific
activity type or worker-count group fails.
"""
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
MODEL_STAGE  = os.environ.get("MODEL_STAGE", "Production")
DATASET_PATH = os.environ.get("DATASET_PATH", "/data/simulation_all.csv")
MIN_R2_SLICE = float(os.environ.get("GATE_MIN_R2_SLICE", "0.50"))  # softer gate per slice
MAX_R2_GAP   = float(os.environ.get("GATE_MAX_R2_GAP",   "0.25"))  # max gap between scenarios

SCENARIOS = {0: "HumanOnly", 1: "WithRobot"}
TARGETS = {
    "TotalRecollected": "TotalRecollectedCrops_crop_units",
    "CargoZoneProd":    "TotalProductionCargoZone_crop_units",
    "TotalWorkload":    "TotalHumanWorkload_kcal",
    "AvgProduction":    "AverageHumanProduction_crop_units",
}
FEATURE_NAMES = ["Humans", "ROW_N", "RandomPosition", "Act_Ladder", "Act_Mixed", "Act_Picker"]
ACTIVITIES    = ["harv_ground", "harv_ladder", "harv_mixed", "harv_picker"]
WORKER_GROUPS = {
    "low":    [1, 3],
    "medium": [6, 8],
    "high":   [10, 12],
}

ALL_SLOTS = [
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


@pytest.fixture(scope="module")
def df():
    return _load_df()


@pytest.fixture(scope="module")
def all_models():
    """Loads all 8 production models once. Tests skip individually if a model is missing."""
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(MLFLOW_URI)
    models = {}
    for sid, slabel, ta, tc in ALL_SLOTS:
        name = f"hri-{slabel}-{ta}"
        try:
            models[(sid, slabel, ta, tc)] = mlflow.sklearn.load_model(
                f"models:/{name}/{MODEL_STAGE}"
            )
        except Exception:
            pass
    return models


# ── Slice tests: activity type ─────────────────────────────────────────────────

class TestActivitySlices:
    """R² per activity must exceed a softer gate than the global 0.70 gate."""

    @pytest.mark.parametrize("scenario_id,scenario_label", list(SCENARIOS.items()))
    @pytest.mark.parametrize("target_alias,target_col", list(TARGETS.items()))
    @pytest.mark.parametrize("activity", ACTIVITIES)
    def test_r2_per_activity(self, all_models, df, scenario_id, scenario_label,
                              target_alias, target_col, activity):
        key = (scenario_id, scenario_label, target_alias, target_col)
        if key not in all_models:
            pytest.skip(f"Model hri-{scenario_label}-{target_alias} not loaded")

        sub = df[(df["Scenario"] == scenario_id) & (df["MainActivity"] == activity)]
        if len(sub) < 5:
            pytest.skip(
                f"Too few samples for scenario={scenario_label}, "
                f"activity={activity} (n={len(sub)})"
            )

        preds = all_models[key].predict(sub[FEATURE_NAMES].values)
        r2    = r2_score(sub[target_col].values, preds)
        assert r2 >= MIN_R2_SLICE, (
            f"SLICE FAIL [{scenario_label}/{target_alias}/{activity}]: "
            f"R²={r2:.4f} < {MIN_R2_SLICE} (n={len(sub)})"
        )


# ── Slice tests: worker count groups ──────────────────────────────────────────

class TestWorkerCountSlices:
    """R² must hold for low / medium / high worker configurations."""

    @pytest.mark.parametrize("scenario_id,scenario_label", list(SCENARIOS.items()))
    @pytest.mark.parametrize("target_alias,target_col", list(TARGETS.items()))
    @pytest.mark.parametrize("group_name,worker_values", list(WORKER_GROUPS.items()))
    def test_r2_per_worker_group(self, all_models, df, scenario_id, scenario_label,
                                  target_alias, target_col, group_name, worker_values):
        key = (scenario_id, scenario_label, target_alias, target_col)
        if key not in all_models:
            pytest.skip(f"Model hri-{scenario_label}-{target_alias} not loaded")

        sub = df[
            (df["Scenario"] == scenario_id) & (df["Humans"].isin(worker_values))
        ]
        if len(sub) < 5:
            pytest.skip(
                f"Too few samples for scenario={scenario_label}, "
                f"workers={worker_values} (n={len(sub)})"
            )

        preds = all_models[key].predict(sub[FEATURE_NAMES].values)
        r2    = r2_score(sub[target_col].values, preds)
        assert r2 >= MIN_R2_SLICE, (
            f"SLICE FAIL [{scenario_label}/{target_alias}/workers={worker_values} ({group_name})]: "
            f"R²={r2:.4f} < {MIN_R2_SLICE} (n={len(sub)})"
        )


# ── Scenario gap ───────────────────────────────────────────────────────────────

class TestScenarioGap:
    """The R² gap between HumanOnly and WithRobot must be bounded for each target."""

    @pytest.mark.parametrize("target_alias,target_col", list(TARGETS.items()))
    def test_scenario_r2_gap_bounded(self, all_models, df, target_alias, target_col):
        keys = {
            slabel: (sid, slabel, target_alias, target_col)
            for sid, slabel in SCENARIOS.items()
        }
        for slabel, key in keys.items():
            if key not in all_models:
                pytest.skip(f"Model hri-{slabel}-{target_alias} not loaded")

        r2s = {}
        for sid, slabel in SCENARIOS.items():
            sub  = df[df["Scenario"] == sid]
            preds = all_models[keys[slabel]].predict(sub[FEATURE_NAMES].values)
            r2s[slabel] = r2_score(sub[target_col].values, preds)

        gap = abs(r2s["HumanOnly"] - r2s["WithRobot"])
        assert gap <= MAX_R2_GAP, (
            f"Large scenario gap for {target_alias}: "
            f"HumanOnly={r2s['HumanOnly']:.4f}, "
            f"WithRobot={r2s['WithRobot']:.4f}, "
            f"gap={gap:.4f} > {MAX_R2_GAP}"
        )


# ── Domain monotonicity ────────────────────────────────────────────────────────

class TestMonotonicityHeuristics:
    """Domain invariant: more workers → more total production on the harv_ground baseline."""

    def test_more_workers_more_total_recollected(self, all_models):
        key = (0, "HumanOnly", "TotalRecollected", "TotalRecollectedCrops_crop_units")
        if key not in all_models:
            pytest.skip("hri-HumanOnly-TotalRecollected not loaded")

        model = all_models[key]
        # harv_ground is the reference category: Act_Ladder=Act_Mixed=Act_Picker=0
        means = {
            w: float(model.predict(np.array([[w, 2, 0, 0, 0, 0]], dtype=float))[0])
            for w in [1, 3, 6, 8, 10, 12]
        }
        low_mean  = np.mean([means[1],  means[3]])
        high_mean = np.mean([means[10], means[12]])
        assert high_mean > low_mean, (
            f"Monotonicity violated: avg(10-12 workers)={high_mean:.1f} "
            f"<= avg(1-3 workers)={low_mean:.1f}"
        )
