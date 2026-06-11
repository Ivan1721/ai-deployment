"""
test_data.py
------------
Validates the HRI Agricultural Harvesting Dataset before training.
Checks schema, types, ranges, nulls, and cardinality per the data contract.
"""

import os
import pytest
import pandas as pd
import numpy as np

DATASET_PATH = os.environ.get("DATASET_PATH", "/data/simulation_all.csv")

EXPECTED_COLUMNS = [
    "ExperimentID", "Scenario", "Scenario_label", "Humans", "ROW_N",
    "RandomPosition", "MainActivity",
    "TimeWeightedAverageMetabolicRate_kcal",
    "AverageHumanProduction_crop_units",
    "TotalHumanWorkload_kcal",
    "TotalProductionCargoZone_crop_units",
    "RemainingCropsHumanBags_crop_units",
    "RemainingCropsBoxes_crop_units",
    "RemainingCropsRobot_crop_units",
    "TotalRecollectedCrops_crop_units",
]

FEATURE_COLS = ["Humans", "ROW_N", "RandomPosition", "MainActivity"]
TARGET_COLS  = [
    "TotalRecollectedCrops_crop_units",
    "TotalProductionCargoZone_crop_units",
    "TotalHumanWorkload_kcal",
    "AverageHumanProduction_crop_units",
]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATASET_PATH)


class TestSchema:
    def test_column_count(self, df):
        assert df.shape[1] == 15, f"Expected 15 columns, got {df.shape[1]}"

    def test_column_names(self, df):
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_minimum_rows(self, df):
        assert len(df) >= 100, f"Expected >= 100 rows, got {len(df)}"

    def test_no_nulls_in_features(self, df):
        nulls = df[FEATURE_COLS + TARGET_COLS].isnull().sum()
        assert nulls.sum() == 0, f"Null values found:\n{nulls[nulls > 0]}"

    def test_no_null_experiment_id(self, df):
        assert df["ExperimentID"].isnull().sum() == 0


class TestScenario:
    def test_scenario_values(self, df):
        assert set(df["Scenario"].unique()) <= {0, 1}, \
            f"Unexpected Scenario values: {df['Scenario'].unique()}"

    def test_scenario_label_consistency(self, df):
        mapping = {0: "Human-Only", 1: "Human-Robot"}
        for scenario_id, label in mapping.items():
            sub = df[df["Scenario"] == scenario_id]["Scenario_label"].unique()
            assert list(sub) == [label], \
                f"Scenario {scenario_id} has unexpected labels: {sub}"

    def test_both_scenarios_present(self, df):
        assert 0 in df["Scenario"].values, "Scenario 0 (Human-Only) missing"
        assert 1 in df["Scenario"].values, "Scenario 1 (Human-Robot) missing"


class TestFeatureRanges:
    def test_humans_values(self, df):
        valid = {1, 3, 6, 8, 10, 12}
        actual = set(df["Humans"].unique())
        unexpected = actual - valid
        assert not unexpected, f"Unexpected Humans values: {unexpected}"

    def test_row_n_values(self, df):
        assert set(df["ROW_N"].unique()) <= {1, 2, 3}, \
            f"Unexpected ROW_N values: {df['ROW_N'].unique()}"

    def test_random_position_values(self, df):
        assert set(df["RandomPosition"].unique()) <= {0, 1}, \
            f"Unexpected RandomPosition values: {df['RandomPosition'].unique()}"

    def test_activity_values(self, df):
        valid    = {"harv_ground", "harv_ladder", "harv_mixed", "harv_picker"}
        actual   = set(df["MainActivity"].unique())
        unexpected = actual - valid
        assert not unexpected, f"Unexpected MainActivity values: {unexpected}"

    def test_humans_positive(self, df):
        assert (df["Humans"] > 0).all(), "Humans must be > 0"

    def test_row_n_positive(self, df):
        assert (df["ROW_N"] > 0).all(), "ROW_N must be > 0"


class TestTargetRanges:
    def test_total_recollected_non_negative(self, df):
        assert (df["TotalRecollectedCrops_crop_units"] >= 0).all()

    def test_cargo_zone_non_negative(self, df):
        assert (df["TotalProductionCargoZone_crop_units"] >= 0).all()

    def test_workload_positive(self, df):
        assert (df["TotalHumanWorkload_kcal"] > 0).all(), \
            "TotalHumanWorkload_kcal must be > 0"

    def test_avg_production_positive(self, df):
        assert (df["AverageHumanProduction_crop_units"] > 0).all(), \
            "AverageHumanProduction_crop_units must be > 0"

    def test_no_leakage_columns_used_as_targets(self, df):
        leakage_cols = [
            "RemainingCropsHumanBags_crop_units",
            "RemainingCropsBoxes_crop_units",
            "RemainingCropsRobot_crop_units",
            "TimeWeightedAverageMetabolicRate_kcal",
        ]
        # Verify leakage cols exist but are NOT in TARGET_COLS
        for col in leakage_cols:
            assert col in df.columns, f"Expected column {col} to exist in dataset"
            assert col not in TARGET_COLS, f"Leakage column {col} must not be a ML target"


class TestDataIntegrity:
    def test_experiment_id_unique(self, df):
        dupes = df["ExperimentID"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate ExperimentIDs found"

    def test_human_only_no_robot_crops(self, df):
        human_only = df[df["Scenario"] == 0]
        assert (human_only["RemainingCropsRobot_crop_units"] == 0).all(), \
            "Human-Only scenario should have 0 robot crops"

    def test_additive_consistency(self, df):
        computed = (
            df["TotalProductionCargoZone_crop_units"]
            + df["RemainingCropsHumanBags_crop_units"]
            + df["RemainingCropsBoxes_crop_units"]
            + df["RemainingCropsRobot_crop_units"]
        )
        diff = (computed - df["TotalRecollectedCrops_crop_units"]).abs()
        assert (diff < 1e-3).all(), \
            f"TotalRecollected != sum of components in {(diff >= 1e-3).sum()} rows"

    def test_sufficient_per_scenario(self, df):
        for scenario_id in [0, 1]:
            n = (df["Scenario"] == scenario_id).sum()
            assert n >= 20, f"Scenario {scenario_id} has only {n} rows (need >= 20)"
