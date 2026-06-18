"""
test_api_contract.py — Nivel 6: API Contract Tests

Freezes the exact JSON schema of every endpoint and detects breaking changes:
field renames, type changes, missing required fields.

Complements test_api.py (which tests behavior) — this file tests the interface
contract so CI catches breaking changes before consumers notice.
"""
import os, json, math, re, pytest
import urllib.request, urllib.error

API_URL = os.environ.get("INFERENCE_API_URL", "http://host.docker.internal:8000")
API_KEY = os.environ.get("API_KEY", "")
_AUTH   = {"X-API-Key": API_KEY} if API_KEY else {}

# ── Frozen schema definitions ──────────────────────────────────────────────────
# Format: field_name → (expected_type_or_tuple, required)
# required=False: field may be absent or None without failing the contract.

HEALTH_CONTRACT = {
    "status":        (str, True),
    "models_loaded": (int, True),
    "model_stage":   (str, True),
    "loaded_at":     (str, False),
}

INFO_CONTRACT = {
    "scenarios":   (list, True),
    "targets":     (list, True),
    "features":    (list, True),
    "activities":  (list, True),
    "model_stage": (str,  True),
}

PREDICT_CONTRACT = {
    "scenario_label":      (str,          True),
    "total_recollected":   ((int, float), True),
    "cargo_zone_prod":     ((int, float), True),
    "total_workload_kcal": ((int, float), True),
    "avg_production":      ((int, float), True),
    "model_stage":         (str,          True),
    "loaded_at":           (str,          True),
}

MODEL_INFO_CONTRACT = {
    "name":    (str, True),
    "version": (str, True),
    "stage":   (str, True),
    "run_id":  (str, True),
}

GENERIC_PREDICT_CONTRACT = {
    "model":      (str,               True),
    "stage":      (str,               True),
    "prediction": ((int, float, list), True),
}

# ── Sample payloads ────────────────────────────────────────────────────────────

PREDICT_BODY = {
    "scenario": 0, "workers": 6, "crop_row": 2,
    "rand_pos": 0, "activity": "harv_mixed",
}
GENERIC_FEATURES = {
    "Humans": 6.0, "ROW_N": 2.0, "RandomPosition": 0.0,
    "Act_Ladder": 0.0, "Act_Mixed": 0.0, "Act_Picker": 0.0,
}
TARGET_MODEL = "hri-HumanOnly-TotalRecollected"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(path):
    req = urllib.request.Request(f"{API_URL}{path}", headers=_AUTH)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        pytest.skip(f"API not reachable: {e}")


def _post(path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{API_URL}{path}", data=data,
        headers={"Content-Type": "application/json", **_AUTH},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            body_out = json.loads(e.read())
        except Exception:
            body_out = {}
        return e.code, body_out
    except Exception as e:
        pytest.skip(f"API not reachable: {e}")


def _check_contract(body, contract, label):
    """Raises AssertionError listing every contract violation found."""
    errors = []
    for field, (expected_type, required) in contract.items():
        if field not in body or body[field] is None:
            if required:
                errors.append(f"MISSING required field '{field}'")
        else:
            val = body[field]
            if not isinstance(val, expected_type):
                type_name = (
                    " | ".join(t.__name__ for t in expected_type)
                    if isinstance(expected_type, tuple)
                    else expected_type.__name__
                )
                errors.append(
                    f"TYPE MISMATCH '{field}': expected {type_name}, "
                    f"got {type(val).__name__} (value={val!r})"
                )
    assert not errors, (
        f"Contract violations for {label}:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealthContract:

    def test_schema(self):
        status, body = _get("/health")
        assert status == 200
        _check_contract(body, HEALTH_CONTRACT, "/health")

    def test_status_is_known_value(self):
        _, body = _get("/health")
        assert body["status"] in ("ok", "degraded"), \
            f"status must be 'ok' or 'degraded', got '{body['status']}'"

    def test_models_loaded_in_range(self):
        _, body = _get("/health")
        n = body["models_loaded"]
        assert 0 <= n <= 8, f"models_loaded={n} out of valid range [0, 8]"

    def test_loaded_at_is_iso8601(self):
        _, body = _get("/health")
        loaded_at = body.get("loaded_at") or ""
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", loaded_at), \
            f"loaded_at is not ISO 8601: '{loaded_at}'"


# ── /info ──────────────────────────────────────────────────────────────────────

class TestInfoContract:

    def test_schema(self):
        status, body = _get("/info")
        assert status == 200
        _check_contract(body, INFO_CONTRACT, "/info")

    def test_scenarios_have_id_and_label(self):
        _, body = _get("/info")
        for s in body["scenarios"]:
            assert "id" in s and "label" in s, \
                f"Scenario entry missing 'id' or 'label': {s}"

    def test_targets_count(self):
        _, body = _get("/info")
        assert len(body["targets"]) == 4, \
            f"Expected 4 targets, got {len(body['targets'])}: {body['targets']}"

    def test_features_count(self):
        _, body = _get("/info")
        assert len(body["features"]) == 6, \
            f"Expected 6 features, got {len(body['features'])}: {body['features']}"

    def test_known_activities(self):
        _, body = _get("/info")
        expected = {"harv_ground", "harv_ladder", "harv_mixed", "harv_picker"}
        actual   = set(body.get("activities", []))
        assert actual == expected, \
            f"Activities mismatch: expected {expected}, got {actual}"


# ── /predict ───────────────────────────────────────────────────────────────────

class TestPredictContract:

    def test_schema(self):
        status, body = _post("/predict", PREDICT_BODY)
        assert status == 200
        _check_contract(body, PREDICT_CONTRACT, "/predict")

    def test_numeric_fields_finite(self):
        _, body = _post("/predict", PREDICT_BODY)
        for field in ("total_recollected", "cargo_zone_prod",
                      "total_workload_kcal", "avg_production"):
            assert math.isfinite(body[field]), \
                f"Field '{field}' is not finite: {body[field]}"

    def test_scenario_label_values(self):
        for scenario, expected_label in [(0, "HumanOnly"), (1, "WithRobot")]:
            _, body = _post("/predict", {**PREDICT_BODY, "scenario": scenario})
            assert body["scenario_label"] == expected_label, (
                f"scenario_label for scenario={scenario}: "
                f"expected '{expected_label}', got '{body.get('scenario_label')}'"
            )

    def test_model_stage_is_production(self):
        _, body = _post("/predict", PREDICT_BODY)
        assert body["model_stage"] == "Production", \
            f"model_stage must be 'Production', got '{body.get('model_stage')}'"


# ── /models ────────────────────────────────────────────────────────────────────

class TestModelsListContract:

    def test_schema_is_list(self):
        status, body = _get("/models")
        assert status == 200
        assert isinstance(body, list), f"Expected list from /models, got {type(body)}"

    def test_each_item_schema(self):
        _, body = _get("/models")
        for item in body:
            _check_contract(item, MODEL_INFO_CONTRACT, "/models[]")

    def test_at_least_eight_models(self):
        _, body = _get("/models")
        assert len(body) >= 8, \
            f"Expected at least 8 models, got {len(body)}"

    def test_all_stages_production(self):
        _, body = _get("/models")
        non_prod = [m for m in body if m.get("stage") != "Production"]
        assert not non_prod, \
            f"Non-Production models returned: {[m['name'] for m in non_prod]}"

    def test_no_duplicate_names(self):
        _, body = _get("/models")
        names = [m["name"] for m in body]
        dupes = {n for n in names if names.count(n) > 1}
        assert not dupes, f"Duplicate model names in /models: {dupes}"


# ── /models/{name} ─────────────────────────────────────────────────────────────

class TestModelDetailContract:

    def test_schema(self):
        status, body = _get(f"/models/{TARGET_MODEL}")
        assert status == 200
        _check_contract(body, MODEL_INFO_CONTRACT, f"/models/{TARGET_MODEL}")

    def test_name_matches_request(self):
        _, body = _get(f"/models/{TARGET_MODEL}")
        assert body["name"] == TARGET_MODEL, \
            f"name mismatch: expected '{TARGET_MODEL}', got '{body.get('name')}'"

    def test_stage_is_production(self):
        _, body = _get(f"/models/{TARGET_MODEL}")
        assert body["stage"] == "Production", \
            f"stage must be 'Production', got '{body.get('stage')}'"

    def test_version_is_numeric_string(self):
        _, body = _get(f"/models/{TARGET_MODEL}")
        v = body.get("version", "")
        assert v.isdigit(), f"version must be a numeric string, got '{v}'"


# ── /models/{name}/predict ─────────────────────────────────────────────────────

class TestGenericPredictContract:

    def test_schema(self):
        status, body = _post(
            f"/models/{TARGET_MODEL}/predict",
            {"features": GENERIC_FEATURES},
        )
        assert status == 200
        _check_contract(body, GENERIC_PREDICT_CONTRACT, f"/models/{TARGET_MODEL}/predict")

    def test_prediction_finite(self):
        _, body = _post(
            f"/models/{TARGET_MODEL}/predict",
            {"features": GENERIC_FEATURES},
        )
        pred = body["prediction"]
        if isinstance(pred, list):
            assert all(math.isfinite(v) for v in pred), \
                f"prediction list contains non-finite values: {pred}"
        else:
            assert math.isfinite(pred), f"prediction is not finite: {pred}"

    def test_model_name_echoed(self):
        _, body = _post(
            f"/models/{TARGET_MODEL}/predict",
            {"features": GENERIC_FEATURES},
        )
        assert body["model"] == TARGET_MODEL, \
            f"model name mismatch: expected '{TARGET_MODEL}', got '{body.get('model')}'"

    def test_stage_echoed(self):
        _, body = _post(
            f"/models/{TARGET_MODEL}/predict",
            {"features": GENERIC_FEATURES, "stage": "Production"},
        )
        assert body["stage"] == "Production", \
            f"stage must be 'Production', got '{body.get('stage')}'"
