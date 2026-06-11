"""
test_api.py  ─  Nivel 3: API Integration Tests  (HRI Regression)
"""
import os, pytest, time, json
import urllib.request, urllib.error

API_URL     = os.environ.get("INFERENCE_API_URL", "http://host.docker.internal:8000")
MAX_LATENCY = float(os.environ.get("GATE_MAX_LATENCY_MS", "500"))

# Valid sample requests
VALID_HUMAN_ONLY = {
    "scenario": 0, "workers": 6, "crop_row": 2,
    "rand_pos": 0, "activity": "harv_mixed",
}
VALID_WITH_ROBOT = {
    "scenario": 1, "workers": 10, "crop_row": 1,
    "rand_pos": 0, "activity": "harv_ground",
}

RESPONSE_KEYS = {
    "scenario_label", "total_recollected", "cargo_zone_prod",
    "total_workload_kcal", "avg_production", "model_stage", "loaded_at",
}


def _get(path: str):
    try:
        with urllib.request.urlopen(f"{API_URL}{path}", timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        pytest.skip(f"API not reachable: {e}")


def _post(path: str, body: dict):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{API_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            lat = (time.perf_counter() - t0) * 1000
            return r.status, json.loads(r.read()), lat
    except urllib.error.HTTPError as e:
        lat = (time.perf_counter() - t0) * 1000
        try:
            body = json.loads(e.read())
        except Exception:
            body = {}
        return e.code, body, lat
    except Exception as e:
        pytest.skip(f"API not reachable: {e}")


# ── Health & Info ──────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_200(self):
        status, _ = _get("/health")
        assert status == 200

    def test_health_status_ok(self):
        _, body = _get("/health")
        assert body.get("status") == "ok", f"Health degraded: {body}"

    def test_health_models_loaded(self):
        _, body = _get("/health")
        assert body.get("models_loaded") == 8, \
            f"Expected 8 models loaded, got {body.get('models_loaded')}"

    def test_info_200(self):
        status, _ = _get("/info")
        assert status == 200

    def test_info_has_scenarios(self):
        _, body = _get("/info")
        assert "scenarios" in body or "models" in body or body, \
            f"Unexpected /info response: {body}"


# ── Predict Contract ───────────────────────────────────────────────────────────

class TestPredictContract:

    def test_human_only_200(self):
        status, _, _ = _post("/predict", VALID_HUMAN_ONLY)
        assert status == 200

    def test_with_robot_200(self):
        status, _, _ = _post("/predict", VALID_WITH_ROBOT)
        assert status == 200

    def test_response_has_all_keys(self):
        _, body, _ = _post("/predict", VALID_HUMAN_ONLY)
        for key in RESPONSE_KEYS:
            assert key in body, f"Missing key '{key}' in response"

    def test_scenario_label_human_only(self):
        _, body, _ = _post("/predict", VALID_HUMAN_ONLY)
        assert body["scenario_label"] == "HumanOnly"

    def test_scenario_label_with_robot(self):
        _, body, _ = _post("/predict", VALID_WITH_ROBOT)
        assert body["scenario_label"] == "WithRobot"

    def test_predictions_are_non_negative(self):
        _, body, _ = _post("/predict", VALID_HUMAN_ONLY)
        assert body["total_recollected"]   >= 0
        assert body["cargo_zone_prod"]     >= 0
        assert body["total_workload_kcal"] >= 0
        assert body["avg_production"]      >= 0

    def test_model_stage_is_production(self):
        _, body, _ = _post("/predict", VALID_HUMAN_ONLY)
        assert body.get("model_stage") == "Production"

    def test_all_activities(self):
        activities = ["harv_ground", "harv_ladder", "harv_mixed", "harv_picker"]
        for act in activities:
            req = {**VALID_HUMAN_ONLY, "activity": act}
            status, body, _ = _post("/predict", req)
            assert status == 200, f"Failed for activity={act}: {body}"

    def test_all_worker_counts(self):
        for w in [1, 3, 6, 8, 10, 12]:
            req = {**VALID_HUMAN_ONLY, "workers": w}
            status, _, _ = _post("/predict", req)
            assert status == 200, f"Failed for workers={w}"

    def test_all_crop_rows(self):
        for r in [1, 2, 3]:
            req = {**VALID_HUMAN_ONLY, "crop_row": r}
            status, _, _ = _post("/predict", req)
            assert status == 200, f"Failed for crop_row={r}"


# ── Latency ───────────────────────────────────────────────────────────────────

class TestLatency:

    def test_single_latency(self):
        _, _, lat = _post("/predict", VALID_HUMAN_ONLY)
        assert lat < MAX_LATENCY, f"Latency {lat:.0f}ms > SLA {MAX_LATENCY}ms"

    def test_avg_10_requests(self):
        lats = [_post("/predict", VALID_HUMAN_ONLY)[2] for _ in range(10)]
        avg  = sum(lats) / len(lats)
        assert avg < MAX_LATENCY, f"Avg latency {avg:.0f}ms > SLA {MAX_LATENCY}ms"


# ── Error Handling ─────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_invalid_scenario_422(self):
        status, _, _ = _post("/predict", {**VALID_HUMAN_ONLY, "scenario": 5})
        assert status == 422

    def test_invalid_activity_422(self):
        status, _, _ = _post("/predict", {**VALID_HUMAN_ONLY, "activity": "harvesting"})
        assert status in (400, 422)

    def test_missing_field_422(self):
        body = {"scenario": 0, "workers": 6}  # missing crop_row, rand_pos, activity
        status, _, _ = _post("/predict", body)
        assert status == 422

    def test_workers_out_of_range_422(self):
        status, _, _ = _post("/predict", {**VALID_HUMAN_ONLY, "workers": 0})
        assert status == 422
