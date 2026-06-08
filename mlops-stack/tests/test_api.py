"""
test_api.py  ─  Nivel 3: API Integration Tests
"""
import os, pytest, time, json
import urllib.request, urllib.error

API_URL     = os.environ.get("API_URL",     "http://host.docker.internal:8000")
MAX_LATENCY = float(os.environ.get("MAX_LATENCY", "500"))

VALID_SINGLE = {"instances": [[5.1, 3.5, 1.4, 0.2]]}
VALID_BATCH  = {"instances": [[5.1, 3.5, 1.4, 0.2],
                               [6.7, 3.0, 5.2, 2.3],
                               [5.8, 2.7, 4.1, 1.0]]}


def get(path: str):
    try:
        with urllib.request.urlopen(f"{API_URL}{path}", timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        pytest.skip(f"API not reachable: {e}")


def post(path: str, body: dict):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{API_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            lat = (time.perf_counter() - t0) * 1000
            return r.status, json.loads(r.read()), lat
    except urllib.error.HTTPError as e:
        lat = (time.perf_counter() - t0) * 1000
        try: body = json.loads(e.read())
        except Exception: body = {}
        return e.code, body, lat
    except Exception as e:
        pytest.skip(f"API not reachable: {e}")


# ── Health ─────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_200(self):
        status, _ = get("/health")
        assert status == 200

    def test_health_status_ok(self):
        _, body = get("/health")
        assert body.get("status") == "ok", f"Status degraded: {body}"

    def test_health_has_version(self):
        _, body = get("/health")
        assert body.get("model_version") is not None

    def test_info_200(self):
        status, _ = get("/info")
        assert status == 200

    def test_info_classes(self):
        _, body = get("/info")
        assert body.get("classes") == ["setosa", "versicolor", "virginica"]


# ── Contract ───────────────────────────────────────────────────────────────

class TestPredictContract:

    def test_single_200(self):
        status, _, _ = post("/predict", VALID_SINGLE)
        assert status == 200

    def test_batch_200(self):
        status, _, _ = post("/predict", VALID_BATCH)
        assert status == 200

    def test_prediction_count_matches(self):
        _, body, _ = post("/predict", VALID_BATCH)
        assert len(body["predictions"]) == len(VALID_BATCH["instances"])

    def test_class_id_valid(self):
        _, body, _ = post("/predict", VALID_BATCH)
        for p in body["predictions"]:
            assert p["class_id"] in [0, 1, 2]

    def test_class_name_valid(self):
        _, body, _ = post("/predict", VALID_BATCH)
        valid = {"setosa", "versicolor", "virginica"}
        for p in body["predictions"]:
            assert p["class_name"] in valid

    def test_probability_in_range(self):
        _, body, _ = post("/predict", VALID_BATCH)
        for p in body["predictions"]:
            assert 0.0 <= p["probability"] <= 1.0

    def test_class_id_and_name_consistent(self):
        mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
        _, body, _ = post("/predict", VALID_BATCH)
        for p in body["predictions"]:
            assert p["class_name"] == mapping[p["class_id"]]

    def test_setosa_correct(self):
        _, body, _ = post("/predict", {"instances": [[5.1, 3.5, 1.4, 0.2]]})
        assert body["predictions"][0]["class_name"] == "setosa"

    def test_virginica_correct(self):
        _, body, _ = post("/predict", {"instances": [[6.7, 3.0, 5.2, 2.3]]})
        assert body["predictions"][0]["class_name"] == "virginica"


# ── Latencia ───────────────────────────────────────────────────────────────

class TestLatency:

    def test_single_latency(self):
        _, _, lat = post("/predict", VALID_SINGLE)
        assert lat < MAX_LATENCY, f"Latency {lat:.0f}ms > SLA {MAX_LATENCY}ms"

    def test_avg_10_requests(self):
        lats = [post("/predict", VALID_SINGLE)[2] for _ in range(10)]
        avg  = sum(lats) / len(lats)
        assert avg < MAX_LATENCY, f"Avg latency {avg:.0f}ms > SLA {MAX_LATENCY}ms"


# ── Error handling ─────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_too_few_features_422(self):
        status, _, _ = post("/predict", {"instances": [[5.1, 3.5, 1.4]]})
        assert status == 422

    def test_too_many_features_422(self):
        status, _, _ = post("/predict", {"instances": [[5.1, 3.5, 1.4, 0.2, 9.9]]})
        assert status == 422

    def test_empty_list_error(self):
        status, _, _ = post("/predict", {"instances": []})
        assert status in (400, 422)
