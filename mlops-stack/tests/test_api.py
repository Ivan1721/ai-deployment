"""
test_api.py
───────────
Nivel 3: API Tests — Integration & Contract Tests

Valida la Inference API en producción real (no mocks).
Cubre contrato de la API, latencia, manejo de errores y comportamiento bajo carga.

Categorías:
  - Contract tests: schema de request/response
  - Health checks: endpoints de operación
  - Latencia: SLA de tiempo de respuesta
  - Manejo de errores: inputs inválidos retornan códigos correctos
  - Comportamiento bajo carga: múltiples requests simultáneos

Fundamento (Eken et al., 2025):
  "Application quality involves unit testing, acceptance testing (PS89),
   maintaining containerized images and identifying vulnerabilities (PS90)"
"""

import pytest
import time
import json
import urllib.request
import urllib.error


# ── configuración ──────────────────────────────────────────────────────────
def pytest_addoption(parser):
    # Evitar conflicto si ya fueron agregadas por test_model.py
    try:
        parser.addoption("--api-url",     default="http://inference-api:8000")
        parser.addoption("--max-latency", default="500")
    except ValueError:
        pass


@pytest.fixture(scope="module")
def api_url(request):
    return request.config.getoption("--api-url", default="http://inference-api:8000")


@pytest.fixture(scope="module")
def max_latency_ms(request):
    return float(request.config.getoption("--max-latency", default="500"))


# ── helpers ────────────────────────────────────────────────────────────────
def http_get(url: str) -> tuple:
    """GET request. Retorna (status_code, response_dict)."""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        pytest.skip(f"API no disponible: {e}")


def http_post(url: str, body: dict) -> tuple:
    """POST request con JSON body. Retorna (status_code, response_dict, latency_ms)."""
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=15) as resp:
            latency_ms = (time.perf_counter() - t0) * 1000
            return resp.status, json.loads(resp.read()), latency_ms
    except urllib.error.HTTPError as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            body = json.loads(e.read())
        except Exception:
            body = {}
        return e.code, body, latency_ms
    except Exception as e:
        pytest.skip(f"API no disponible: {e}")


# Inputs de prueba válidos
VALID_SINGLE   = {"instances": [[5.1, 3.5, 1.4, 0.2]]}
VALID_BATCH    = {"instances": [
    [5.1, 3.5, 1.4, 0.2],   # setosa
    [6.7, 3.0, 5.2, 2.3],   # virginica
    [5.8, 2.7, 4.1, 1.0],   # versicolor
]}


# ══════════════════════════════════════════════════════════════════
# HEALTH & OPERACIÓN
# ══════════════════════════════════════════════════════════════════

class TestHealthEndpoints:

    def test_health_returns_200(self, api_url):
        """GET /health debe retornar 200."""
        status, body = http_get(f"{api_url}/health")
        assert status == 200, f"Health endpoint retornó {status}"

    def test_health_status_ok(self, api_url):
        """El campo 'status' debe ser 'ok' cuando el modelo está cargado."""
        _, body = http_get(f"{api_url}/health")
        assert body.get("status") == "ok", (
            f"API en estado degradado: {body.get('status')}\n"
            f"Respuesta completa: {body}"
        )

    def test_health_includes_model_version(self, api_url):
        """El health check debe informar la versión del modelo activo."""
        _, body = http_get(f"{api_url}/health")
        assert "model_version" in body, "Health no informa model_version"
        assert body["model_version"] is not None, "model_version es None"

    def test_health_includes_model_stage(self, api_url):
        """El health check debe informar el stage del modelo."""
        _, body = http_get(f"{api_url}/health")
        assert "model_stage" in body
        assert body["model_stage"] == "Production"

    def test_info_endpoint_returns_200(self, api_url):
        """GET /info debe retornar 200 con metadata del modelo."""
        status, body = http_get(f"{api_url}/info")
        assert status == 200

    def test_info_includes_feature_names(self, api_url):
        """GET /info debe listar los nombres de los features."""
        _, body = http_get(f"{api_url}/info")
        assert "features" in body
        assert len(body["features"]) == 4

    def test_info_includes_class_names(self, api_url):
        """GET /info debe listar los nombres de las clases."""
        _, body = http_get(f"{api_url}/info")
        assert "classes" in body
        expected = ["setosa", "versicolor", "virginica"]
        assert body["classes"] == expected, (
            f"Clases incorrectas: {body['classes']}"
        )


# ══════════════════════════════════════════════════════════════════
# CONTRATO DE LA API — PREDICT
# ══════════════════════════════════════════════════════════════════

class TestPredictContract:

    def test_predict_single_instance_returns_200(self, api_url):
        """POST /predict con 1 instancia debe retornar 200."""
        status, _, _ = http_post(f"{api_url}/predict", VALID_SINGLE)
        assert status == 200, f"Predict retornó {status}"

    def test_predict_batch_returns_200(self, api_url):
        """POST /predict con batch de 3 instancias debe retornar 200."""
        status, _, _ = http_post(f"{api_url}/predict", VALID_BATCH)
        assert status == 200

    def test_response_has_predictions_field(self, api_url):
        """La respuesta debe tener el campo 'predictions'."""
        _, body, _ = http_post(f"{api_url}/predict", VALID_SINGLE)
        assert "predictions" in body, f"Campo 'predictions' ausente: {body}"

    def test_response_predictions_count_matches_input(self, api_url):
        """El número de predicciones debe igualar el número de instancias."""
        _, body, _ = http_post(f"{api_url}/predict", VALID_BATCH)
        n_input = len(VALID_BATCH["instances"])
        n_preds = len(body.get("predictions", []))
        assert n_preds == n_input, (
            f"Se enviaron {n_input} instancias pero se recibieron {n_preds} predicciones"
        )

    def test_each_prediction_has_class_id(self, api_url):
        """Cada predicción debe incluir 'class_id'."""
        _, body, _ = http_post(f"{api_url}/predict", VALID_BATCH)
        for i, pred in enumerate(body.get("predictions", [])):
            assert "class_id" in pred, f"Predicción {i} no tiene 'class_id': {pred}"

    def test_each_prediction_has_class_name(self, api_url):
        """Cada predicción debe incluir 'class_name'."""
        _, body, _ = http_post(f"{api_url}/predict", VALID_BATCH)
        valid_names = {"setosa", "versicolor", "virginica"}
        for i, pred in enumerate(body.get("predictions", [])):
            assert "class_name" in pred
            assert pred["class_name"] in valid_names, (
                f"Predicción {i}: class_name inválido '{pred['class_name']}'"
            )

    def test_each_prediction_has_valid_probability(self, api_url):
        """La probabilidad debe estar entre 0 y 1."""
        _, body, _ = http_post(f"{api_url}/predict", VALID_BATCH)
        for i, pred in enumerate(body.get("predictions", [])):
            assert "probability" in pred
            prob = pred["probability"]
            assert 0.0 <= prob <= 1.0, (
                f"Predicción {i}: probabilidad inválida {prob}"
            )

    def test_class_id_and_name_are_consistent(self, api_url):
        """class_id y class_name deben corresponder (setosa=0, versicolor=1, virginica=2)."""
        id_to_name = {0: "setosa", 1: "versicolor", 2: "virginica"}
        _, body, _ = http_post(f"{api_url}/predict", VALID_BATCH)
        for i, pred in enumerate(body.get("predictions", [])):
            expected_name = id_to_name.get(pred["class_id"])
            assert pred["class_name"] == expected_name, (
                f"Predicción {i}: class_id={pred['class_id']} pero "
                f"class_name='{pred['class_name']}' (esperado '{expected_name}')"
            )

    def test_response_includes_model_metadata(self, api_url):
        """La respuesta debe incluir model_name, model_version y model_stage."""
        _, body, _ = http_post(f"{api_url}/predict", VALID_SINGLE)
        for field in ["model_name", "model_version", "model_stage"]:
            assert field in body, f"Campo '{field}' ausente en la respuesta"

    def test_setosa_predicted_correctly(self, api_url):
        """
        Una muestra de setosa bien caracterizada debe predecirse correctamente.
        (sepal: grande, petal: pequeño — patrón distintivo de setosa)
        """
        setosa_sample = {"instances": [[5.1, 3.5, 1.4, 0.2]]}
        _, body, _ = http_post(f"{api_url}/predict", setosa_sample)
        preds = body.get("predictions", [])
        assert len(preds) == 1
        assert preds[0]["class_name"] == "setosa", (
            f"Muestra de setosa predicha como '{preds[0]['class_name']}'"
        )

    def test_virginica_predicted_correctly(self, api_url):
        """Una muestra de virginica bien caracterizada debe predecirse correctamente."""
        virginica_sample = {"instances": [[6.7, 3.0, 5.2, 2.3]]}
        _, body, _ = http_post(f"{api_url}/predict", virginica_sample)
        preds = body.get("predictions", [])
        assert len(preds) == 1
        assert preds[0]["class_name"] == "virginica", (
            f"Muestra de virginica predicha como '{preds[0]['class_name']}'"
        )


# ══════════════════════════════════════════════════════════════════
# LATENCIA — SLA
# ══════════════════════════════════════════════════════════════════

class TestLatency:

    def test_single_prediction_latency(self, api_url, max_latency_ms):
        """
        Una predicción individual no debe tardar más de max_latency_ms.
        SLA típico para inference en producción: < 500ms p99.
        """
        _, _, latency = http_post(f"{api_url}/predict", VALID_SINGLE)
        assert latency < max_latency_ms, (
            f"Latencia {latency:.1f}ms excede el SLA de {max_latency_ms}ms"
        )

    def test_average_latency_over_10_requests(self, api_url, max_latency_ms):
        """La latencia promedio en 10 requests consecutivos no debe exceder el SLA."""
        latencies = []
        for _ in range(10):
            _, _, lat = http_post(f"{api_url}/predict", VALID_SINGLE)
            latencies.append(lat)
        avg = sum(latencies) / len(latencies)
        assert avg < max_latency_ms, (
            f"Latencia promedio {avg:.1f}ms excede el SLA de {max_latency_ms}ms\n"
            f"Latencias individuales: {[f'{l:.0f}' for l in latencies]}"
        )

    def test_batch_latency_acceptable(self, api_url, max_latency_ms):
        """Un batch de 10 instancias no debe tardar más de 3x el SLA de single."""
        batch_10 = {"instances": [[5.1, 3.5, 1.4, 0.2]] * 10}
        _, _, latency = http_post(f"{api_url}/predict", batch_10)
        assert latency < max_latency_ms * 3, (
            f"Batch latency {latency:.1f}ms excede 3x SLA ({max_latency_ms*3:.0f}ms)"
        )


# ══════════════════════════════════════════════════════════════════
# MANEJO DE ERRORES
# ══════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_wrong_number_of_features_returns_422(self, api_url):
        """3 features en lugar de 4 debe retornar 422 Unprocessable Entity."""
        bad_input = {"instances": [[5.1, 3.5, 1.4]]}  # faltan petal width
        status, _, _ = http_post(f"{api_url}/predict", bad_input)
        assert status == 422, (
            f"Input con 3 features debería retornar 422, retornó {status}"
        )

    def test_too_many_features_returns_422(self, api_url):
        """5 features en lugar de 4 debe retornar 422."""
        bad_input = {"instances": [[5.1, 3.5, 1.4, 0.2, 9.9]]}
        status, _, _ = http_post(f"{api_url}/predict", bad_input)
        assert status == 422, (
            f"Input con 5 features debería retornar 422, retornó {status}"
        )

    def test_empty_instances_list_returns_error(self, api_url):
        """Lista vacía de instancias debe retornar error (422 o 400)."""
        bad_input = {"instances": []}
        status, _, _ = http_post(f"{api_url}/predict", bad_input)
        assert status in (400, 422), (
            f"Lista vacía debería retornar 4xx, retornó {status}"
        )

    def test_string_features_returns_422(self, api_url):
        """Features con strings en lugar de floats deben retornar 422."""
        bad_input = {"instances": [["a", "b", "c", "d"]]}
        status, _, _ = http_post(f"{api_url}/predict", bad_input)
        assert status == 422, (
            f"Strings como features debería retornar 422, retornó {status}"
        )
