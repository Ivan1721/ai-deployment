"""
run_tests.py
────────────
Orquestador de la suite de QA para MLOps.

Ejecuta tres niveles de tests en orden:
  1. Data Tests      — valida el dataset antes del entrenamiento
  2. Model Tests     — valida el modelo entrenado antes del deploy
  3. API Tests       — valida la Inference API en producción

Si cualquier nivel falla, el proceso termina con exit code 1,
lo que bloquea la promoción del modelo en el pipeline CI/CD.

Todos los resultados se registran en MLFlow como métricas
del experimento "quality-assurance".

Fundamento (Eken et al., 2025):
  "Application quality involves unit testing, acceptance testing (PS89),
   maintaining containerized images and identifying vulnerabilities (PS90)"
  "Model quality is managed through data collection, model training and
   evaluation, and monitoring phases (PS111)"
"""

import os
import sys
import json
import time
import subprocess
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [QA]  %(levelname)s  %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

MLFLOW_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5001")
API_URL     = os.environ.get("INFERENCE_API_URL",   "http://inference-api:8000")
MODEL_NAME  = os.environ.get("MODEL_NAME",          "iris-classifier")
MODEL_STAGE = os.environ.get("MODEL_STAGE",         "Production")

# ── quality gates (umbrales mínimos aceptables) ────────────────────────────
GATE_MIN_ACCURACY  = float(os.environ.get("GATE_MIN_ACCURACY",  "0.90"))
GATE_MIN_F1        = float(os.environ.get("GATE_MIN_F1",        "0.90"))
GATE_MAX_LATENCY_MS= float(os.environ.get("GATE_MAX_LATENCY_MS","500"))


def wait_for(url: str, label: str, retries=15, delay=4):
    import urllib.request
    for i in range(retries):
        try:
            urllib.request.urlopen(url, timeout=3)
            log.info(f"{label} ready ✓")
            return True
        except Exception as e:
            log.info(f"Waiting for {label} ({i+1}/{retries}): {e}")
            time.sleep(delay)
    log.warning(f"{label} not available — some tests may be skipped")
    return False


def run_pytest(test_file: str, extra_args: list = None) -> dict:
    """Ejecuta pytest sobre un archivo y retorna resultados parseados."""
    report_path = f"/tmp/report_{test_file.replace('/', '_')}.json"
    cmd = [
        "python", "-m", "pytest", test_file,
        "-v",
        "--tb=short",
        "--json-report", f"--json-report-file={report_path}",
        "--no-header",
    ]
    if extra_args:
        cmd.extend(extra_args)

    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Mostrar output de pytest en el log
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log.info(f"  {line}")
    if result.returncode not in (0, 1) and result.stderr:
        log.error(result.stderr[:500])

    # Parsear reporte JSON
    try:
        with open(report_path) as f:
            report = json.load(f)
        return {
            "passed":   report["summary"].get("passed", 0),
            "failed":   report["summary"].get("failed", 0),
            "errors":   report["summary"].get("error",  0),
            "total":    report["summary"].get("total",  0),
            "duration": report.get("duration", 0),
            "ok":       result.returncode == 0,
        }
    except Exception:
        return {
            "passed": 0, "failed": 1, "errors": 0,
            "total": 1, "duration": 0, "ok": False,
        }


def log_results_to_mlflow(results: dict, gates_passed: bool):
    """Registra todos los resultados de QA en MLFlow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("quality-assurance")

        with mlflow.start_run(run_name=f"qa-suite-{time.strftime('%Y%m%d-%H%M%S')}"):
            # Resultados por nivel
            for level, r in results.items():
                mlflow.log_metric(f"{level}_passed",  r.get("passed",  0))
                mlflow.log_metric(f"{level}_failed",  r.get("failed",  0))
                mlflow.log_metric(f"{level}_total",   r.get("total",   0))
                mlflow.log_metric(f"{level}_ok",      int(r.get("ok", False)))
                mlflow.log_metric(f"{level}_duration",r.get("duration",0))

            # Quality gates
            mlflow.log_metric("gates_passed",    int(gates_passed))
            mlflow.log_metric("suite_ok",        int(gates_passed))
            mlflow.log_params({
                "gate_min_accuracy":   GATE_MIN_ACCURACY,
                "gate_min_f1":         GATE_MIN_F1,
                "gate_max_latency_ms": GATE_MAX_LATENCY_MS,
                "model_name":          MODEL_NAME,
                "model_stage":         MODEL_STAGE,
            })
            mlflow.set_tag("event_type", "quality_assurance")
            mlflow.set_tag("promoted",   str(gates_passed))

        log.info("QA results logged to MLFlow ✓")
    except Exception:
        log.error(f"MLFlow logging failed:\n{traceback.format_exc()}")


def print_summary(results: dict, gates_passed: bool):
    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║              QA SUITE — RESULTS SUMMARY              ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    total_passed = total_failed = 0
    for level, r in results.items():
        status = "✓ PASS" if r["ok"] else "✗ FAIL"
        log.info(f"║  {level:<20} {status}  "
                 f"({r['passed']} passed, {r['failed']} failed) "
                 f"  {r['duration']:.1f}s  ║")
        total_passed += r["passed"]
        total_failed += r["failed"]
    log.info("╠══════════════════════════════════════════════════════╣")
    gate_status = "✓ PROMOTION ALLOWED" if gates_passed else "✗ PROMOTION BLOCKED"
    log.info(f"║  {gate_status:<50} ║")
    log.info(f"║  Total: {total_passed} passed, {total_failed} failed"
             f"{'':>35}║")
    log.info("╚══════════════════════════════════════════════════════╝")


def main():
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║          MLOps Quality Assurance Suite               ║")
    log.info("║          Clase 03 — Testing en MLOps                 ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info(f"  MLFLOW_URI:        {MLFLOW_URI}")
    log.info(f"  API_URL:           {API_URL}")
    log.info(f"  Gate accuracy:     >= {GATE_MIN_ACCURACY}")
    log.info(f"  Gate f1:           >= {GATE_MIN_F1}")
    log.info(f"  Gate latency:      <= {GATE_MAX_LATENCY_MS}ms")

    # Esperar servicios
    wait_for(f"{MLFLOW_URI}/", "MLFlow")
    mlflow_ok = True
    api_ok = wait_for(f"{API_URL}/health", "InferenceAPI", retries=10, delay=3)

    results = {}
    all_ok  = True

    # ── NIVEL 1: Data Tests ────────────────────────────────────────────────
    log.info("")
    log.info("━━━ LEVEL 1: DATA TESTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    r = run_pytest("test_data.py")
    results["data"] = r
    if not r["ok"]:
        log.error("Data tests FAILED — pipeline should not proceed to training")
        all_ok = False

    # ── NIVEL 2: Model Tests (quality gates) ──────────────────────────────
    log.info("")
    log.info("━━━ LEVEL 2: MODEL TESTS (QUALITY GATES) ━━━━━━━━━━━━")
    env_args = [
        f"--mlflow-uri={MLFLOW_URI}",
        f"--model-name={MODEL_NAME}",
        f"--model-stage={MODEL_STAGE}",
        f"--min-accuracy={GATE_MIN_ACCURACY}",
        f"--min-f1={GATE_MIN_F1}",
    ]
    r = run_pytest("test_model.py", extra_args=env_args)
    results["model"] = r
    if not r["ok"]:
        log.error("Model tests FAILED — model does not meet quality gates")
        all_ok = False

    # ── NIVEL 3: API Tests ─────────────────────────────────────────────────
    log.info("")
    log.info("━━━ LEVEL 3: API TESTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if api_ok:
        api_args = [
            f"--api-url={API_URL}",
            f"--max-latency={GATE_MAX_LATENCY_MS}",
        ]
        r = run_pytest("test_api.py", extra_args=api_args)
        results["api"] = r
        if not r["ok"]:
            log.error("API tests FAILED — inference service has issues")
            all_ok = False
    else:
        log.warning("Inference API not available — skipping API tests")
        results["api"] = {"passed": 0, "failed": 0, "total": 0,
                          "duration": 0, "ok": True}

    # ── Registrar en MLFlow ────────────────────────────────────────────────
    log_results_to_mlflow(results, all_ok)
    print_summary(results, all_ok)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
