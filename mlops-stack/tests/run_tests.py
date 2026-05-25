"""
run_tests.py  ─  MLOps QA Suite Orchestrator
"""

import os, sys, json, time, subprocess, logging, traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [QA]  %(levelname)s  %(message)s",
    stream=sys.stdout, force=True,
)
log = logging.getLogger(__name__)

MLFLOW_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
API_URL     = os.environ.get("INFERENCE_API_URL",   "http://host.docker.internal:8000")
MODEL_NAME  = os.environ.get("MODEL_NAME",  "iris-classifier")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")

GATE_MIN_ACCURACY   = float(os.environ.get("GATE_MIN_ACCURACY",   "0.90"))
GATE_MIN_F1         = float(os.environ.get("GATE_MIN_F1",         "0.90"))
GATE_MAX_LATENCY_MS = float(os.environ.get("GATE_MAX_LATENCY_MS", "500"))


def probe(url: str) -> bool:
    """Intento único, sin espera. Retorna True si el servicio responde."""
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=5)
        return True
    except Exception:
        return False


def wait_for(url: str, label: str, retries=20, delay=3) -> bool:
    import urllib.request
    log.info(f"Waiting for {label} at {url} ...")
    for i in range(retries):
        try:
            urllib.request.urlopen(url, timeout=5)
            log.info(f"{label} ready ✓")
            return True
        except Exception as e:
            log.info(f"  {label} not ready ({i+1}/{retries}): {e}")
            time.sleep(delay)
    log.warning(f"{label} not available after {retries} attempts")
    return False


def run_pytest(test_file: str, extra_env: dict = None) -> dict:
    report = f"/tmp/report_{test_file}.json"
    cmd    = [
        "python", "-m", "pytest", test_file,
        "-v", "--tb=short",
        "--json-report", f"--json-report-file={report}",
        "--no-header",
    ]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    for line in result.stdout.strip().split("\n"):
        log.info(f"  {line}")
    if result.returncode not in (0, 1) and result.stderr:
        log.error(result.stderr[:500])

    try:
        with open(report) as f:
            rpt = json.load(f)
        return {
            "passed":   rpt["summary"].get("passed",  0),
            "failed":   rpt["summary"].get("failed",  0),
            "total":    rpt["summary"].get("total",   0),
            "duration": round(rpt.get("duration", 0), 1),
            "ok":       result.returncode == 0,
        }
    except Exception:
        return {"passed": 0, "failed": 1, "total": 1, "duration": 0, "ok": False}


def log_mlflow(results: dict, gates_ok: bool):
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("quality-assurance")
        with mlflow.start_run(run_name=f"qa-{time.strftime('%Y%m%d-%H%M%S')}"):
            for level, r in results.items():
                mlflow.log_metric(f"{level}_passed",  r["passed"])
                mlflow.log_metric(f"{level}_failed",  r["failed"])
                mlflow.log_metric(f"{level}_ok",      int(r["ok"]))
            mlflow.log_metric("gates_passed", int(gates_ok))
            mlflow.log_params({
                "gate_accuracy": GATE_MIN_ACCURACY,
                "gate_f1":       GATE_MIN_F1,
                "gate_latency":  GATE_MAX_LATENCY_MS,
            })
            mlflow.set_tag("event_type", "quality_assurance")
        log.info("Results logged to MLFlow ✓")
    except Exception:
        log.warning(f"MLFlow logging skipped: {traceback.format_exc()}")


def print_summary(results: dict, gates_ok: bool):
    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║              QA SUITE — RESULTS SUMMARY              ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    for level, r in results.items():
        status = "✓ PASS" if r["ok"] else "✗ FAIL"
        log.info(f"║  {level:<12} {status}  "
                 f"({r['passed']} passed, {r['failed']} failed, {r['duration']}s)  ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    verdict = "✓ PROMOTION ALLOWED" if gates_ok else "✗ PROMOTION BLOCKED"
    log.info(f"║  {verdict:<50} ║")
    log.info("╚══════════════════════════════════════════════════════╝")


def main():
    log.info("=== MLOps Quality Assurance Suite ===")
    log.info(f"  MLFLOW_URI = {MLFLOW_URI}")
    log.info(f"  API_URL    = {API_URL}")

    mlflow_ok = wait_for(f"{MLFLOW_URI}/", "MLFlow")
    if not mlflow_ok:
        log.error("MLFlow not reachable — aborting")
        sys.exit(1)

    # Comprobar API una sola vez sin espera larga
    api_ok = probe(f"{API_URL}/health")
    log.info(f"InferenceAPI reachable: {api_ok}")

    results  = {}
    all_ok   = True

    # ── Nivel 1: Data Tests ────────────────────────────────────────────────
    log.info("")
    log.info("━━━ LEVEL 1: DATA TESTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    r = run_pytest("test_data.py")
    results["data"] = r
    if not r["ok"]:
        all_ok = False
        log.error("Data tests FAILED")

    # ── Nivel 2: Model Tests ───────────────────────────────────────────────
    log.info("")
    log.info("━━━ LEVEL 2: MODEL TESTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    r = run_pytest("test_model.py", extra_env={
        "MLFLOW_URI":   MLFLOW_URI,
        "MODEL_NAME":   MODEL_NAME,
        "MODEL_STAGE":  MODEL_STAGE,
        "MIN_ACCURACY": str(GATE_MIN_ACCURACY),
        "MIN_F1":       str(GATE_MIN_F1),
    })
    results["model"] = r
    if not r["ok"]:
        all_ok = False
        log.error("Model tests FAILED — model does not meet quality gates")

    # ── Nivel 3: API Tests ─────────────────────────────────────────────────
    log.info("")
    log.info("━━━ LEVEL 3: API TESTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if api_ok:
        r = run_pytest("test_api.py", extra_env={
            "API_URL":      API_URL,
            "MAX_LATENCY":  str(GATE_MAX_LATENCY_MS),
        })
        results["api"] = r
        if not r["ok"]:
            all_ok = False
            log.error("API tests FAILED")
    else:
        log.warning("InferenceAPI not reachable — skipping API tests")
        results["api"] = {"passed": 0, "failed": 0, "total": 0, "duration": 0, "ok": True}

    log_mlflow(results, all_ok)
    print_summary(results, all_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
