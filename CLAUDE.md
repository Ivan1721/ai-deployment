# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The actual project lives entirely under `mlops-stack/`. The repo root only has
top-level docs (`README.md`, `SECURITY_ASSESSMENT.md`, `CHANGES_P0_SECURITY.md`,
`DEPLOYMENT_GUIDE_P0.md`, `QUICK_START_P0.md`) describing a recent security
hardening pass — read these if working on auth/TLS/networking changes.

## What this is

A self-contained MLOps reference stack (Docker Compose) that trains, serves,
monitors, and auto-retrains an Iris RandomForest classifier via MLFlow. It's
designed to run entirely on one host with a self-hosted GitHub Actions runner
(no external registry, no SSH deploy — CI/CD operates on the local Docker
daemon directly).

## Common commands

All commands run from `mlops-stack/`.

```bash
# Full automated bring-up (build, train, start, wait for health)
bash start.sh

# Manual step-by-step
docker compose build --parallel
docker compose up -d mlflow                 # wait for "healthy"
docker compose run --rm model-trainer       # trains + registers + promotes to Production
docker compose up -d inference-api nginx drift-detector

# Run the full QA suite (4 levels: data, model, api, performance-drift)
docker compose run --rm \
  -e MLFLOW_TRACKING_URI=http://mlflow:5001 \
  -e INFERENCE_API_URL=http://inference-api:8000 \
  test-runner

# Run a single test file/test (inside the test-runner image)
docker compose run --rm test-runner python -m pytest test_model.py -v
docker compose run --rm test-runner python -m pytest test_api.py::TestHealth -v

# Lint (matches CI)
flake8 model-trainer/train.py inference-api/app.py drift-detector/detector.py \
       tests/run_tests.py tests/test_data.py tests/test_model.py tests/test_api.py \
       --max-line-length=110 --extend-ignore=E501,W503

# Validate compose file
docker compose config --quiet

# Hot-reload the model in the running API after retraining
curl -X POST http://localhost:8000/reload

# Generate self-signed TLS certs for nginx (required before `docker compose up`)
bash generate-certs.sh localhost

# Stop stack (use -v only if you intend to wipe MLFlow/Keycloak data)
docker compose down
docker compose down -v
```

Note: `MLFLOW_TRACKING_URI` / inference port changed from `:5000`/`:8000` in
older docs to `:5001` for MLFlow after the P0 security pass — `nginx` is now
the only service meant to be reached from outside `127.0.0.1`.

## Architecture

### Service graph (docker-compose.yml, network `mlops-net`)

- **mlflow** — MLFlow Tracking Server + Model Registry, SQLite backend at
  `/mlflow/mlflow.db`, artifacts at `/mlflow/artifacts` (volume `mlflow-data`,
  shared by every service that touches the registry). `127.0.0.1:5001` only.
- **model-trainer** — one-shot job (`train.py`): trains a RandomForest on
  sklearn's Iris dataset, logs params/metrics/model to MLFlow, registers as
  `iris-classifier`, and transitions the new version to `Production`
  (archiving the previous one). Runs on demand, not as a long-lived service.
- **inference-api** — FastAPI (`app.py`) loading `models:/iris-classifier/Production`
  from the registry on startup (retries until available). Endpoints:
  `/health`, `/info`, `/predict`, `/reload` (hot-swaps the model without
  restarting the container). `127.0.0.1:8000` only.
- **drift-detector** — long-running loop (`detector.py`):
  - **Data drift**: KS test per feature + Chi-square on prediction
    distribution vs. a reference built from the Iris dataset.
  - **Performance drift** (`performance_drift_detector.py`,
    `PerformanceDriftDetector`/`PerformanceDriftMonitor`): t-test + EWMA +
    effect-size checks on accuracy/precision/recall/F1 vs. baseline.
  - Both paths log to MLFlow experiments (`drift-monitoring`,
    `performance-drift-monitoring`) and, after `N` consecutive drifted
    windows, call `retrain_trigger.trigger(...)`.
- **retrain_trigger.py** — champion/challenger flow: trains a new
  "challenger", compares its accuracy to the current `Production` champion
  (plus `CHALLENGER_MIN_IMPROVEMENT`), and promotes/archives it accordingly.
  On promotion it calls `inference-api`'s `/reload`. Logs everything to the
  `retraining-events` MLFlow experiment.
- **test-runner** (`tests/`) — `run_tests.py` orchestrates 4 pytest levels in
  order, treats each as a gate, and logs a summary run to the
  `quality-assurance` MLFlow experiment:
  1. `test_data.py` — dataset completeness/consistency/schema checks.
  2. `test_model.py` — quality gates (`GATE_MIN_ACCURACY`, `GATE_MIN_F1`,
     per-class recall, CV stability) against the `Production` model.
  3. `test_api.py` — live HTTP tests against `inference-api`
     (`GATE_MAX_LATENCY_MS`).
  4. `test_performance_drift.py` / `test_performance_drift_integration.py`.
  The container's `entrypoint.sh` rewrites `MLFLOW_TRACKING_URI` /
  `INFERENCE_API_URL` to use the Docker gateway IP if they point at internal
  service names — relevant when running this container outside the compose
  network.
- **nginx** — single public entrypoint (ports 80→redirect, 443 HTTPS).
  Reverse-proxies `/mlflow/` → mlflow, `/api/` → inference-api, both gated by
  `auth_request` against **oauth2-proxy**.
- **keycloak** / **oauth2-proxy** — OIDC auth in front of nginx (realm
  `mlops`, client `mlops-client`). `setup-keycloak.sh` bootstraps the realm,
  client, and a test user (`mlops-user`/`mlops123`) post-deploy and prints the
  client secret that must be put into `OAUTH2_PROXY_CLIENT_SECRET`.

### Drift → retrain → promote loop

```
drift-detector (KS/Chi2 or perf metrics drift, N consecutive windows)
  → retrain_trigger.trigger()
     → trains challenger, registers in MLFlow (stage "None")
     → compares challenger vs. champion accuracy
     → promotes to Production (archives old) or archives challenger
     → POST inference-api:8000/reload
```
The same hot-reload step is also driven manually/by CI
(`.github/workflows/retrain.yml`).

### CI/CD (`.github/workflows/`, all `runs-on: self-hosted`)

- **ci.yml**: on PRs/push to `main`/`develop` — flake8 lint, `docker compose
  config`, build all images, bring up `mlflow` → `model-trainer` →
  `inference-api`, then run `test-runner` as the QA gate.
- **cd.yml**: on push to `main` — syncs the repo to `/opt/mlops-stack`
  (project name `mlops`), rebuilds images, ensures `mlflow` is healthy, trains
  only if no `Production` model exists yet, does a zero-downtime update of
  `inference-api` first (health-checked), then updates `drift-detector` and
  `nginx`, then runs the QA suite post-deploy.
- **retrain.yml**: weekly cron (Mon 03:00 UTC) or manual dispatch with
  `min_accuracy`/`min_f1` inputs — retrains, runs QA gates with those
  thresholds, and hot-reloads `inference-api` on success.

### Security posture (P0 hardening — see `CHANGES_P0_SECURITY.md`)

All internal service ports are bound to `127.0.0.1` — only nginx (80/443) is
publicly reachable, and it requires OAuth2 auth via Keycloak for `/mlflow/`
and `/api/`. `inference-api` CORS is restricted via the `CORS_ORIGINS` env var
(default localhost only). Default Keycloak/OAuth2 credentials in
`docker-compose.yml` are dev-only placeholders and **must** be changed before
any real deployment (see "Notas Importantes" in `CHANGES_P0_SECURITY.md`).
