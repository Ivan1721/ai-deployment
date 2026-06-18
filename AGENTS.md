# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Dockerized MLOps stack in `mlops-stack/`. Service code is split by responsibility: `model-trainer/train.py` trains and registers models, `inference-api/app.py` serves FastAPI predictions, and `drift-detector/` contains data and performance drift logic. Shared configuration helpers live in `mlops-stack/mlops_common/`, with model profiles in `mlops-stack/configs/`. Test code is in `mlops-stack/tests/`, input data is in `mlops-stack/data/`, and deployment support lives in `mlops-stack/nginx/`, `mlops-stack/mlflow/`, and `.github/workflows/`.

## Build, Test, and Development Commands

Run commands from `mlops-stack/` unless noted.

- `bash start.sh`: build, train, and start the local stack using the project script.
- `docker compose build`: build all service images.
- `docker compose up -d mlflow`: start MLflow before training.
- `docker compose run --rm model-trainer`: train models and register production versions.
- `docker compose up -d inference-api nginx`: start the prediction API and reverse proxy.
- `docker compose run --rm test-runner`: run the full QA suite.
- `docker compose config --quiet`: validate Compose syntax.
- `docker compose down`: stop containers while keeping MLflow data; add `-v` only when intentionally clearing models.

## Coding Style & Naming Conventions

Python code uses 4-space indentation and clear, module-level separation by service. Prefer explicit environment/config access through `mlops_common` and YAML profiles instead of hard-coded paths or model names. Keep filenames lowercase with underscores for Python modules, such as `performance_drift_detector.py`. CI runs `flake8` with `--max-line-length=110` and ignores `E501,W503`, so keep lines readable and avoid broad formatting churn.

## Testing Guidelines

Tests use Python test files under `mlops-stack/tests/`, named `test_*.py`. Add focused tests near the affected area: data schema checks in `test_data.py`, model quality gates in `test_model.py`, API behavior in `test_api.py`, and drift logic in `test_performance_drift.py`. The quality gates include minimum R2, maximum sMAPE, and API latency thresholds configured in `docker-compose.yml`.

## Commit & Pull Request Guidelines

Git history follows Conventional Commit style, for example `fix: ...`, `docs: ...`, `feat(security): ...`, and `refactor(tests): ...`. Keep commit subjects imperative and scoped when helpful. Pull requests should describe the change, list validation commands run, mention config or model-impacting changes, and include API screenshots or sample `curl` output when prediction behavior changes.

## Security & Configuration Tips

Keep service ports bound to localhost unless intentionally exposing them. Do not commit secrets, generated MLflow volumes, or local datasets beyond the tracked sample data. Use `MODEL_CONFIG` to switch profiles such as `configs/hri-harvesting.yaml` or `configs/iris-classifier.yaml`.
