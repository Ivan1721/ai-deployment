# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Start full stack (build + train + inference API + nginx)
bash start.sh

# Rebuild + wipe volumes (required when train.py changes metrics or model structure)
docker compose down -v
bash start.sh

# Manual step-by-step
docker compose build --parallel
docker compose up -d mlflow
docker compose run --rm model-trainer
docker compose up -d inference-api nginx

# Run full QA suite (4 test levels)
docker compose run --rm test-runner

# Run a single test file
docker compose run --rm test-runner python -m pytest test_model.py -v

# Run tests with custom gates
docker compose run --rm -e GATE_MIN_R2=0.80 -e GATE_MAX_SMAPE=15 test-runner

# Quick predict (scenario 0 = HumanOnly)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"scenario":0,"workers":6,"crop_row":2,"rand_pos":0,"activity":"harv_mixed"}'

# Hot-reload models in inference-api without restart
curl -X POST http://localhost:8000/reload

# Watch drift detector logs
docker compose logs -f drift-detector

# Lint (matches CI)
flake8 model-trainer/train.py inference-api/app.py drift-detector/detector.py \
  tests/run_tests.py tests/test_data.py tests/test_model.py tests/test_api.py \
  --max-line-length=110 --extend-ignore=E501,W503
```

## Configuration (mlops_common)

The stack is driven by a YAML config file loaded via `MODEL_CONFIG` env var (default: `/app/configs/hri-harvesting.yaml`).

| Config | Use case |
|---|---|
| `configs/hri-harvesting.yaml` | HRI dataset — 2 scenarios × 4 targets, 12-model tournament |
| `configs/iris-classifier.yaml` | Iris dataset — single RandomForest classifier (reference example) |

`mlops_common/` is a shared Python package copied into every container at build time:
- `config.py` — `load_config()` returns `ExperimentConfig` + optional `MultiSlotConfig`
- `data_sources/` — `sklearn_dataset`, `csv`, `hri_csv` (HRI preprocessing)
- `models/` — 14 model strategies (2 classification + 12 regression)
- `problem_types/` — `ClassificationProblem` and `RegressionProblem` (metrics, quality gates, drift)

To switch to Iris classification mode, change `MODEL_CONFIG` in docker-compose.yml to `iris-classifier.yaml` and rebuild.

## Architecture

### Services (docker-compose.yml)

| Service | Port | Restart | Role |
|---|---|---|---|
| `mlflow` | 5001 | unless-stopped | Tracking server + Model Registry (SQLite + filesystem artifacts) |
| `model-trainer` | — | no | One-shot training job; exits after registering 8 models |
| `inference-api` | 8000 | unless-stopped | FastAPI; loads all 8 Production models at startup |
| `drift-detector` | — | unless-stopped | Background loop; KS tests + performance drift every 30s |
| `test-runner` | — | no | QA suite; invoked manually or from CI |
| `nginx` | 80 | unless-stopped | Reverse proxy |

All services share `mlops-net` bridge. The `mlflow-data` volume is the single source of truth for all models — it is mounted into `mlflow`, `model-trainer`, `inference-api`, `drift-detector`, and `test-runner`.

### Dataset

`data/simulation_all.csv` — HRI Agricultural Harvesting Dataset (284 rows, 15 cols).

- Column `Scenario`: 0 = HumanOnly, 1 = WithRobot
- `MainActivity` is one-hot encoded into `Act_Ladder`, `Act_Mixed`, `Act_Picker` (reference = `harv_ground`)
- `FEATURE_NAMES = ["Humans", "ROW_N", "RandomPosition", "Act_Ladder", "Act_Mixed", "Act_Picker"]`
- 4 regression targets: `TotalRecollected`, `CargoZoneProd`, `TotalWorkload`, `AvgProduction`

### Training Pipeline (`model-trainer/train.py`)

**2 scenarios × 4 targets = 8 model slots.** For each slot:

1. **Phase 1 — Tournament**: All 12 candidate pipelines (LinearRegression, Ridge, Lasso, ElasticNet, SVR, ExtraTrees, RandomForest, GradientBoosting, MLP, XGBoost, LightGBM, CatBoost) evaluated with 5-fold CV. Each CV run logged as `phase=comparison` in MLflow experiment `hri-harvesting`.
2. **Phase 2 — Winner**: Best R² candidate re-fit on full train set. Metrics logged: `ho_r2`, `ho_rmse`, `ho_mae`, `ho_smape`, `ho_max_error`. Feature ranking via `permutation_importance` saved as `feature_ranking.json` artifact. Run tagged `phase=winner`.
3. **Phase 3 — Promotion**: If `ho_r2 >= QUALITY_GATE (0.70)`, model promoted to `Production` in MLflow Registry as `hri-{scenario}-{target}` (e.g., `hri-HumanOnly-TotalRecollected`).

All pipelines include `StandardScaler` — required for SVR and MLP. XGBoost/LightGBM/CatBoost are optional (graceful skip if not installed).

### Inference API (`inference-api/app.py`)

On startup, loads all 8 models from `models:/hri-{scenario}-{target}/Production`. The in-memory `state["models"]` dict is keyed by `{scenario_label: {target_alias: model}}`.

- `POST /predict` — takes `{scenario, workers, crop_row, rand_pos, activity}`, encodes activity to one-hot, returns all 4 targets in one call.
- `GET /health` — reports `models_loaded` (should be 8); `"degraded"` if < 8.
- `POST /reload` — hot-reloads all 8 models without container restart.

### Drift Detector (`drift-detector/detector.py`)

Runs an infinite loop every `CHECK_INTERVAL_S=30` seconds:

- **Data drift**: KS test (`scipy.stats.ks_2samp`) on each of the 6 features vs. training distribution.
- **Concept drift**: KS test on live prediction distribution vs. reference prediction distribution.
- **Performance drift**: `PerformanceDriftDetector` on R²/RMSE/MAE/sMAPE/Max Error using 3 statistical tests (effect size, t-test, EWMA).

After `CONSECUTIVE_DRIFT_WINDOWS=2` consecutive windows with drift, calls `retrain_trigger.trigger()`. Logs all metrics to MLflow experiments `drift-monitoring` and `performance-drift-monitoring`.

`PerformanceDriftDetector` (in `drift-detector/performance_drift_detector.py`) is the canonical version — **also duplicated** to `tests/performance_drift_detector.py` (both must stay in sync).

### Quality Gates

| Gate | Default | Env var |
|---|---|---|
| R² (hold-out) | ≥ 0.70 | `GATE_MIN_R2` |
| sMAPE | < 20% | `GATE_MAX_SMAPE` |
| API latency | < 500ms | `GATE_MAX_LATENCY_MS` |

sMAPE is used instead of MAPE because some targets (`TotalWorkload` in `WithRobot` scenario) can be 0, causing MAPE to be `inf`.

### Test Suite (`tests/`)

Orchestrated by `run_tests.py` (4 levels run in sequence):

| Level | File | What it validates |
|---|---|---|
| 1 | `test_data.py` | CSV schema, row count, scenario split, feature ranges |
| 2 | `test_model.py` | All 8 models: R² ≥ 0.70, sMAPE < 20%, metrics logged in MLflow, `feature_ranking.json` artifact present |
| 3 | `test_api.py` | HTTP 200, schema, non-negative predictions, latency < 500ms |
| 4 | `test_performance_drift.py` | Drift detector logic (unit tests) |

### CI/CD (`.github/workflows/`)

All workflows use `runs-on: self-hosted` — the runner shares the Docker socket with the host.

- `ci.yml`: triggers on PR/push to `main`/`develop`. Builds → trains → runs QA suite. Cleans containers but **not volumes** (`docker compose down` without `-v`).
- `cd.yml`: triggers on push to `main`. Syncs files to `/opt/mlops-stack`, rebuilds, checks if `hri-HumanOnly-TotalRecollected` exists in Production before deciding to retrain.
- `retrain.yml`: scheduled Mondays 03:00 UTC, or manual dispatch.

**Note**: `mlops-stack/.github/workflows/retrain.yml` and `.github/workflows/retrain.yml` both pass `GATE_MIN_R2`/`GATE_MAX_SMAPE`; the former is used when the stack is deployed at `/opt/mlops-stack`.

### BuildKit

`start.sh` creates a named buildx builder `mlops-builder` on first run to avoid Docker's default builder lease-corruption bug on interrupted builds:

```bash
docker buildx inspect mlops-builder > /dev/null 2>&1 \
  || docker buildx create --name mlops-builder --driver docker-container
export BUILDX_BUILDER=mlops-builder
```

### When to use `docker compose down -v`

Required when `train.py` changes (new metrics, new model structure) because old models in the `mlflow-data` volume lack the new metadata and tests will fail against them. Without `-v`, just rebuilding images is sufficient.
