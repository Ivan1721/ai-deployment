# MLOps Stack — Architecture & Technical Reference

## Overview

This MLOps stack trains, serves, and monitors **8 regression models** (2 scenarios × 4 targets) on the **HRI Agricultural Harvesting Dataset**. The system runs entirely in Docker Compose and integrates MLflow for experiment tracking, a FastAPI inference server, and an automated drift-detection + retraining loop.

---

## Dataset

| Field | Value |
|---|---|
| File | `data/simulation_all.csv` |
| Rows | 284 |
| Columns | 15 |
| Scenarios | `0` = HumanOnly, `1` = WithRobot |

### Features used for training

| Feature | Description |
|---|---|
| `Humans` | Number of workers (1–12) |
| `ROW_N` | Crop row (1–3) |
| `RandomPosition` | Random position flag (0/1) |
| `Act_Ladder` | One-hot: MainActivity == ladder |
| `Act_Mixed` | One-hot: MainActivity == mixed |
| `Act_Picker` | One-hot: MainActivity == picker |

> `Act_*` columns are derived via `pd.get_dummies(df["MainActivity"], prefix="Act", drop_first=True)`.  
> `harv_ground` is the dropped reference level.

### Regression targets (4 per scenario)

| Key | CSV Column |
|---|---|
| `TotalRecollected` | `TotalRecollectedCrops_crop_units` |
| `CargoZoneProd` | `TotalProductionCargoZone_crop_units` |
| `TotalWorkload` | `TotalHumanWorkload_kcal` |
| `AvgProduction` | `AverageHumanProduction_crop_units` |

---

## Services (docker-compose.yml)

```
mlflow-server      → http://localhost:5000   Experiment tracking + model registry
inference-api      → http://localhost:8000   FastAPI: /predict /health /info /reload
drift-detector     → (background)            Detects drift; triggers retraining
model-trainer      → (one-shot job)          Trains & registers models on startup
test-runner        → (one-shot job)          Runs pytest test suite
```

All services mount `./data:/data:ro` so they share the same CSV.

---

## Training (model-trainer/train.py)

### Multi-model tournament

For each of the **8 (scenario, target) slots** the trainer:

1. **Phase 1 — Comparison**: cross-validates 12 candidates, logs each as a separate MLflow run tagged `phase=comparison`.
2. **Phase 2 — Winner**: re-fits the best CV-R² candidate on the full training set, logs a run tagged `phase=winner`, and registers it in the MLflow Model Registry.
3. **Phase 3 — Quality gate**: promotes to `Production` stage only if hold-out R² ≥ `QUALITY_GATE` (0.70).

### 12 candidate models

```
LinearRegression  Ridge         Lasso         ElasticNet
SVR               ExtraTrees    RandomForest  GradientBoosting
MLP               XGBoost       LightGBM      CatBoost
```

Each candidate is a `Pipeline(StandardScaler → Estimator)`.

### MLflow experiment structure

```
Experiment: hri-harvesting
  Run: HumanOnly_TotalRecollected_LinearRegression  (phase=comparison)
  Run: HumanOnly_TotalRecollected_Ridge              (phase=comparison)
  ...
  Run: HumanOnly_TotalRecollected_winner             (phase=winner)
  Run: WithRobot_TotalRecollected_winner             (phase=winner)
  ...
Registered models (8):
  hri-HumanOnly-TotalRecollected   → Production
  hri-HumanOnly-CargoZoneProd      → Production
  ...
```

### Logged metrics (winner runs)

| Metric | Description |
|---|---|
| `cv_r2_mean` | Mean 5-fold cross-validated R² |
| `cv_r2_std` | Std of 5-fold CV R² |
| `ho_r2` | Hold-out R² (20% test split) |
| `ho_rmse` | Hold-out RMSE |
| `ho_mae` | Hold-out MAE |

---

## Inference API (inference-api/app.py)

### Request schema

```json
{
  "scenario": 0,
  "workers":  6,
  "crop_row": 2,
  "rand_pos": 0,
  "activity": "harv_mixed"
}
```

| Field | Type | Range / Values |
|---|---|---|
| `scenario` | int | 0 (HumanOnly) or 1 (WithRobot) |
| `workers` | int | 1–12 |
| `crop_row` | int | 1–3 |
| `rand_pos` | int | 0 or 1 |
| `activity` | str | `harv_ground`, `harv_ladder`, `harv_mixed`, `harv_picker` |

### Response schema

```json
{
  "scenario_label":      "HumanOnly",
  "total_recollected":   87.4,
  "cargo_zone_prod":     42.1,
  "total_workload_kcal": 312.8,
  "avg_production":      14.6,
  "model_stage":         "Production",
  "loaded_at":           "2025-01-15T10:30:00"
}
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{ status, models_loaded (0–8) }` |
| GET | `/info` | Dataset + model metadata |
| POST | `/predict` | Run all 4 targets for a given scenario |
| POST | `/reload` | Hot-reload models from registry |

---

## Drift Detection (drift-detector/)

### Data drift (detector.py)

- **Reference**: 50 randomly sampled rows from the HRI dataset, queried through the API to get reference predictions.
- **Production buffer**: 30 rows sampled per cycle; from cycle 3 onward, drift is injected (double `Humans`, shift `ROW_N`).
- **KS test on 6 features**: flags `data_drift=True` if any feature's KS p-value < 0.05.
- **KS test on predictions**: compares production prediction distribution against reference for each of the 4 targets.

### Performance drift (performance_drift_detector.py)

Three-layer statistical check comparing current window metrics against a fixed baseline:

| Layer | Method | Trigger |
|---|---|---|
| Effect size | `|current − baseline| > threshold` | threshold = 0.05 |
| T-test | One-sample t-test, last 5 observations | p < 0.05 AND abs change > 0.02 |
| EWMA | Exponential moving average divergence | EWMA deviation > 0.07 |

**Drift declared** when ≥ 2 metrics (`r2`, `rmse`, `mae`) are flagged simultaneously.

### Retraining trigger (retrain_trigger.py)

When drift is detected the trigger:

1. Trains **8 challenger models** (one per slot) using `GradientBoostingRegressor` (sklearn-only — no extra deps in this container).
2. Loads current champion's `ho_r2` from the MLflow registry.
3. Promotes the challenger if `challenger_r2 >= champion_r2 + MIN_DELTA` (0.02).
4. Calls `POST /reload` on the inference API to swap models live.

---

## Test Suite (tests/)

### Test files

| File | Scope | What it tests |
|---|---|---|
| `test_data.py` | Data | Schema, 15 cols, no nulls, both scenarios, feature ranges, additive consistency |
| `test_model.py` | Model quality | R² ≥ 0.70, CV stability, beats DummyRegressor, reproducible, MLflow registry |
| `test_api.py` | API contract | HTTP 200/422, all activities/workers/crop_rows, latency SLA, error handling |
| `test_performance_drift.py` | Unit | R²/RMSE/MAE metrics, effect size, t-test, EWMA, monitor buffer management |
| `test_performance_drift_integration.py` | Integration | Gradual degradation, sudden shift, drift recovery, threshold sensitivity |

### test_model.py — test classes

| Class | Tests |
|---|---|
| `TestPerformanceGates` | `test_r2_above_gate`, `test_cv_stability` |
| `TestSanityChecks` | `test_beats_dummy`, `test_non_negative_predictions`, `test_reproducible`, `test_single_sample` |
| `TestMLFlowRegistry` | `test_model_in_registry`, `test_ho_r2_logged` |

### Running tests

```bash
# Inside test-runner container (runs automatically on docker compose up):
pytest tests/ -v

# Manually against a running stack:
docker compose exec test-runner pytest tests/ -v

# Single file:
docker compose exec test-runner pytest tests/test_model.py -v

# With custom quality gate:
docker compose run -e GATE_MIN_R2=0.75 test-runner pytest tests/test_model.py
```

---

## Quality Gate

| Parameter | Default | Set via |
|---|---|---|
| Minimum R² | 0.70 | `GATE_MIN_R2` env var / `--min_r2` CI input |

Applied in three places:
- `train.py`: refuses to promote a model to Production if hold-out R² < 0.70
- `test_model.py`: `TestPerformanceGates.test_r2_above_gate` fails the test suite
- GitHub Actions: `retrain.yml` input `min_r2` (default 0.70) passed as `GATE_MIN_R2`

---

## Docker Images

### model-trainer

```dockerfile
Base: python:3.11-slim
Packages: numpy, pandas, scikit-learn, mlflow, xgboost, lightgbm, catboost
System:   libgomp1 (required by LightGBM)
```

### inference-api

```dockerfile
Base: python:3.11-slim
Packages: fastapi, uvicorn, mlflow, scikit-learn, xgboost, lightgbm, catboost
System:   libgomp1 (required by LightGBM)
```

> Both images need xgboost/lightgbm/catboost because the **trainer saves** these models and the **API must deserialize** them. Mismatched packages cause `ModuleNotFoundError` at load time.

### drift-detector

```dockerfile
Base: python:3.11-slim
Packages: numpy, pandas, scipy, scikit-learn, mlflow
Files:    detector.py, retrain_trigger.py, performance_drift_detector.py
```

> Uses GradientBoosting (sklearn) as challenger — no need for xgboost/lightgbm/catboost in this image.

---

## CI/CD (.github/workflows/retrain.yml)

### Workflow inputs

| Input | Default | Description |
|---|---|---|
| `min_r2` | `0.70` | Quality gate threshold passed as `GATE_MIN_R2` |

### Pipeline steps

```
checkout → build images → docker compose up → wait for training
→ run test suite → check GATE_MIN_R2 → notify
```

---

## Quick Reference

### Start the stack

```bash
cd mlops-stack
bash start.sh
```

### Stop the stack

```bash
docker compose down
# Remove volumes too (wipes MLflow DB and registered models):
docker compose down -v
```

### Manual retrain

```bash
docker compose run --rm model-trainer python train.py
```

### Call the API

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"scenario":0,"workers":6,"crop_row":2,"rand_pos":0,"activity":"harv_mixed"}' \
  | python3 -m json.tool
```

### Check model registry

```bash
# Open browser:
open http://localhost:5000

# Or via CLI:
docker compose exec mlflow-server mlflow models list
```

### View logs

```bash
docker compose logs -f inference-api
docker compose logs -f drift-detector
docker compose logs -f model-trainer
```

### Reload models without restart

```bash
curl -s -X POST http://localhost:8000/reload | python3 -m json.tool
```

---

## File Map

```
mlops-stack/
├── data/
│   └── simulation_all.csv          HRI dataset (284 rows)
├── model-trainer/
│   ├── train.py                    Multi-model tournament (12 candidates, 8 slots)
│   ├── requirements.txt            + xgboost, lightgbm, catboost
│   └── Dockerfile                  + libgomp1
├── inference-api/
│   ├── app.py                      FastAPI: /predict /health /info /reload
│   ├── requirements.txt            + xgboost, lightgbm, catboost
│   └── Dockerfile                  + libgomp1
├── drift-detector/
│   ├── detector.py                 Data drift + KS test on predictions
│   ├── retrain_trigger.py          Champion/Challenger (GradientBoosting)
│   ├── performance_drift_detector.py  R²/RMSE/MAE drift detection
│   └── Dockerfile
├── tests/
│   ├── test_data.py                HRI schema validation
│   ├── test_model.py               Regression quality gates (8 models)
│   ├── test_api.py                 API contract tests
│   ├── test_performance_drift.py   Unit tests for drift detector
│   ├── test_performance_drift_integration.py  Integration tests
│   └── performance_drift_detector.py  Shared copy for test-runner container
├── docker-compose.yml
├── start.sh
├── CHANGELOG.md                    Session-by-session change log
└── ARCHITECTURE.md                 This file
```
