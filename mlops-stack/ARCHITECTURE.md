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
| `ho_smape` | Hold-out sMAPE (%) — robust to near-zero targets |
| `ho_max_error` | Hold-out worst-case single prediction error |

### Logged artifacts (winner runs)

| Artifact | Format | Description |
|---|---|---|
| `feature_ranking.json` | JSON | Permutation importance for each of the 6 features |
| `model/` | MLflow sklearn | Serialized Pipeline (StandardScaler + best estimator) |

Example `feature_ranking.json`:
```json
{
  "Humans":          {"importance_mean": 0.42, "importance_std": 0.03, "rank": 1},
  "ROW_N":           {"importance_mean": 0.31, "importance_std": 0.02, "rank": 2},
  "Act_Mixed":       {"importance_mean": 0.18, "importance_std": 0.01, "rank": 3},
  "RandomPosition":  {"importance_mean": 0.05, "importance_std": 0.01, "rank": 4},
  "Act_Ladder":      {"importance_mean": 0.03, "importance_std": 0.01, "rank": 5},
  "Act_Picker":      {"importance_mean": 0.01, "importance_std": 0.00, "rank": 6}
}
```

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

**Drift declared** when ≥ 2 metrics (`r2`, `rmse`, `mae`, `smape`, `max_error`) are flagged simultaneously.

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

| Class | Test | What validates |
|---|---|---|
| `TestPerformanceGates` | `test_r2_gate` | Hold-out R² ≥ `GATE_MIN_R2` (default 0.70) |
| `TestPerformanceGates` | `test_smape_gate` | Hold-out sMAPE < `GATE_MAX_SMAPE` (default 20%) |
| `TestPerformanceGates` | `test_max_error_finite` | Worst-case error is finite (no `inf`/`nan`) |
| `TestPerformanceGates` | `test_cv_stability` | CV std < 0.15 and CV mean ≥ `GATE_MIN_R2` |
| `TestSanityChecks` | `test_better_than_dummy` | Model beats mean-baseline by at least 0.30 R² |
| `TestSanityChecks` | `test_non_negative_predictions` | All predictions ≥ 0 (crops/workload can't be negative) |
| `TestSanityChecks` | `test_single_sample` | Model accepts 1 row and returns a finite scalar |
| `TestSanityChecks` | `test_reproducible` | Same input always produces same output |
| `TestMLFlowRegistry` | `test_model_in_registry` | Model exists in registry at `Production` stage |
| `TestMLFlowRegistry` | `test_all_metrics_logged` | All 5 scalar metrics exist in the MLflow run |
| `TestMLFlowRegistry` | `test_feature_ranking_artifact` | `feature_ranking.json` artifact exists in the run |
| `TestMLFlowRegistry` | `test_r2_metric_logged` | Logged `ho_r2` ≥ `GATE_MIN_R2` |

### Running validation

```bash
# Full test suite (all 5 files, all 8 models):
docker compose run --rm test-runner

# Only model quality gates:
docker compose run --rm test-runner pytest tests/test_model.py -v

# Only API contract tests:
docker compose run --rm test-runner pytest tests/test_api.py -v

# Only data validation:
docker compose run --rm test-runner pytest tests/test_data.py -v

# Only drift detector unit tests:
docker compose run --rm test-runner pytest tests/test_performance_drift.py -v

# Stricter quality gate (R² ≥ 0.80, sMAPE < 10%):
docker compose run --rm \
  -e GATE_MIN_R2=0.80 \
  -e GATE_MAX_SMAPE=10.0 \
  test-runner pytest tests/test_model.py -v
```

---

## Validation Flow

Validation happens in **three independent layers**, each with a different purpose:

```
bash start.sh
     │
     ├─ 1. TRAINING GATE (automático, bloqueante)
     │      train.py  ──►  ho_r2 < 0.70 → modelo NO se registra
     │
     ├─ 2. TEST SUITE (manual, bajo demanda)
     │      docker compose run --rm test-runner
     │            ├── test_data.py              validación del dataset
     │            ├── test_model.py             quality gates sobre los 8 modelos
     │            ├── test_api.py               contrato HTTP de la API
     │            ├── test_performance_drift.py tests unitarios del detector
     │            └── test_performance_drift_integration.py  escenarios de drift
     │
     └─ 3. DRIFT MONITORING (automático, continuo en background)
            docker compose up -d drift-detector
                  ├── KS test sobre features      → data drift
                  ├── KS test sobre predicciones  → concept drift
                  └── R²/RMSE/MAE/sMAPE/MaxErr    → performance drift
                        └── si detecta drift → entrena challenger → promueve si mejora
```

### Capa 1 — Training gate

Se ejecuta automáticamente dentro de `train.py` al correr `bash start.sh`.

| Condición | Resultado |
|---|---|
| `ho_r2 >= 0.70` | Modelo promovido a `Production` en MLflow |
| `ho_r2 < 0.70` | Modelo registrado pero **no promovido** (queda en `None` stage) |

Para cambiar el umbral:
```bash
docker compose run -e GATE_MIN_R2=0.80 --rm model-trainer python train.py
```

### Capa 2 — Test suite

Se ejecuta manualmente después del training.

```bash
# Correr toda la suite:
docker compose run --rm test-runner

# Solo quality gates del modelo (R², sMAPE, Max Error, CV, sanity checks):
docker compose run --rm test-runner pytest tests/test_model.py -v

# Con umbrales personalizados:
docker compose run --rm \
  -e GATE_MIN_R2=0.80 \
  -e GATE_MAX_SMAPE=10.0 \
  test-runner pytest tests/test_model.py -v
```

**Métricas evaluadas por el test suite:**

| Métrica | Gate | Archivo |
|---|---|---|
| R² (hold-out) | ≥ 0.70 | `test_model.py` |
| sMAPE (hold-out) | < 20% | `test_model.py` |
| Max Error | debe ser finito | `test_model.py` |
| CV R² std | < 0.15 | `test_model.py` |
| R² vs DummyRegressor | modelo ≥ dummy + 0.30 | `test_model.py` |
| Predicciones negativas | 0 permitidas | `test_model.py` |
| Métricas en MLflow | ho_r2, ho_rmse, ho_mae, ho_smape, ho_max_error | `test_model.py` |
| Artefacto feature ranking | `feature_ranking.json` debe existir | `test_model.py` |
| HTTP 200 en `/predict` | todas las actividades/workers/crop_rows | `test_api.py` |
| HTTP 422 en inputs inválidos | scenario fuera de rango, activity inválida | `test_api.py` |
| Latencia | < 500ms por predicción | `test_api.py` |
| Dataset schema | 15 cols, sin nulos, ambos scenarios | `test_data.py` |

### Capa 3 — Drift monitoring

Se ejecuta en background de forma continua.

```bash
docker compose up -d drift-detector
docker compose logs -f drift-detector
```

| Tipo de drift | Método estadístico | Trigger |
|---|---|---|
| Data drift | KS test, p < 0.05 en cualquier feature | Log + alerta |
| Concept drift | KS test sobre distribución de predicciones | Log + alerta |
| Performance drift | Effect size + t-test + EWMA sobre R²/RMSE/MAE/sMAPE/MaxErr | Reentrenamiento automático |

Cuando se detecta performance drift: entrena un challenger con `GradientBoostingRegressor`, compara `ho_r2` contra el champion actual, y promueve si `challenger_r2 >= champion_r2 + 0.02`. Luego llama `POST /reload` para actualizar los modelos en la API sin reiniciar el servicio.

### Cuándo hacer `docker compose down -v`

| Situación | Comando |
|---|---|
| Cambios en `train.py` (nuevas métricas, nuevos modelos) | `docker compose down -v && bash start.sh` |
| Cambios solo en `app.py`, tests, o detector | `docker compose down && bash start.sh` |
| Solo reconstruir imágenes sin reentrenar | `docker compose build && docker compose up -d` |

> `-v` borra el volumen `mlflow-data` (base de datos + modelos registrados). Úsalo cuando los modelos en producción necesiten regenerarse con los nuevos cambios de código.

## Quality Gates — Referencia rápida

| Parámetro | Default | Variable de entorno |
|---|---|---|
| R² mínimo | 0.70 | `GATE_MIN_R2` |
| sMAPE máximo | 20% | `GATE_MAX_SMAPE` |

Aplicados en:
- `train.py` → gate de promoción a Production
- `test_model.py` → falla la suite si no se cumple
- `retrain.yml` (CI) → input `min_r2` pasado como `GATE_MIN_R2`

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
