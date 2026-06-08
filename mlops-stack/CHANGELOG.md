# Changelog — HRI Dataset Integration

## Branch `feature/dataset-update`  |  2026-06-08

### Contexto
Reemplazo completo del dataset de juguete Iris (clasificación, 150 filas, 1 target)
por el **HRI Agricultural Harvesting Dataset** real (Vasconez & Auat Cheein, 2022,
*Biosystems Engineering* Vol. 223).  
El stack MLOps pasa de un problema de clasificación a uno de **regresión múltiple**:
2 escenarios × 4 targets = **8 modelos** en producción.

---

## Archivos modificados

### `data/simulation_all.csv` — NUEVO
- Dataset HRI Agricultural Harvesting añadido directamente al repositorio
- 284 filas, 15 columnas, codificación UTF-8
- Escenario 0 (Human-Only): 144 filas
- Escenario 1 (Human-Robot): 140 filas
- Columnas clave: `Scenario`, `Humans`, `ROW_N`, `RandomPosition`, `MainActivity`,
  `TotalRecollectedCrops_crop_units`, `TotalProductionCargoZone_crop_units`,
  `TotalHumanWorkload_kcal`, `AverageHumanProduction_crop_units`

---

### `model-trainer/train.py` — REESCRITO COMPLETAMENTE

**Antes:** Cargaba Iris con `load_iris()`, entrenaba un `RandomForestClassifier`, métricas de clasificación (accuracy, F1).

**Ahora:**
- Carga `simulation_all.csv` desde `DATASET_PATH` (variable de entorno)
- Pre-procesa `MainActivity` con one-hot encoding → columnas `Act_Ladder`, `Act_Mixed`, `Act_Picker`
- **Torneo multi-modelo**: 12 algoritmos candidatos evaluados con 5-fold CV
  - LinearRegression, Ridge, Lasso, ElasticNet
  - SVR
  - ExtraTrees, RandomForest, GradientBoosting
  - MLP (Red Neuronal)
  - XGBoost, LightGBM, CatBoost
- El candidato con **mayor R² medio en CV** gana y se registra en MLflow
- 2 fases en MLflow: `phase=comparison` (1 run por candidato) y `phase=winner` (run registrado)
- Métricas logueadas: `cv_r2_mean`, `cv_r2_std`, `ho_r2`, `ho_rmse`, `ho_mae`
- Cada winner run incluye `cmp_<algoritmo>_cv_r2` para tabla comparativa del paper
- Quality gate: R² ≥ 0.70 para pasar a Production
- Modelos registrados: `hri-{HumanOnly|WithRobot}-{TotalRecollected|CargoZoneProd|TotalWorkload|AvgProduction}`

---

### `model-trainer/requirements.txt` — MODIFICADO

Paquetes añadidos:
```
xgboost==2.0.3
lightgbm==4.3.0
catboost==1.2.3
```

---

### `model-trainer/Dockerfile` — MODIFICADO

Nueva capa de instalación para los tres nuevos paquetes (separada del resto para
aprovechar caché de Docker):
```dockerfile
RUN pip install xgboost==2.0.3 lightgbm==4.3.0 catboost==1.2.3
```

---

### `docker-compose.yml` — MODIFICADO

Cambios en todos los servicios que necesitan acceso al dataset:

| Servicio | Cambio |
|---|---|
| `model-trainer` | `DATASET_PATH: /data/simulation_all.csv` + volumen `./data:/data:ro` |
| `inference-api` | Eliminado `MODEL_NAME: iris-classifier`, agregado `DATASET_PATH`, volumen `./data:/data:ro` |
| `drift-detector` | Eliminado `MODEL_NAME` y `CHI2_P_VALUE_THRESHOLD`, agregado `DATASET_PATH`, volumen `./data:/data:ro` |
| `test-runner` | Eliminados `GATE_MIN_ACCURACY`, `GATE_MIN_F1`; agregados `DATASET_PATH`, `GATE_MIN_R2: "0.70"`, volumen `./data:/data:ro` |

---

### `inference-api/app.py` — REESCRITO COMPLETAMENTE

**Antes:** Endpoint único `/predict` con `instances: [[f1, f2, f3, f4]]` para Iris.

**Ahora:**
- Carga los 8 modelos al arrancar (`load_all_models()`)
- Endpoints:
  - `GET /health` — estado del servidor, informa cuántos modelos están cargados (0-8)
  - `GET /info` — metadatos de los modelos en producción
  - `POST /predict` — recibe escenario + parámetros de cosecha, devuelve los 4 targets
  - `POST /reload` — recarga los modelos desde MLflow sin reiniciar el contenedor
- Schema de entrada:
  ```json
  {
    "scenario": 0,
    "workers": 6,
    "crop_row": 2,
    "rand_pos": 0,
    "activity": "harv_mixed"
  }
  ```
- Schema de salida:
  ```json
  {
    "scenario_label": "HumanOnly",
    "total_recollected": 142.3,
    "cargo_zone_prod": 98.1,
    "total_workload_kcal": 2340.5,
    "avg_production": 23.7,
    "model_stage": "Production",
    "loaded_at": "..."
  }
  ```

---

### `tests/test_data.py` — REESCRITO COMPLETAMENTE

**Antes:** Validaba el dataset Iris.

**Ahora:** Valida `simulation_all.csv` con las siguientes clases de test:

| Clase | Qué verifica |
|---|---|
| `TestSchema` | 15 columnas, sin nulos, nombres correctos |
| `TestScenario` | Ambos escenarios presentes (0 y 1) |
| `TestFeatureRanges` | `Humans` ∈ {1,3,6,8,10,12}, `ROW_N` ∈ {1,2,3}, `MainActivity` ∈ {harv_ground/ladder/mixed/picker} |
| `TestTargetRanges` | Los 4 targets son no-negativos |
| `TestDataIntegrity` | Consistencia aditiva (TotalRecollected = suma de 4 componentes), `ExperimentID` único, escenario Human-Only sin producción de robot |

---

### `tests/test_model.py` — REESCRITO COMPLETAMENTE

**Antes:** Cargaba Iris, testaba `accuracy`, `f1`, `predict_proba`, clases de clasificación.

**Ahora:** Testea los **8 modelos de regresión** en producción:

| Clase | Qué verifica |
|---|---|
| `TestPerformanceGates` | R² ≥ 0.70 en holdout, estabilidad CV (std < 0.15) |
| `TestSanityChecks` | Supera a `DummyRegressor` por ≥ 0.30 R², predicciones no-negativas, resultado reproducible, acepta muestra única |
| `TestMLFlowRegistry` | Modelo en registry en stage Production, métrica `ho_r2` logueada |

---

### `.github/workflows/retrain.yml` — MODIFICADO

- Input `min_accuracy` (default 0.90) → `min_r2` (default 0.70)
- Input `min_f1` (default 0.90) → eliminado
- Variable de entorno del test-runner: `GATE_MIN_R2` en lugar de `GATE_MIN_ACCURACY`/`GATE_MIN_F1`

---

### `start.sh` — MODIFICADO

- Ejemplo de `curl` al final corregido para el nuevo schema de la API:
  ```bash
  # Antes (Iris)
  curl -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}'

  # Ahora (HRI)
  curl -d '{"scenario":0,"workers":6,"crop_row":2,"rand_pos":0,"activity":"harv_mixed"}'
  ```

---

## Archivos pendientes de actualizar

Los siguientes archivos aún usan `load_iris()` o métricas de clasificación y **no fueron modificados** en esta sesión:

| Archivo | Problema |
|---|---|
| `drift-detector/detector.py` | Usa `load_iris()` para datos de referencia, prueba Chi² (clasificación) |
| `drift-detector/retrain_trigger.py` | Usa `load_iris()` para entrenar challenger |
| `drift-detector/performance_drift_detector.py` | Puede necesitar ajuste para métricas de regresión |
| `tests/test_api.py` | Usa schema antiguo `instances: [[...]]` |
| `tests/test_performance_drift.py` | Usa `load_iris()` |
| `tests/test_performance_drift_integration.py` | Usa `load_iris()` |

---

## Commits en esta rama

| Hash | Descripción |
|---|---|
| `9667d87` | Replace Iris dataset with HRI Agricultural Harvesting Dataset |
| `bb09461` | Update test_model.py and retrain.yml for HRI regression |
| `9bafd90` | Add multi-model comparison to training pipeline |
| `da2620d` | Fix start.sh predict example for HRI API schema |
| `a9c1b33` | Fix bad mlflow import order in test_model.py model_ctx fixture |
