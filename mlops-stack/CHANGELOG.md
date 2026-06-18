# Changelog — HRI Dataset Integration

---

## Block 4 — Generic Inference API + Multi-Experiment  |  2026-06-18

### Gap 1 — API genérica `POST /models/{name}/predict`

**Problema:** el endpoint `/predict` estaba hardcodeado para HRI harvesting — features fijas, encoding de actividad HRI, respuesta con 4 campos específicos. No podía servir ningún otro modelo registrado en MLflow.

**Solución:** tres nuevos endpoints genéricos en `inference-api/app.py`:

| Endpoint | Descripción |
|---|---|
| `GET /models` | Lista todos los modelos con versión `Production` en el registry |
| `GET /models/{name}` | Metadata de un modelo específico (version, stage, run_id) |
| `POST /models/{name}/predict` | Predicción con cualquier modelo sklearn del registry |

**`POST /models/{name}/predict`** acepta `{"features": {"col": value, ...}, "stage": "Production"}` — construye un DataFrame con los features, llama al modelo, devuelve `{"model": name, "stage": stage, "prediction": float}`. El modelo se carga en el primer request y se guarda en `_model_cache` para requests posteriores.

**`POST /reload`** ahora también limpia `_model_cache` (fuerza recarga de modelos genéricos en el siguiente request).

### Gap 2 — Multi-experimento

**Solución:** el API genérico da soporte multi-experimento automáticamente — no hay cambios de código necesarios para añadir nuevos experimentos. Para servir modelos de un segundo experimento:
1. Entrenar con diferente `MODEL_CONFIG` (e.g., `iris-classifier.yaml`) → registra modelos con otro prefijo en MLflow
2. `GET /models` los lista automáticamente
3. `POST /models/{nombre}/predict` los sirve sin reiniciar el API

**Backward compatibility:** los endpoints `/predict`, `/reload`, `/health`, `/info` no cambian. Los tests existentes siguen pasando.

**Tests añadidos** (`tests/test_api.py`):
- `TestGenericModelList` — verifica `/models` lista correctamente los 8 modelos HRI
- `TestGenericModelGet` — verifica `/models/{name}` devuelve metadata correcta + 404 para no existentes
- `TestGenericPredict` — verifica predicción, schema, tipo numérico, latencia SLA, cache hit

**Prometheus:** nueva métrica `inference_generic_predictions_total{model}` — contador por nombre de modelo.

---

## Block 3 — Prometheus + Grafana (Observabilidad)  |  2026-06-18

### Gap 6 — Métricas de operación con Prometheus + Grafana

**Problema:** el stack no tiene visibilidad en tiempo real de latencia, tasa de errores, estado de modelos ni detección de drift. Sin métricas operacionales no es posible hacer SLA ni root cause analysis.

**Solución implementada:**

**Instrumentación (`prometheus-client==0.20.0`):**
- `inference-api/app.py` — 4 métricas nuevas expuestas en `GET /metrics`:
  - `inference_requests_total{endpoint, status}` — contador por endpoint y código HTTP
  - `inference_request_duration_seconds{endpoint}` — histograma de latencia (buckets hasta 2.5s)
  - `inference_models_loaded` — gauge: cuántos modelos están actualmente cargados (objetivo: 8)
  - `inference_predictions_total{scenario, activity}` — contador de predicciones por escenario y actividad
- `drift-detector/detector.py` — 4 métricas expuestas en `:9091` vía `start_http_server`:
  - `drift_ks_pval{feature}` — gauge: p-value del KS test por feature (< 0.05 = drift)
  - `drift_detected_total{type}` — contador: detecciones de drift data/concept
  - `drift_consecutive_windows` — gauge: ventanas consecutivas con drift activo
  - `drift_retrain_triggered_total{reason}` — contador: reentrenamientos disparados por razón

**Infraestructura:**
- `prometheus/prometheus.yml` — scraping de `inference-api:8000/metrics` y `drift-detector:9091` cada 15s
- `grafana/provisioning/` — datasource Prometheus + provider de dashboards auto-provisionados
- `grafana/dashboards/mlops.json` — dashboard con 8 paneles:
  - Fila 1 (stats): Models Loaded, Request Rate, Consecutive Drift Windows, Retrain Triggers
  - Fila 2 (time series): Latencia p50/p95/p99 de `/predict`, Requests/s por endpoint
  - Fila 3 (time series): KS p-values por feature con línea threshold 0.05, Drift detections por tipo

**Docker Compose:**
- Servicio `prometheus` (prom/prometheus:v2.51.2) — `127.0.0.1:9090`, retención 30d
- Servicio `grafana` (grafana/grafana:10.4.2) — `127.0.0.1:3000`, acceso vía nginx `/grafana/`
- Volúmenes: `prometheus-data`, `grafana-data`

**nginx:** nueva `location /grafana/` — proxy a `grafana:3000` con sub-path routing correcto.

**Variables de entorno añadidas:**
- `GRAFANA_ADMIN_PASSWORD` (`.env.example`)

**URLs de acceso (tras `docker compose up`):**
| Panel | URL |
|---|---|
| Grafana | https://localhost/grafana/ |
| Prometheus UI | http://localhost:9090 (solo localhost) |
| Inference metrics | http://localhost:8000/metrics (solo localhost) |
| Drift metrics | http://localhost:9091/metrics (solo localhost) |

---

## Block 2 — PostgreSQL + DVC  |  2026-06-17

### Gap 5 — PostgreSQL como backend de MLflow (reemplaza SQLite)

**Problema:** SQLite no soporta escrituras concurrentes; en producción con múltiples workers o runs paralelos puede corromperse o bloquearse.

**Cambios:**

| Archivo | Cambio |
|---|---|
| `mlflow/Dockerfile` | Añade `psycopg2-binary==2.9.9` para que MLflow pueda conectarse a PostgreSQL |
| `mlflow/entrypoint.sh` | Lee `POSTGRES_URI` del entorno; si está vacío cae back a SQLite (dev local sin `.env`) |
| `docker-compose.yml` | Nuevo servicio `postgres` (imagen `postgres:16-alpine`) con healthcheck, volumen `postgres-data`. MLflow depende de `postgres: condition: service_healthy` |
| `.env.example` | Añade `POSTGRES_PASSWORD` |

**Primera vez con PostgreSQL:** el volumen MLflow anterior (SQLite) es incompatible. Correr `docker compose down -v` antes de levantar con el nuevo stack.

**Variables de entorno nuevas:**

| Variable | Dónde se usa | Valor ejemplo |
|---|---|---|
| `POSTGRES_PASSWORD` | docker-compose → postgres + mlflow URI | `mi-clave-segura` |
| `POSTGRES_URI` | mlflow container (seteado internamente por compose) | `postgresql://mlflow:<PW>@postgres:5432/mlflow` |

---

### Gap 7 — DVC para versionar `data/simulation_all.csv`

**Problema:** El dataset estaba en git directamente. Para datasets más grandes o en producción, los archivos de datos no deben estar en el repositorio.

**Cambios:**

| Archivo | Cambio |
|---|---|
| `.dvc/config` | Remote local en `/opt/mlops-dvc-store` |
| `data/simulation_all.csv.dvc` | Puntero DVC (md5 + tamaño del CSV) |
| `data/.gitignore` | Excluye `simulation_all.csv` del repositorio git |
| `.github/workflows/ci.yml` | Paso `dvc pull` antes de build |
| `.github/workflows/cd.yml` | Paso `dvc pull` después de rsync, antes de build |
| `setup_runner.sh` | Instala DVC y crea `/opt/mlops-dvc-store` |

**Flujo de datos con DVC:**

```
Primera vez (máquina con el CSV):
  cd mlops-stack
  dvc init
  dvc add data/simulation_all.csv   # crea data/simulation_all.csv.dvc
  sudo mkdir -p /opt/mlops-dvc-store
  dvc push                           # sube el CSV al remote local

CI/CD y otras máquinas:
  dvc pull                           # descarga el CSV del remote
```

**Remote configurado:** `/opt/mlops-dvc-store` (directorio local en el servidor). Para producción real se cambiaría a S3/GCS en `.dvc/config`.

---

## Block 1 — TLS + Autenticación API  |  2026-06-17

### Gap 4 — TLS con nginx (HTTPS)

**Problema:** Todo el tráfico viajaba en HTTP plano.

**Cambios:**

| Archivo | Cambio |
|---|---|
| `nginx/Dockerfile` | Instala `openssl`, genera certificado auto-firmado en build time (`/etc/nginx/ssl/server.crt`) |
| `nginx/nginx.conf` | Servidor en puerto 80 redirige 301 → HTTPS. Servidor en 443 con `ssl_protocols TLSv1.2 TLSv1.3`, `ssl_ciphers HIGH:!aNULL:!MD5`. Pasa header `X-API-Key` al backend |
| `docker-compose.yml` | Puerto `443:443` añadido a nginx |

Certificado auto-firmado → usar `-k` en curl. Para producción usar Let's Encrypt.

### Gap 3 — Autenticación con API key

**Problema:** Cualquier cliente podía llamar a `/predict` y `/reload` sin autenticarse.

**Cambios:**

| Archivo | Cambio |
|---|---|
| `inference-api/app.py` | `APIKeyHeader("X-API-Key")` como dependencia FastAPI en `/predict` y `/reload`. Auth desactivada si `API_KEY=""` (dev local) |
| `drift-detector/detector.py` | Envía `X-API-Key` en llamadas internas a `/predict` |
| `drift-detector/retrain_trigger.py` | Envía `X-API-Key` en llamadas a `/reload` |
| `tests/test_api.py` | `_AUTH_HEADERS` inyectado en todas las llamadas HTTP del test suite |
| `.github/workflows/` | `API_KEY=${{ secrets.API_KEY }}` pasado como `-e` en todos los `docker compose run test-runner` |
| `.env.example` | Documenta `API_KEY` |

**Endpoints públicos** (sin auth): `GET /health`, `GET /info`  
**Endpoints protegidos**: `POST /predict`, `POST /reload`

**Secret de GitHub necesario:** `API_KEY` — Settings → Secrets → Actions → New repository secret.

---

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

## Comandos de uso

### Prerequisitos
- Docker Desktop corriendo en Windows con integración WSL2 activada
- Terminal WSL Ubuntu (no PowerShell)

```bash
# Ir al directorio del stack
cd ~/ai-deployment/mlops-stack
```

---

### Levantar el stack completo (primera vez o tras cambios)

```bash
bash start.sh
```

Hace en orden: build de imágenes → MLflow → entrenamiento de los 12 candidatos × 8 slots → Inference API + Nginx.

---

### Apagar el stack

```bash
# Apaga y elimina contenedores — los modelos en MLflow quedan guardados
docker compose down

# Apaga Y borra todos los datos (modelos, runs de MLflow) — requiere reentrenar
docker compose down -v
```

| Comando | Contenedores | Volúmenes (modelos) | Imágenes |
|---|---|---|---|
| `docker compose down` | eliminados | ✅ intactos | ✅ intactas |
| `docker compose down -v` | eliminados | ❌ borrados | ✅ intactas |
| `docker compose stop` | pausados | ✅ intactos | ✅ intactas |

---

### Reentrenar sin bajar el stack

```bash
docker compose run --rm model-trainer
```

---

### Recargar modelos en la API sin reiniciar

```bash
curl -X POST http://localhost:8000/reload
```

---

### Correr los tests

```bash
docker compose run --rm test-runner
```

---

### Ver logs en tiempo real

```bash
# Todos los servicios
docker compose logs -f

# Un servicio específico
docker compose logs -f inference-api
docker compose logs -f model-trainer
docker compose logs -f mlflow-server
```

---

### Hacer predicciones

```bash
# Escenario 0 (Human-Only), 6 trabajadores, fila 2, actividad mixta
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"scenario":0,"workers":6,"crop_row":2,"rand_pos":0,"activity":"harv_mixed"}'

# Escenario 1 (Human-Robot), 10 trabajadores, fila 1, recolección en suelo
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"scenario":1,"workers":10,"crop_row":1,"rand_pos":0,"activity":"harv_ground"}'
```

**Valores válidos por campo:**

| Campo | Opciones |
|---|---|
| `scenario` | `0` = HumanOnly, `1` = WithRobot |
| `workers` | 1, 3, 6, 8, 10, 12 |
| `crop_row` | 1, 2, 3 |
| `rand_pos` | 0 = No, 1 = Sí |
| `activity` | `harv_ground`, `harv_ladder`, `harv_mixed`, `harv_picker` |

---

### URLs del stack

| Servicio | URL |
|---|---|
| MLflow UI (vía nginx, HTTPS) | https://localhost/mlflow/ |
| MLflow UI (directo, dev) | http://localhost:5001 |
| Inference API vía nginx | https://localhost/api/ |
| Inference API (directo, dev) | http://localhost:8000 |
| API docs interactivos (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

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
| `751a4a7` | Add CHANGELOG.md documenting HRI dataset integration changes |
| `13ca2a4` | Fix CRLF line endings in shell scripts, add .gitattributes |
| `44286e6` | Fix missing libgomp1 for LightGBM in model-trainer Docker image |
| `018b8e5` | Add XGBoost/LightGBM/CatBoost to inference-api image |
