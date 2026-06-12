# MLOps Stack — HRI Agricultural Harvesting

Stack de despliegue en producción para modelos de regresión ML, containerizado con Docker Compose y respaldado por MLflow.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                         Docker Network                       │
│                                                             │
│  ┌────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │   Nginx    │───▶│  MLFlow      │    │ Model Trainer  │  │
│  │ :80        │    │  Tracking    │◀───│ (one-shot job) │  │
│  │            │    │  Server :5001│    └────────────────┘  │
│  │            │    │              │             │           │
│  │            │    │  - UI        │    registers model      │
│  │            │    │  - REST API  │             │           │
│  │            │    │  - Registry  │◀────────────┘           │
│  │            │    └──────────────┘                         │
│  │            │                                             │
│  │            │    ┌──────────────┐                         │
│  │            │───▶│  Inference   │                         │
│  │            │    │  API  :8000  │                         │
│  │            │    │  (FastAPI)   │                         │
│  └────────────┘    └──────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Dataset

**HRI Agricultural Harvesting Dataset** (Vasconez & Auat Cheein, 2022, *Biosystems Engineering* Vol. 223)

- 284 filas, 15 columnas
- 2 escenarios: `HumanOnly` (0), `WithRobot` (1)
- 4 targets de regresión: `TotalRecollected`, `CargoZoneProd`, `TotalWorkload`, `AvgProduction`
- 8 modelos en producción (2 escenarios × 4 targets)

## Servicios

| Servicio         | Puerto | Descripción                                    |
|-----------------|--------|------------------------------------------------|
| `mlflow`        | 5001   | MLFlow Tracking Server + Model Registry        |
| `inference-api` | 8000   | FastAPI que sirve los 8 modelos en producción  |
| `nginx`         | 80     | Reverse proxy (unifica acceso)                 |
| `model-trainer` | —      | Job de entrenamiento (se ejecuta una vez)      |

## Requisitos

- Docker ≥ 24
- Docker Compose ≥ v2
- 4 GB RAM libres

## Inicio rápido

```bash
# Opción 1: script automático (recomendado)
bash start.sh

# Opción 2: manual paso a paso
docker compose build
docker compose up -d mlflow
docker compose run --rm model-trainer
docker compose up -d inference-api nginx
```

## URLs

| Recurso        | URL                           |
|---------------|-------------------------------|
| MLFlow UI     | http://localhost:5001         |
| API docs      | http://localhost:8000/docs    |
| Predict       | http://localhost:8000/predict |
| Health check  | http://localhost:8000/health  |

## Hacer una predicción

```bash
# Escenario 0 = HumanOnly | 6 trabajadores | fila 2 | actividad mixta
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"scenario":0,"workers":6,"crop_row":2,"rand_pos":0,"activity":"harv_mixed"}'
```

**Respuesta esperada:**
```json
{
  "scenario": 0,
  "predictions": {
    "TotalRecollected": 48.7,
    "CargoZoneProd":    31.2,
    "TotalWorkload":     0.0,
    "AvgProduction":     8.1
  },
  "model_versions": {
    "TotalRecollected": "3",
    "CargoZoneProd":    "2",
    "TotalWorkload":    "2",
    "AvgProduction":    "2"
  },
  "model_stage": "Production"
}
```

## Features

| Feature         | Descripción                                      |
|----------------|--------------------------------------------------|
| `Humans`        | Número de trabajadores humanos                  |
| `ROW_N`         | Número de fila del cultivo                      |
| `RandomPosition`| Posición aleatoria (0/1)                        |
| `Act_Ladder`    | Actividad: escalera (one-hot)                   |
| `Act_Mixed`     | Actividad: mixta (one-hot)                      |
| `Act_Picker`    | Actividad: recolector (one-hot)                 |

> `harv_ground` es la categoría de referencia (todas las Act_* en 0).

## Métricas de evaluación

Cada modelo es evaluado con 6 métricas:

| Métrica      | Gate        | Descripción                                       |
|-------------|-------------|---------------------------------------------------|
| R²          | ≥ 0.70      | Varianza explicada                                |
| RMSE        | —           | Error cuadrático medio                            |
| MAE         | —           | Error absoluto medio                              |
| sMAPE       | < 20%       | Error porcentual simétrico (robusto a ceros)      |
| Max Error   | —           | Peor error individual                             |
| Feature Ranking | —       | Importancia relativa por permutación              |

## Torneo multi-modelo

12 algoritmos compiten por escenario × target:

```
LinearRegression · Ridge · Lasso · ElasticNet · SVR
ExtraTrees · RandomForest · GradientBoosting · MLP
XGBoost · LightGBM · CatBoost
```

El ganador (mejor R² en cross-validation) se registra en MLflow y se promueve a `Production`.

## Monitoreo

### Data Drift
Detecta cambios en la distribución de features usando Kolmogorov-Smirnov test.

### Performance Drift
Monitorea degradación en métricas de regresión:
- **R²**: Varianza explicada
- **RMSE**: Error cuadrático medio
- **MAE**: Error absoluto medio
- **sMAPE**: Error porcentual simétrico
- **Max Error**: Peor predicción individual

**Métodos estadísticos:**
- **Effect Size**: Cambio absoluto > umbral (default 5%)
- **T-test**: Media de últimas 5 ventanas vs baseline
- **EWMA**: Tendencia sostenida via moving average (α=0.3)

## Suite de tests

```bash
docker compose run --rm test-runner
```

| Nivel | Archivo                            | Qué valida                        |
|------|------------------------------------|-----------------------------------|
| 1    | `test_data.py`                     | Schema del dataset HRI            |
| 2    | `test_model.py`                    | R² ≥ 0.70, sMAPE < 20% por modelo |
| 3    | `test_api.py`                      | HTTP 200, latencia < 500ms        |
| 4    | `test_performance_drift.py`        | Lógica del detector               |

## Actualizar el modelo

```bash
# Re-entrenar (crea nuevas versiones en el Registry)
docker compose run --rm model-trainer

# Recargar modelo en la API sin reiniciar el contenedor
curl -X POST http://localhost:8000/reload
```

## Detener el stack

```bash
docker compose down          # detiene contenedores (mantiene datos)
docker compose down -v       # detiene + elimina volúmenes (borra modelos)
```

> Usar `-v` cuando se modifica `train.py` para forzar re-entrenamiento limpio.

## Estructura del proyecto

```
mlops-stack/
├── docker-compose.yml
├── start.sh
├── data/
│   └── simulation_all.csv          ← HRI dataset
├── model-trainer/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── train.py                    ← torneo 12 modelos + registro MLflow
├── inference-api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                      ← FastAPI + 8 modelos de producción
├── drift-detector/
│   ├── Dockerfile
│   ├── detector.py                 ← data drift (KS test)
│   ├── performance_drift_detector.py
│   ├── retrain_trigger.py
│   └── requirements.txt
├── nginx/
│   ├── Dockerfile
│   └── nginx.conf
└── tests/
    ├── Dockerfile
    ├── run_tests.py                ← orquestador QA (4 niveles)
    ├── test_data.py
    ├── test_model.py
    ├── test_api.py
    ├── test_performance_drift.py
    ├── test_performance_drift_integration.py
    └── entrypoint.sh
```
