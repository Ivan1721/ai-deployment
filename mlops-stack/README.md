# MLOps Stack con MLFlow + Docker

Stack de despliegue en producción para modelos ML, containerizado con Docker Compose.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                         Docker Network                       │
│                                                             │
│  ┌────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │   Nginx    │───▶│  MLFlow      │    │ Model Trainer  │  │
│  │ :80        │    │  Tracking    │◀───│ (one-shot job) │  │
│  │            │    │  Server :5000│    └────────────────┘  │
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

## Servicios

| Servicio        | Puerto | Descripción                                    |
|----------------|--------|------------------------------------------------|
| `mlflow`       | 5000   | MLFlow Tracking Server + Model Registry        |
| `inference-api`| 8000   | FastAPI que sirve el modelo en producción       |
| `nginx`        | 80     | Reverse proxy (unifica acceso)                 |
| `model-trainer`| —      | Job de entrenamiento (se ejecuta una vez)       |

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

| Recurso        | URL                          |
|---------------|------------------------------|
| MLFlow UI     | http://localhost:5000        |
| API docs      | http://localhost:8000/docs   |
| Predict       | http://localhost:8000/predict|
| Health check  | http://localhost:8000/health |

## Hacer una predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      [5.1, 3.5, 1.4, 0.2],
      [6.7, 3.0, 5.2, 2.3],
      [4.9, 3.1, 1.5, 0.1]
    ]
  }'
```

**Respuesta esperada:**
```json
{
  "predictions": [
    {"class_id": 0, "class_name": "setosa",     "probability": 1.0},
    {"class_id": 2, "class_name": "virginica",  "probability": 0.96},
    {"class_id": 0, "class_name": "setosa",     "probability": 1.0}
  ],
  "model_name":    "iris-classifier",
  "model_version": "1",
  "model_stage":   "Production"
}
```

## Features incluidas

- **MLFlow Tracking**: métricas, parámetros, artefactos por experiment run
- **Model Registry**: versionado, stages (Staging → Production → Archived)
- **Auto-promotion**: el trainer promueve automáticamente a Production
- **Hot-reload**: `POST /reload` recarga el modelo sin reiniciar el contenedor
- **Health checks**: Docker espera a que cada servicio esté sano antes de continuar
- **Volúmenes persistentes**: la DB y artefactos de MLFlow sobreviven reinicios
- **Data Drift Detection**: Detecta cambios en distribuciones de features (KS test) y predicciones (Chi2)
- **Performance Drift Detection**: Monitorea caídas en Accuracy, Precision, Recall, F1 usando t-tests y EWMA
- **Quality Gates**: Valida métricas antes de promoción a Production
- **Automated Testing**: Suite QA con tests de data, model, API y performance drift

## Monitoreo de Drifts

### Data Drift
Detecta cambios en la distribución de features usando Kolmogorov-Smirnov test y distribución de predicciones con Chi-square test.

### Performance Drift  
Monitorea métricas de clasificación:
- **Accuracy**: Exactitud general
- **Precision**: Precisión weighted, macro, micro
- **Recall**: Recall weighted, macro, micro  
- **F1-score**: F1 weighted, macro, micro

**Métodos estadísticos:**
- **t-test**: Compara media de ventana actual vs baseline (requiere >= 5 samples)
- **EWMA**: Exponential Moving Average detecta tendencias sostenidas
- **Effect Size**: Cambio absoluto > umbral (default 5%)

**Configuración (variables de entorno):**
```bash
ENABLE_PERFORMANCE_DRIFT=true           # Habilitar detección
PERF_DRIFT_EFFECT_SIZE=0.05             # Cambio mínimo (5%)
PERF_DRIFT_P_VALUE=0.05                 # Significancia estadística
PERF_DRIFT_CONSECUTIVE=2                # Ventanas consecutivas con drift
```

**Triggers:**
- Se logean métricas en MLFlow bajo experimento `performance-drift-monitoring`
- Si >= 2 métricas muestran drift en N ventanas consecutivas → trigger retraining automático
- Detención correcta de falsas alarmas con umbrales ajustables

## Actualizar el modelo

Para entrenar una nueva versión y promoverla:

```bash
# Re-entrenar (crea una nueva versión en el Registry)
docker compose run --rm model-trainer

# El endpoint /reload actualiza el modelo en producción sin downtime
curl -X POST http://localhost:8000/reload
```

## Detener el stack

```bash
docker compose down          # detiene contenedores
docker compose down -v       # detiene + elimina volúmenes (borra datos)
```

## Estructura del proyecto

```
mlops-stack/
├── docker-compose.yml
├── start.sh
├── README.md
├── mlflow/
│   └── Dockerfile
├── model-trainer/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── train.py                           ← entrenamiento + registro en MLFlow
├── inference-api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                             ← FastAPI + carga desde Model Registry
├── drift-detector/
│   ├── Dockerfile
│   ├── detector.py                        ← Data drift + Performance drift detection
│   ├── performance_drift_detector.py      ← PerformanceDriftDetector class
│   ├── retrain_trigger.py                 ← Trigger retraining on drift
│   └── requirements.txt
├── nginx/
│   ├── Dockerfile
│   └── nginx.conf
└── tests/
    ├── Dockerfile
    ├── run_tests.py                       ← Orquestador de pruebas (4 niveles)
    ├── test_data.py                       ← Level 1: Data validation tests
    ├── test_model.py                      ← Level 2: Model quality gates
    ├── test_api.py                        ← Level 3: API tests
    ├── test_performance_drift.py          ← Level 4: Performance drift tests
    ├── test_performance_drift_integration.py ← Integration tests
    └── entrypoint.sh
```
