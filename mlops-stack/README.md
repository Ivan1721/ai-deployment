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
│   └── train.py            ← entrenamiento + registro en MLFlow
├── inference-api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py              ← FastAPI + carga desde Model Registry
└── nginx/
    ├── Dockerfile
    └── nginx.conf
```
