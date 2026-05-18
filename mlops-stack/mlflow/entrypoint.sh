#!/bin/bash
set -e

# Crear directorios necesarios dentro del volumen montado
# Esto corre DESPUÉS de que Docker monta el volumen, así que los permisos persisten
mkdir -p /mlflow/artifacts
chmod -R 777 /mlflow

echo "MLFlow data dir ready: $(ls -la /mlflow)"

exec mlflow server \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5001
