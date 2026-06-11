#!/bin/bash
set -e

# Crear directorios necesarios dentro del volumen montado
# Esto corre DESPUÉS de que Docker monta el volumen, así que los permisos persisten
mkdir -p /mlflow/artifacts

# SECURITY FIX P0-5: Restringir permisos
# - Base de datos: 700 (solo dueño puede leer/escribir)
# - Artifacts: 755 (dueño lee/escribe, otros solo leen)
chmod 700 /mlflow/mlflow.db 2>/dev/null || true
chmod 755 /mlflow/artifacts
chown -R mlflow:mlflow /mlflow 2>/dev/null || true

echo "MLFlow data dir ready: $(ls -la /mlflow)"

exec mlflow server \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5001
