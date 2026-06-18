#!/bin/bash
set -e

mkdir -p /mlflow/artifacts
chmod 755 /mlflow/artifacts

if [ -n "$POSTGRES_URI" ]; then
    BACKEND_URI="$POSTGRES_URI"
else
    # Fallback for local dev without postgres
    chmod 700 /mlflow/mlflow.db 2>/dev/null || true
    BACKEND_URI="sqlite:////mlflow/mlflow.db"
fi

echo "MLFlow backend: ${BACKEND_URI%%@*}@..."
echo "MLFlow data dir: $(ls -la /mlflow)"

exec mlflow server \
    --backend-store-uri "$BACKEND_URI" \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5001
