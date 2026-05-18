#!/bin/bash
set -e

echo "=== Test Runner Entrypoint ==="

# Detectar IP del gateway (host Docker) en Linux
GATEWAY=$(ip route | awk '/default/ {print $3; exit}')
echo "Docker gateway IP: $GATEWAY"

# Si las URLs aún apuntan a nombres DNS que no resuelven, reemplazar con IP
if [ -z "$MLFLOW_TRACKING_URI" ] || echo "$MLFLOW_TRACKING_URI" | grep -q "mlflow\|inference-api\|host.docker.internal"; then
    export MLFLOW_TRACKING_URI="http://${GATEWAY}:5001"
    echo "MLFLOW_TRACKING_URI overridden to: $MLFLOW_TRACKING_URI"
fi

if [ -z "$INFERENCE_API_URL" ] || echo "$INFERENCE_API_URL" | grep -q "mlflow\|inference-api\|host.docker.internal"; then
    export INFERENCE_API_URL="http://${GATEWAY}:8000"
    echo "INFERENCE_API_URL overridden to: $INFERENCE_API_URL"
fi

echo "Final MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"
echo "Final INFERENCE_API_URL:   $INFERENCE_API_URL"
echo ""

exec python run_tests.py
