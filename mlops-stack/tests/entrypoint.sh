#!/bin/bash
set -e

echo "=== Test Runner Entrypoint ==="
echo "Python: $(python --version)"

# Detectar IP del gateway Docker
GATEWAY=$(ip route 2>/dev/null | awk '/default/ {print $3; exit}')
echo "Gateway: ${GATEWAY:-not found}"

# Sobreescribir URLs con la IP del gateway si contienen nombres DNS internos
if echo "${MLFLOW_TRACKING_URI:-}" | grep -qE "mlflow|inference-api|host\.docker\.internal|^$"; then
    export MLFLOW_TRACKING_URI="http://${GATEWAY}:5001"
fi
if echo "${INFERENCE_API_URL:-}" | grep -qE "mlflow|inference-api|host\.docker\.internal|^$"; then
    export INFERENCE_API_URL="http://${GATEWAY}:8000"
fi

echo "MLFLOW_TRACKING_URI = $MLFLOW_TRACKING_URI"
echo "INFERENCE_API_URL   = $INFERENCE_API_URL"
echo ""

# Probar conectividad antes de arrancar
echo "Testing connectivity..."
curl -sf "$MLFLOW_TRACKING_URI/" > /dev/null 2>&1 \
    && echo "MLFlow: reachable" \
    || echo "MLFlow: NOT reachable (will retry in Python)"

curl -sf "$INFERENCE_API_URL/health" > /dev/null 2>&1 \
    && echo "InferenceAPI: reachable" \
    || echo "InferenceAPI: NOT reachable (API tests will be skipped)"

echo ""
exec python run_tests.py
