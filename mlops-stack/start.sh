#!/usr/bin/env bash
set -e

MLFLOW_PORT=5001
API_PORT=8000

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        MLOps Stack  ·  powered by MLFlow             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── verificar puerto libre ──────────────────────────────────
check_port() {
  local port=$1
  if lsof -i TCP:$port -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✗  El puerto $port está ocupado. Libéralo antes de continuar."
    echo "   (en macOS el puerto 5000 lo usa AirPlay — usa otro puerto)"
    exit 1
  fi
}
check_port $MLFLOW_PORT
check_port $API_PORT

# ── build ──────────────────────────────────────────────────
echo "▶ Building images…"
docker compose build --parallel

# ── levantar MLFlow ────────────────────────────────────────
echo "▶ Starting MLFlow Tracking Server (puerto $MLFLOW_PORT)…"
docker compose up -d mlflow

# ── esperar con logs visibles ──────────────────────────────
echo "▶ Waiting for MLFlow… (máx 120s)"
MAX=120; ELAPSED=0
until curl -sf http://localhost:$MLFLOW_PORT/ > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX ]; then
        echo ""
        echo "✗ Timeout. Últimas líneas de log:"
        docker compose logs --tail=20 mlflow
        exit 1
    fi
    printf "."
    sleep 4
    ELAPSED=$((ELAPSED + 4))
done
echo " ✓  (${ELAPSED}s)"

# ── entrenar ───────────────────────────────────────────────
echo "▶ Training & registering model…"
docker compose run --rm model-trainer

# ── levantar resto ─────────────────────────────────────────
echo "▶ Starting Inference API & Nginx…"
docker compose up -d inference-api nginx

echo "▶ Waiting for Inference API… (máx 90s)"
MAX=90; ELAPSED=0
until curl -sf http://localhost:$API_PORT/health > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX ]; then
        echo ""
        echo "✗ Timeout. Últimas líneas de log:"
        docker compose logs --tail=20 inference-api
        exit 1
    fi
    printf "."
    sleep 4
    ELAPSED=$((ELAPSED + 4))
done
echo " ✓"

echo ""
echo "════════════════════════════════════════════════════════"
echo " Stack is UP 🚀"
echo ""
echo "  MLFlow UI       →  http://localhost:$MLFLOW_PORT"
echo "  Inference API   →  http://localhost:$API_PORT"
echo "  API docs        →  http://localhost:$API_PORT/docs"
echo ""
echo "  Quick predict test (scenario 0 = HumanOnly, 6 workers, row 2, mixed activity):"
echo "  curl -X POST http://localhost:$API_PORT/predict \\"
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"scenario":0,"workers":6,"crop_row":2,"rand_pos":0,"activity":"harv_mixed"}'"'"
echo "════════════════════════════════════════════════════════"
