#!/usr/bin/env bash
set -e

MLFLOW_PORT=5001
API_PORT=8000
SKIP_TRAIN=0

for arg in "$@"; do
  case $arg in
    --skip-train|-s) SKIP_TRAIN=1 ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        MLOps Stack  ·  powered by MLFlow             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── build ──────────────────────────────────────────────────
echo "▶ Building images…"
docker compose build --parallel

# ── levantar MLFlow ────────────────────────────────────────
echo "▶ Starting MLFlow Tracking Server (puerto $MLFLOW_PORT)…"
docker compose up -d mlflow

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
if [ $SKIP_TRAIN -eq 1 ]; then
  echo "▶ Skipping training (--skip-train)"
else
  echo "▶ Training & registering models…"
  docker compose run --rm model-trainer
fi

# ── levantar inference API ─────────────────────────────────
echo "▶ Starting Inference API…"
docker compose up -d inference-api

echo "▶ Waiting for Inference API… (máx 180s)"
MAX=180; ELAPSED=0
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

# ── levantar observabilidad ────────────────────────────────
echo "▶ Starting Prometheus & Grafana…"
docker compose up -d prometheus grafana drift-detector

# ── levantar Keycloak y esperar ────────────────────────────
echo "▶ Starting Keycloak… (puede tardar ~60s)"
docker compose up -d keycloak

MAX=120; ELAPSED=0
until docker inspect keycloak --format='{{.State.Health.Status}}' 2>/dev/null | grep -q "healthy"; do
    if [ $ELAPSED -ge $MAX ]; then
        echo ""
        echo "✗ Keycloak timeout. Últimas líneas de log:"
        docker compose logs --tail=10 keycloak
        exit 1
    fi
    printf "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo " ✓  (${ELAPSED}s)"

# ── configurar Keycloak si el realm no existe ─────────────
REALM_STATUS=$(curl -so /dev/null -w "%{http_code}" http://localhost:8080/realms/mlops)
if [ "$REALM_STATUS" != "200" ]; then
  echo "▶ Keycloak realm 'mlops' not found — running setup…"
  NEW_SECRET=$(bash setup-keycloak.sh http://localhost:8080 admin admin123 2>&1 \
    | grep "Client secret:" | awk '{print $NF}')
  if [ -n "$NEW_SECRET" ] && [ "$NEW_SECRET" != "null" ]; then
    sed -i "s/OAUTH2_CLIENT_SECRET=.*/OAUTH2_CLIENT_SECRET=$NEW_SECRET/" .env
    echo "  OAUTH2_CLIENT_SECRET updated ✓"
  else
    echo "  ⚠ Could not extract client secret — update .env manually"
  fi
else
  echo "▶ Keycloak realm 'mlops' exists ✓"
fi

# ── levantar oauth2-proxy y nginx ─────────────────────────
echo "▶ Starting OAuth2 Proxy & Nginx…"
docker compose up -d oauth2-proxy nginx

echo ""
echo "════════════════════════════════════════════════════════"
echo " Stack is UP"
echo ""
echo "  MLflow UI     →  https://localhost/mlflow/  (login: mlops-user / mlops123)"
echo "  Grafana       →  https://localhost/grafana/ (login: admin / GRAFANA_ADMIN_PASSWORD)"
echo "  Inference API →  http://localhost:$API_PORT"
echo "  API docs      →  http://localhost:$API_PORT/docs"
echo "  Keycloak      →  http://localhost:8080"
echo ""
echo "  Quick predict:"
echo "  curl -X POST http://localhost:$API_PORT/predict \\"
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"scenario":0,"workers":6,"crop_row":2,"rand_pos":0,"activity":"harv_mixed"}'"'"
echo "════════════════════════════════════════════════════════"
