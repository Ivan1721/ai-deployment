#!/bin/bash
# setup-env.sh — genera .env con secretos aleatorios para entorno local/dev
# Uso: bash setup-env.sh

set -euo pipefail

ENV_FILE="$(dirname "$0")/.env"

if [ -f "$ENV_FILE" ]; then
  echo "⚠  .env ya existe. ¿Sobreescribir? [y/N]"
  read -r answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo "Cancelado."; exit 0; }
fi

API_KEY=$(openssl rand -hex 24)
POSTGRES_PASSWORD=$(openssl rand -hex 20)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 16)
OAUTH2_COOKIE_SECRET=$(openssl rand -base64 32 | tr -d '\n')

cat > "$ENV_FILE" <<EOF
# Generado por setup-env.sh — NO commitear
# OAUTH2_CLIENT_SECRET se completa después con setup-keycloak.sh

API_KEY=${API_KEY}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
OAUTH2_COOKIE_SECRET=${OAUTH2_COOKIE_SECRET}
OAUTH2_CLIENT_SECRET=PENDING_KEYCLOAK_SETUP
EOF

echo "╔══════════════════════════════════════════════════════╗"
echo "║  .env generado ✓                                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  API_KEY:               ${API_KEY}"
echo "  POSTGRES_PASSWORD:     ${POSTGRES_PASSWORD}"
echo "  GRAFANA_ADMIN_PASSWORD:${GRAFANA_ADMIN_PASSWORD}"
echo ""
echo "  OAUTH2_CLIENT_SECRET:  PENDIENTE (ver pasos abajo)"
echo ""
echo "  Próximos pasos:"
echo "  1. Generar certs TLS:     bash generate-certs.sh localhost"
echo "  2. Crear DVC store:       sudo mkdir -p /opt/mlops-dvc-store && sudo chown \$USER:\$USER /opt/mlops-dvc-store"
echo "  3. Levantar stack:        docker compose down -v && bash start.sh"
echo "  4. Configurar Keycloak:   bash setup-keycloak.sh http://localhost:8080 admin admin123"
echo "     → copia el CLIENT_SECRET impreso y ejecuta:"
echo "     sed -i 's/PENDING_KEYCLOAK_SETUP/<secret>/' .env"
echo "     docker compose up -d oauth2-proxy"
echo ""
