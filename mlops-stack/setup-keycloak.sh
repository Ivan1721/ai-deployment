#!/bin/bash
# setup-keycloak.sh
#
# SECURITY FIX P0-1: Configura Keycloak automáticamente después del despliegue
# Este script crea el realm "mlops" y el cliente OIDC
#
# Uso:
#   bash setup-keycloak.sh [keycloak_url] [admin_user] [admin_password]
#
# Ejemplo:
#   bash setup-keycloak.sh http://localhost:8080 admin admin123

set -euo pipefail

KEYCLOAK_URL="${1:-http://localhost:8080}"
ADMIN_USER="${2:-admin}"
ADMIN_PASSWORD="${3:-admin123}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Keycloak Setup — Configure Realm & Client          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  URL:      $KEYCLOAK_URL"
echo "  Admin:    $ADMIN_USER"
echo ""

# Esperar a que Keycloak esté disponible
echo "▶ Waiting for Keycloak to be ready..."
for i in {1..60}; do
  if curl -sf "$KEYCLOAK_URL" >/dev/null 2>&1; then
    echo "  Keycloak is ready ✓"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "  ✗ Keycloak did not start in time"
    exit 1
  fi
  echo "  Attempt $i/60..."
  sleep 1
done

# Obtener token de admin
echo "▶ Getting admin token..."
ADMIN_TOKEN=$(curl -s -X POST \
  "$KEYCLOAK_URL/realms/master/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=admin-cli" \
  -d "username=$ADMIN_USER" \
  -d "password=$ADMIN_PASSWORD" \
  -d "grant_type=password" | jq -r '.access_token')

if [ -z "$ADMIN_TOKEN" ] || [ "$ADMIN_TOKEN" == "null" ]; then
  echo "  ✗ Failed to get admin token"
  exit 1
fi
echo "  Admin token obtained ✓"

# Crear realm "mlops"
echo "▶ Creating realm: mlops..."
curl -s -X POST "$KEYCLOAK_URL/admin/realms" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "realm": "mlops",
    "enabled": true,
    "displayName": "MLOps Platform"
  }' > /dev/null || true
echo "  Realm created ✓"

# Crear cliente OIDC "mlops-client"
echo "▶ Creating OIDC client: mlops-client..."
CLIENT_ID=$(curl -s -X POST \
  "$KEYCLOAK_URL/admin/realms/mlops/clients" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "clientId": "mlops-client",
    "enabled": true,
    "publicClient": false,
    "redirectUris": [
      "http://localhost/oauth2/callback",
      "https://localhost/oauth2/callback"
    ],
    "webOrigins": [
      "http://localhost",
      "https://localhost"
    ],
    "standardFlowEnabled": true,
    "implicitFlowEnabled": false,
    "directAccessGrantsEnabled": false
  }' | jq -r '.id')

echo "  Client ID: $CLIENT_ID ✓"

# Obtener client secret
echo "▶ Getting client secret..."
CLIENT_SECRET=$(curl -s -X GET \
  "$KEYCLOAK_URL/admin/realms/mlops/clients/$CLIENT_ID/client-secret" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq -r '.value')
echo "  Client secret: $CLIENT_SECRET"

# Crear usuario de prueba
echo "▶ Creating test user: mlops-user..."
curl -s -X POST \
  "$KEYCLOAK_URL/admin/realms/mlops/users" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "mlops-user",
    "firstName": "MLOps",
    "lastName": "User",
    "email": "mlops@example.com",
    "enabled": true
  }' > /dev/null || true

# Obtener user ID y establecer password
USER_ID=$(curl -s -X GET \
  "$KEYCLOAK_URL/admin/realms/mlops/users?username=mlops-user" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq -r '.[0].id')

curl -s -X PUT \
  "$KEYCLOAK_URL/admin/realms/mlops/users/$USER_ID/reset-password" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "password",
    "value": "mlops123",
    "temporary": false
  }' > /dev/null

echo "  Test user created ✓"
echo "  Username: mlops-user"
echo "  Password: mlops123"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Keycloak Setup Complete ✓                          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Next steps:"
echo "  1. Update docker-compose.yml with client secret:"
echo "     OAUTH2_PROXY_CLIENT_SECRET=$CLIENT_SECRET"
echo ""
echo "  2. Restart docker-compose:"
echo "     docker compose up -d"
echo ""
echo "  3. Access the platform:"
echo "     https://localhost/mlflow/"
echo ""
echo "  4. Login with:"
echo "     Username: mlops-user"
echo "     Password: mlops123"
echo ""
