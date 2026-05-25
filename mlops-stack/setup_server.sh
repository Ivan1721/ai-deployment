#!/usr/bin/env bash
# setup_server.sh
#
# NUEVO ARCHIVO — no existía en la versión anterior.
#
# Prepara un servidor Ubuntu/Debian para recibir deploys del pipeline CI/CD.
# Ejecutar UNA SOLA VEZ en el servidor de producción como root o con sudo.
#
# Uso:
#   curl -fsSL https://raw.githubusercontent.com/TU_ORG/TU_REPO/main/setup_server.sh | bash
#   # O localmente:
#   bash setup_server.sh

set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-/opt/mlops-stack}"
DEPLOY_USER="${DEPLOY_USER:-$(whoami)}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║     MLOps Stack — Server Setup                       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Deploy path : $DEPLOY_PATH"
echo "  Deploy user : $DEPLOY_USER"
echo ""

# ── 1. Instalar Docker ─────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "▶ Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  usermod -aG docker "$DEPLOY_USER"
  echo "  Docker installed ✓"
else
  echo "  Docker already installed: $(docker --version)"
fi

# ── 2. Instalar Docker Compose plugin ─────────────────────────────────────
if ! docker compose version &>/dev/null; then
  echo "▶ Installing Docker Compose plugin..."
  COMPOSE_VERSION="v2.27.0"
  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  echo "  Docker Compose installed ✓"
else
  echo "  Docker Compose already installed: $(docker compose version)"
fi

# ── 3. Crear directorio de deploy ──────────────────────────────────────────
echo "▶ Creating deploy directory: $DEPLOY_PATH"
mkdir -p "$DEPLOY_PATH"
chown "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_PATH"
echo "  Directory created ✓"

# ── 4. Crear usuario de deploy con acceso limitado ─────────────────────────
if [ "$DEPLOY_USER" != "$(whoami)" ]; then
  if ! id "$DEPLOY_USER" &>/dev/null; then
    echo "▶ Creating deploy user: $DEPLOY_USER"
    useradd -m -s /bin/bash "$DEPLOY_USER"
    usermod -aG docker "$DEPLOY_USER"
    echo "  User created ✓"
  fi
fi

# ── 5. Configurar SSH para GitHub Actions ──────────────────────────────────
echo "▶ Configuring SSH..."
SSH_DIR="/home/$DEPLOY_USER/.ssh"
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"
touch "$SSH_DIR/authorized_keys"
chmod 600 "$SSH_DIR/authorized_keys"
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$SSH_DIR"

echo ""
echo "════════════════════════════════════════════════════════"
echo " PRÓXIMO PASO: Agregar la clave pública de GitHub Actions"
echo " Ejecutar en tu máquina local:"
echo ""
echo "   ssh-keygen -t ed25519 -C 'github-actions-deploy' -f ~/.ssh/mlops_deploy"
echo "   cat ~/.ssh/mlops_deploy.pub"
echo ""
echo " Luego agregar el output a: $SSH_DIR/authorized_keys"
echo " Y agregar el contenido de ~/.ssh/mlops_deploy como Secret:"
echo "   GitHub Repo → Settings → Secrets → PROD_SSH_KEY"
echo "════════════════════════════════════════════════════════"

# ── 6. Crear archivo .env para producción ─────────────────────────────────
ENV_FILE="$DEPLOY_PATH/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "▶ Creating .env template..."
  cat > "$ENV_FILE" << ENVEOF
# Producción — completar con valores reales
REGISTRY_IMAGE=ghcr.io/TU_ORG
IMAGE_TAG=latest
ENVEOF
  chown "$DEPLOY_USER:$DEPLOY_USER" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "  .env created at $ENV_FILE ✓"
  echo "  ⚠  Editar $ENV_FILE con los valores correctos antes del primer deploy"
fi

# ── 7. Systemd service para auto-restart ───────────────────────────────────
echo "▶ Creating systemd service..."
cat > /etc/systemd/system/mlops-stack.service << SVCEOF
[Unit]
Description=MLOps Stack
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$DEPLOY_PATH
EnvironmentFile=$DEPLOY_PATH/.env
ExecStart=/usr/bin/docker compose up -d mlflow inference-api drift-detector nginx
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300
User=$DEPLOY_USER

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable mlops-stack.service
echo "  systemd service enabled ✓ (auto-start on boot)"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete ✓                                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Secrets a configurar en GitHub:"
echo "    PROD_HOST        = $(hostname -I | awk '{print $1}')"
echo "    PROD_USER        = $DEPLOY_USER"
echo "    PROD_SSH_KEY     = (contenido de ~/.ssh/mlops_deploy)"
echo "    PROD_DEPLOY_PATH = $DEPLOY_PATH"
