#!/usr/bin/env bash
# setup_runner.sh
#
# REEMPLAZA setup_server.sh de la versión anterior.
# Instala el self-hosted GitHub Actions runner en la misma máquina
# donde corren los contenedores Docker del stack MLOps.
#
# Todo en una sola máquina JFCALDER:
#   ┌─────────────────────────────────────────┐
#   │            Host Linux                   │
#   │  ┌──────────────────────────────────┐   │
#   │  │  GitHub Actions Runner (proceso) │   │
#   │  │  escucha GitHub → ejecuta jobs   │   │
#   │  └──────────────────┬───────────────┘   │
#   │                     │ docker compose     │
#   │  ┌──────────────────▼───────────────┐   │
#   │  │  Docker (mlops-net)              │   │
#   │  │  mlflow · inference-api          │   │
#   │  │  drift-detector · nginx          │   │
#   │  └──────────────────────────────────┘   │
#   └─────────────────────────────────────────┘
#
# Uso:
#   export GITHUB_REPO="https://github.com/TU_ORG/TU_REPO"
#   export RUNNER_TOKEN="TOKEN_DEL_PASO_3"
#   bash setup_runner.sh

set -euo pipefail

GITHUB_REPO="${GITHUB_REPO:-}"
RUNNER_TOKEN="${RUNNER_TOKEN:-}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)-mlops}"
RUNNER_USER="${RUNNER_USER:-$(whoami)}"
DEPLOY_DIR="${DEPLOY_DIR:-/opt/mlops-stack}"
RUNNER_DIR="${RUNNER_DIR:-/opt/actions-runner}"
RUNNER_VERSION="2.317.0"

# ── Validación ─────────────────────────────────────────────────────────────
if [ -z "$GITHUB_REPO" ] || [ -z "$RUNNER_TOKEN" ]; then
  echo "ERROR: Definir GITHUB_REPO y RUNNER_TOKEN antes de ejecutar."
  echo ""
  echo "  Obtener el token en:"
  echo "  GitHub → Settings → Actions → Runners → New self-hosted runner"
  echo ""
  echo "  export GITHUB_REPO='https://github.com/ORG/REPO'"
  echo "  export RUNNER_TOKEN='AXXXXXXXXXXXXXXXXXX'"
  echo "  bash setup_runner.sh"
  exit 1
fi

echo "╔══════════════════════════════════════════════════════╗"
echo "║     MLOps Stack — Self-Hosted Runner Setup           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Repo:        $GITHUB_REPO"
echo "  Runner name: $RUNNER_NAME"
echo "  Runner user: $RUNNER_USER"
echo "  Runner dir:  $RUNNER_DIR"
echo "  Deploy dir:  $DEPLOY_DIR"
echo ""

# ── 1. Docker ──────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "▶ Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  usermod -aG docker "$RUNNER_USER"
  echo "  Docker installed ✓"
else
  echo "  Docker: $(docker --version) ✓"
fi

# ── 2. Docker Compose plugin ───────────────────────────────────────────────
if ! docker compose version &>/dev/null; then
  echo "▶ Installing Docker Compose..."
  COMPOSE_VERSION="v2.27.0"
  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  echo "  Docker Compose installed ✓"
else
  echo "  Docker Compose: $(docker compose version) ✓"
fi

# ── 3. Directorio de deploy ────────────────────────────────────────────────
echo "▶ Creating deploy directory: $DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"
chown "$RUNNER_USER:$RUNNER_USER" "$DEPLOY_DIR"
echo "  Deploy dir ready ✓"

# ── 4. Descargar GitHub Actions runner ────────────────────────────────────
echo "▶ Installing GitHub Actions runner v${RUNNER_VERSION}..."
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

ARCH=$(uname -m)
case $ARCH in
  x86_64)  ARCH_SUFFIX="x64" ;;
  aarch64) ARCH_SUFFIX="arm64" ;;
  *)        echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

RUNNER_PACKAGE="actions-runner-linux-${ARCH_SUFFIX}-${RUNNER_VERSION}.tar.gz"
if [ ! -f "$RUNNER_PACKAGE" ]; then
  curl -fsSL \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_PACKAGE}" \
    -o "$RUNNER_PACKAGE"
  tar xzf "$RUNNER_PACKAGE"
fi
chown -R "$RUNNER_USER:$RUNNER_USER" "$RUNNER_DIR"
echo "  Runner downloaded ✓"

# ── 5. Configurar el runner ────────────────────────────────────────────────
echo "▶ Configuring runner..."
sudo -u "$RUNNER_USER" "$RUNNER_DIR/config.sh" \
  --url "$GITHUB_REPO" \
  --token "$RUNNER_TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "self-hosted,mlops,docker" \
  --work "_work" \
  --unattended \
  --replace
echo "  Runner configured ✓"

# ── 6. Instalar como servicio systemd ─────────────────────────────────────
echo "▶ Installing runner as systemd service..."
cd "$RUNNER_DIR"
./svc.sh install "$RUNNER_USER"
./svc.sh start
echo "  Runner service started ✓"

# ── 7. Servicio systemd para el stack MLOps ───────────────────────────────
echo "▶ Creating mlops-stack systemd service..."
cat > /etc/systemd/system/mlops-stack.service << SVCEOF
[Unit]
Description=MLOps Stack (Docker Compose)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$DEPLOY_DIR
ExecStart=/usr/bin/docker compose -p mlops up -d mlflow inference-api drift-detector nginx
ExecStop=/usr/bin/docker compose -p mlops stop
TimeoutStartSec=300
User=$RUNNER_USER

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable mlops-stack.service
echo "  mlops-stack service enabled ✓"

# ── Resumen ────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete ✓                                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Runner activo en: $GITHUB_REPO"
echo "  Verificar en:     GitHub → Settings → Actions → Runners"
echo "                    (debe aparecer '$RUNNER_NAME' como 'Idle')"
echo ""
echo "  Próximos pasos:"
echo "  1. git push a main → el CD se ejecuta automáticamente"
echo "  2. Ver logs del runner: sudo journalctl -u actions.runner.* -f"
echo "  3. Stack logs: docker compose -p mlops logs -f"
echo ""
echo "  URLs (una vez desplegado):"
echo "    MLFlow UI    → http://localhost:5001"
echo "    API docs     → http://localhost:8000/docs"
echo "    Nginx proxy  → http://localhost"
