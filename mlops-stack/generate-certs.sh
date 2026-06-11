#!/bin/bash
# generate-certs.sh
#
# SECURITY FIX P0-3: Genera certificados autofirmados para HTTPS
# Usar en desarrollo. En producción, usar Let's Encrypt (certbot)
#
# Uso:
#   bash generate-certs.sh [hostname]
#
# Ejemplo:
#   bash generate-certs.sh mlops.example.com

set -euo pipefail

HOSTNAME="${1:-localhost}"
CERT_DIR="./nginx/certs"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Generating self-signed certificates for HTTPS      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Hostname: $HOSTNAME"
echo "  Output:   $CERT_DIR/"
echo ""

# Crear directorio de certificados
mkdir -p "$CERT_DIR"

# Generar clave privada (4096-bit RSA)
echo "▶ Generating private key..."
openssl genrsa -out "$CERT_DIR/mlops-key.pem" 4096

# Generar certificate signing request (CSR)
echo "▶ Generating certificate signing request..."
openssl req -new -key "$CERT_DIR/mlops-key.pem" \
  -out "$CERT_DIR/mlops.csr" \
  -subj "/C=US/ST=State/L=City/O=MLOps/CN=$HOSTNAME/subjectAltName=DNS:$HOSTNAME,DNS:localhost,IP:127.0.0.1"

# Generar certificado autofirmado (válido 365 días)
echo "▶ Generating self-signed certificate..."
openssl x509 -req -days 365 \
  -in "$CERT_DIR/mlops.csr" \
  -signkey "$CERT_DIR/mlops-key.pem" \
  -out "$CERT_DIR/mlops-cert.pem" \
  -extfile <(printf "subjectAltName=DNS:$HOSTNAME,DNS:localhost,IP:127.0.0.1")

# Establecer permisos correctos
chmod 600 "$CERT_DIR/mlops-key.pem"
chmod 644 "$CERT_DIR/mlops-cert.pem"

# Mostrar información del certificado
echo ""
echo "✓ Certificates generated:"
echo "  Key:  $CERT_DIR/mlops-key.pem"
echo "  Cert: $CERT_DIR/mlops-cert.pem"
echo ""

# Mostrar información del certificado
echo "Certificate details:"
openssl x509 -in "$CERT_DIR/mlops-cert.pem" -text -noout | grep -E "Not Before|Not After|Subject:"

echo ""
echo "⚠  For production, use Let's Encrypt instead:"
echo "   https://letsencrypt.org/"
echo ""
