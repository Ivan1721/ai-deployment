# Guía de Re-despliegue y Validación de Cambios P0

**Fecha:** 2026-06-11  
**Versión:** 1.0  
**Audiencia:** Ingenieros de DevOps, SRE

---

## 📋 Tabla de Contenidos
1. [Pre-despliegue](#pre-despliegue)
2. [Despliegue del Sistema](#despliegue-del-sistema)
3. [Validación de Cambios P0](#validación-de-cambios-p0)
4. [Troubleshooting](#troubleshooting)
5. [Rollback (en caso de emergencia)](#rollback)
6. [Post-despliegue (Producción)](#post-despliegue-producción)

---

## Pre-despliegue

### Paso 1: Verificar requisitos

```bash
# Ubicarse en directorio del proyecto
cd mlops-stack

# Verificar Docker
docker --version
# Esperado: Docker version 24.x.x o superior

# Verificar Docker Compose
docker compose version
# Esperado: Docker Compose version 2.27.0 o superior

# Verificar git
git status
# Esperado: En rama develop o feature branch
```

### Paso 2: Validar archivos modificados

```bash
# Verificar que los archivos clave existen
test -f docker-compose.yml && echo "✓ docker-compose.yml"
test -f inference-api/app.py && echo "✓ inference-api/app.py"
test -f nginx/nginx.conf && echo "✓ nginx/nginx.conf"
test -f mlflow/entrypoint.sh && echo "✓ mlflow/entrypoint.sh"
test -f generate-certs.sh && echo "✓ generate-certs.sh"
test -f setup-keycloak.sh && echo "✓ setup-keycloak.sh"
```

### Paso 3: Validar sintaxis de configuración

```bash
# Validar docker-compose.yml
docker-compose config > /dev/null && echo "✓ docker-compose.yml válido" || echo "✗ Error en docker-compose.yml"

# Validar nginx.conf (requiere imagen nginx)
docker run --rm -v $(pwd)/nginx:/etc/nginx:ro nginx:latest nginx -t
# Esperado: nginx: configuration file /etc/nginx/nginx.conf test is successful

# Validar Python app.py (sintaxis)
python3 -m py_compile inference-api/app.py && echo "✓ app.py válido" || echo "✗ Error en app.py"
```

### Paso 4: Limpiar despliegue anterior (si existe)

```bash
# Detener y remover contenedores antiguos
echo "▶ Stopping old containers..."
docker compose down

# Verificar que no hay contenedores corriendo
docker ps | grep -E "mlflow|inference|drift|test|nginx|keycloak|oauth2" || echo "✓ No hay contenedores previos"

# Limpiar volúmenes de desarrollo (SOLO en development, no en producción)
# ⚠️ ADVERTENCIA: Esto borrará modelos y datos entrenados
# read -p "¿Descartar volúmenes de desarrollo? (s/n) " -n 1 -r
# if [[ $REPLY =~ ^[Ss]$ ]]; then
#   docker volume rm mlops-stack_mlflow-data 2>/dev/null || true
# fi
```

### Paso 5: Generar certificados HTTPS

```bash
# Hacer script ejecutable
chmod +x generate-certs.sh

# Generar certificados autofirmados
echo "▶ Generating HTTPS certificates..."
bash generate-certs.sh localhost

# Verificar que los certificados se crearon
ls -la nginx/certs/
# Esperado:
#   -rw------- mlops-key.pem   (clave privada, 600)
#   -rw-r--r-- mlops-cert.pem  (certificado)
```

---

## Despliegue del Sistema

### Paso 1: Construir imágenes Docker

```bash
echo "▶ Building Docker images..."
docker compose build

# Esto puede tomar 3-5 minutos en la primera ejecución
# Verificar que no hay errores
# Esperado: Successfully tagged mlops-stack_mlflow:latest, etc.
```

### Paso 2: Iniciar servicios core (MLFlow, API, Nginx)

```bash
echo "▶ Starting core services..."
docker compose up -d mlflow inference-api nginx

# Esperar a que los servicios estén healthy
echo "▶ Waiting for services to be healthy..."
sleep 10

# Verificar status
docker compose ps
# Esperado: mlflow, inference-api, nginx con estado "Up"
```

### Paso 3: Iniciar servicios de autenticación

```bash
echo "▶ Starting authentication services..."
docker compose up -d keycloak oauth2-proxy

# Esperar a que Keycloak esté ready (puede tomar 30-60 segundos)
echo "▶ Waiting for Keycloak..."
for i in {1..60}; do
  if docker compose logs keycloak | grep -q "Listening on"; then
    echo "✓ Keycloak is ready"
    break
  fi
  echo "  Attempt $i/60..."
  sleep 1
done
```

### Paso 4: Iniciar servicios adicionales

```bash
echo "▶ Starting additional services..."
docker compose up -d drift-detector

# Verificar todos los servicios
docker compose ps
# Esperado: Todos los servicios con estado "Up"
```

### Paso 5: Configurar Keycloak

```bash
# Hacer script ejecutable
chmod +x setup-keycloak.sh

# Ejecutar configuración automática
echo "▶ Configuring Keycloak..."
bash setup-keycloak.sh http://localhost:8080 admin admin123

# Output esperado:
# ✓ Keycloak is ready
# ✓ Admin token obtained
# ✓ Realm created
# ✓ Client ID: ...
# ✓ Client secret: ...
# ✓ Test user created
#   Username: mlops-user
#   Password: mlops123
```

**IMPORTANTE:** Copiar el `Client secret` para el siguiente paso.

### Paso 6: Actualizar OAuth2 Client Secret (si es diferente)

```bash
# Si el client secret obtenido es diferente del hardcoded:
# 1. Editar docker-compose.yml
# 2. Buscar: OAUTH2_PROXY_CLIENT_SECRET: mlops-client-secret
# 3. Reemplazar con el secret obtenido

# O usar variable de entorno:
export OAUTH2_PROXY_CLIENT_SECRET="<secret-obtenido>"
docker compose up -d oauth2-proxy

echo "✓ OAuth2 proxy updated"
```

---

## Validación de Cambios P0

### Validación P0-1: Autenticación

#### Test 1.1: Verificar Keycloak accesible

```bash
# Desde el host (no desde contenedores internos)
curl -k http://localhost:8080 2>&1 | head -20
# Esperado: 200 OK, HTML de login de Keycloak (localhost-only, no desde internet)

# Verificar que NO es accesible desde otra IP
# (En máquina diferente) curl -k http://<MACHINE_IP>:8080
# Esperado: Timeout o Connection refused
```

#### Test 1.2: Verificar OAuth2 Proxy

```bash
# Verificar health check
curl -s http://localhost:4180/ping
# Esperado: "pong"

# Verificar que redirige a login
curl -k -L https://localhost/mlflow/ 2>&1 | grep -i "keycloak\|login"
# Esperado: Redirección a Keycloak login
```

#### Test 1.3: Verificar flujo de autenticación

```bash
# Obtener cookie de sesión
COOKIE=$(curl -s -k -c /tmp/cookies.txt -d "username=mlops-user&password=mlops123" \
  http://localhost:8080/realms/mlops/protocol/openid-connect/token \
  -H "Content-Type: application/x-www-form-urlencoded" | jq -r '.access_token')

# Intentar acceder a /mlflow con el token
curl -k -H "Authorization: Bearer $COOKIE" https://localhost/mlflow/ 2>&1 | head -20
# Esperado: 200 OK, puede haber error de credential pero NO 401/403
```

---

### Validación P0-2: CORS Restricción

#### Test 2.1: Verificar CORS bloqueado para origen no permitido

```bash
# Intentar desde origen no permitido
curl -k -i \
  -H "Origin: https://malicious.com" \
  -H "Access-Control-Request-Method: POST" \
  https://localhost/api/health 2>&1 | grep -i "access-control"
# Esperado: NO debe tener header "Access-Control-Allow-Origin" para origen no permitido
# O: Debe retornar 403 Forbidden
```

#### Test 2.2: Verificar CORS permitido para origen válido

```bash
# Intentar desde localhost (permitido)
curl -k -i \
  -H "Origin: http://localhost" \
  -H "Access-Control-Request-Method: POST" \
  https://localhost/api/health 2>&1 | grep -i "access-control"
# Esperado: "Access-Control-Allow-Origin: http://localhost" (u origen permitido)
```

#### Test 2.3: Verificar métodos limitados

```bash
# Intentar DELETE (no permitido, solo GET/POST)
curl -k -X DELETE https://localhost/api/health 2>&1
# Esperado: 405 Method Not Allowed (CORS no es lo importante, método no permitido)
```

---

### Validación P0-3: HTTPS

#### Test 3.1: Verificar certificado

```bash
# Ver información del certificado
openssl s_client -connect localhost:443 -showcerts 2>/dev/null | \
  openssl x509 -noout -dates -subject
# Esperado:
#   notBefore=... (fecha pasada)
#   notAfter=... (fecha futura, ~365 días desde hoy)
#   subject=... CN=localhost
```

#### Test 3.2: Verificar HTTP → HTTPS redirect

```bash
# HTTP debe redirigir a HTTPS
curl -i http://localhost/nginx-health 2>&1 | head -5
# Esperado: 301 Moved Permanently, Location: https://...

# HTTPS debe ser accesible
curl -k https://localhost/nginx-health
# Esperado: ok
```

#### Test 3.3: Verificar TLS versión

```bash
# Verificar que TLS 1.2+ se usa
openssl s_client -connect localhost:443 -tls1_2 2>/dev/null | grep -i "protocol"
# Esperado: "Protocol: TLSv1.2" o "TLSv1.3"

# Verificar que SSLv3 NO funciona (inseguro)
openssl s_client -connect localhost:443 -ssl3 2>&1 | grep -i "unknown protocol\|error"
# Esperado: Error (no debe conectar con SSL 3.0)
```

---

### Validación P0-4: Puertos Localhost-Only

#### Test 4.1: Verificar que puertos internos NO son accesibles públicamente

```bash
# MLFlow en 5001 (debe estar localhost-only)
curl -i http://localhost:5001 2>&1 | head -5
# Esperado: 200 OK (desde localhost funciona)

# Desde otra máquina (reemplazar MACHINE_IP):
# curl -i http://<MACHINE_IP>:5001
# Esperado: Timeout o Connection refused

# Métodos alternativos de verificación:
netstat -tlnp | grep -E "5001|8000|8080|4180"
# Esperado: 127.0.0.1:5001, 127.0.0.1:8000, etc. (no 0.0.0.0)
```

#### Test 4.2: Verificar que nginx es punto de entrada público

```bash
# Puertos 80 y 443 deben ser accesibles desde cualquier IP
netstat -tlnp | grep -E ":80|:443"
# Esperado: 0.0.0.0:80, 0.0.0.0:443 (accesibles públicamente)
```

#### Test 4.3: Verificar que acceso directo está bloqueado

```bash
# Intentar acceder directamente a MLFlow (localhost funciona):
curl -i http://localhost:5001 2>&1 | head -5
# Esperado: 200 OK (directo desde localhost)

# Pero tiene que pasar por nginx para estar autenticado:
curl -k https://localhost/mlflow/ 2>&1 | grep -i "keycloak\|login"
# Esperado: Requiere autenticación
```

---

### Validación P0-5: Permisos de Volumen

#### Test 5.1: Verificar permisos de mlflow.db

```bash
# Desde dentro del contenedor mlflow
docker compose exec mlflow ls -la /mlflow/mlflow.db
# Esperado: -rwx------ (700) mlflow:mlflow mlflow.db
# NO: -rwxrwxrwx (777)

# Verificar en volumen Docker (si accesible)
docker volume inspect mlflow-data | jq '.Mountpoint'
# Luego verificar:
ls -la <mountpoint>/mlflow.db
# Esperado: -rwx------ mlflow mlflow
```

#### Test 5.2: Verificar permisos de artifacts

```bash
docker compose exec mlflow ls -la /mlflow/
# Esperado: drwxr-xr-x (755) mlflow:mlflow artifacts/
# NO: drwxrwxrwx (777) artifacts/
```

#### Test 5.3: Verificar que modelos son legibles pero no writeable

```bash
# Encontrar un modelo entrenado
docker compose exec mlflow find /mlflow/artifacts -name "model.pkl" -type f 2>/dev/null | head -1
# Ejemplo output: /mlflow/artifacts/0/run-id/model/model.pkl

# Verificar permisos
docker compose exec mlflow ls -la /mlflow/artifacts/0/run-id/model/
# Esperado: -rw-r--r-- (644 o similar) mlflow:mlflow model.pkl
# No: -rwxrwxrwx (777)
```

---

## Verificación Integrada

### Script de Validación Completa

```bash
#!/bin/bash
# validate-p0.sh

echo "╔════════════════════════════════════════════════════════╗"
echo "║  Validating P0 Security Changes                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

FAILED=0

# P0-1: Autenticación
echo "▶ P0-1: Autenticación"
if curl -s -k http://localhost:8080 >/dev/null 2>&1; then
  echo "  ✓ Keycloak accessible"
else
  echo "  ✗ Keycloak NOT accessible"
  ((FAILED++))
fi

# P0-2: CORS
echo "▶ P0-2: CORS Restriction"
CORS_HEADER=$(curl -s -k -H "Origin: https://test.com" https://localhost/api/health | grep -i access-control-allow || echo "")
if [ -z "$CORS_HEADER" ]; then
  echo "  ✓ CORS restricted (no header for unauthorized origin)"
else
  echo "  ✗ CORS not restricted"
  ((FAILED++))
fi

# P0-3: HTTPS
echo "▶ P0-3: HTTPS"
if curl -k -I https://localhost/nginx-health 2>/dev/null | grep -q "200\|ok"; then
  echo "  ✓ HTTPS working"
else
  echo "  ✗ HTTPS NOT working"
  ((FAILED++))
fi

# P0-4: Localhost-only
echo "▶ P0-4: Localhost-only ports"
NETSTAT=$(netstat -tlnp 2>/dev/null | grep -E ":5001|:8000" | grep "127.0.0.1" | wc -l)
if [ "$NETSTAT" -ge 2 ]; then
  echo "  ✓ Internal ports are localhost-only"
else
  echo "  ✗ Some ports may be public"
  ((FAILED++))
fi

# P0-5: Permisos
echo "▶ P0-5: Volume Permissions"
PERMS=$(docker compose exec mlflow ls -la /mlflow/mlflow.db 2>/dev/null | awk '{print $1}' | grep -E "^-rwx------")
if [ -n "$PERMS" ]; then
  echo "  ✓ Database permissions are restricted (700)"
else
  echo "  ✗ Database permissions may be too open"
  ((FAILED++))
fi

echo ""
echo "════════════════════════════════════════════════════════"
if [ $FAILED -eq 0 ]; then
  echo "✓ All P0 validations PASSED"
  exit 0
else
  echo "✗ $FAILED validations FAILED"
  exit 1
fi
```

**Ejecutar:**
```bash
chmod +x validate-p0.sh
./validate-p0.sh
```

---

## Troubleshooting

### Problema: Keycloak no inicia

**Síntomas:** `docker compose ps` muestra keycloak con estado "Restarting"

**Solución:**
```bash
# Ver logs
docker compose logs keycloak

# Esperar más tiempo (primera ejecución puede tardar 60s)
sleep 60
docker compose ps

# Si sigue fallando, verifi car que la imagen existe
docker image ls | grep keycloak

# Si no existe, reconstruir
docker compose pull keycloak
docker compose up -d keycloak
```

---

### Problema: OAuth2-proxy no conecta a Keycloak

**Síntomas:** `docker compose logs oauth2-proxy` muestra errores de conexión

**Solución:**
```bash
# Verificar que Keycloak está healthy
docker compose exec keycloak curl -s http://localhost:8080 | head -10

# Verificar conectividad entre contenedores (Docker network)
docker compose exec oauth2-proxy ping -c 1 keycloak

# Si no resuelve, problema de network:
docker network ls | grep mlops

# Reconstruir network:
docker compose down
docker compose up -d
```

---

### Problema: Certificados no validan

**Síntomas:** Browser muestra "SSL_ERROR_BAD_CERT_DOMAIN"

**Solución:**
```bash
# Regenerar certificados con el hostname correcto
bash generate-certs.sh <tu-hostname-real>

# Reiniciar nginx
docker compose restart nginx

# Probar:
curl -k https://localhost/nginx-health
```

---

### Problema: CORS sigue devolviendo "*"

**Síntomas:** Responses aún tienen `Access-Control-Allow-Origin: *`

**Solución:**
```bash
# Verificar que app.py fue actualizado correctamente
grep "allow_origins=CORS_ORIGINS" inference-api/app.py

# Si no está, editar manualmente

# Reconstruir imagen
docker compose build inference-api

# Reiniciar
docker compose down
docker compose up -d inference-api
```

---

### Problema: Acceso bloqueado a puertos internos

**Síntomas:** No puedo conectar a `localhost:5001`

**Solución (desarrollo):**
```bash
# Temporal: cambiar docker-compose.yml para desarrollo
# Reemplazar:
#   - "127.0.0.1:5001:5001"
# Con:
#   - "5001:5001"

# O: Usar SSH tunneling
ssh -L 5001:localhost:5001 user@remote-host
curl http://localhost:5001
```

---

## Rollback

### En caso de emergencia (problemas críticos)

```bash
# Paso 1: Detener todos los servicios
docker compose down

# Paso 2: Restaurar configuración anterior (si está en git)
git checkout HEAD -- docker-compose.yml nginx/nginx.conf inference-api/app.py mlflow/entrypoint.sh

# Paso 3: Limpiar volúmenes de Keycloak (si causa problemas)
docker volume rm mlops-stack_keycloak-data 2>/dev/null || true

# Paso 4: Reiniciar con configuración anterior
docker compose build
docker compose up -d

# Paso 5: Verificar
docker compose ps
curl -i http://localhost/mlflow/
```

---

## Post-despliegue (Producción)

### Paso 1: Cambiar credenciales por defecto

```bash
# KEYCLOAK_ADMIN_PASSWORD en docker-compose.yml
# Cambiar de: admin123
# A: Generar contraseña fuerte
openssl rand -base64 32

# Editar docker-compose.yml
vi docker-compose.yml
# Buscar: KEYCLOAK_ADMIN_PASSWORD: admin123
# Cambiar a: KEYCLOAK_ADMIN_PASSWORD: <contraseña-fuerte>

# Cambiar usuario mlops-user
# Acceder a Keycloak UI: https://tu-dominio/oauth2/login
# Cambiar contraseña de mlops-user desde Admin Console

docker compose restart keycloak
```

### Paso 2: Configurar CORS para dominio real

```bash
# Editar docker-compose.yml
# Buscar: CORS_ORIGINS=http://localhost,...
# Cambiar a: CORS_ORIGINS=https://tu-app.com,https://tu-dashboard.com

docker compose restart inference-api
```

### Paso 3: Obtener certificados Let's Encrypt

```bash
# En servidor de producción, reemplazar certificados autofirmados
sudo apt install certbot python3-certbot-nginx

# Generar certificado
sudo certbot certonly --standalone -d tu-dominio.com

# Copiar a mlops-stack
sudo cp /etc/letsencrypt/live/tu-dominio.com/fullchain.pem nginx/certs/mlops-cert.pem
sudo cp /etc/letsencrypt/live/tu-dominio.com/privkey.pem nginx/certs/mlops-key.pem
sudo chown $(id -u):$(id -g) nginx/certs/*.pem

# Configurar renovación automática
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

### Paso 4: Configurar respaldos

```bash
# Backup de volumen mlflow-data
docker volume create backup-mlflow-$(date +%Y%m%d)
docker run --rm -v mlops-stack_mlflow-data:/data \
  -v backup-mlflow-$(date +%Y%m%d):/backup \
  alpine tar -czf /backup/mlflow-$(date +%Y%m%d).tar.gz -C /data .

# Respaldo regular (agregar a cron)
# 0 2 * * * docker volume create backup-mlflow-$(date +\%Y\%m\%d) && ...
```

### Paso 5: Configurar monitoreo

```bash
# Agregar health checks a sistema de monitoreo
# Ejemplos:
- https://tu-dominio/mlflow/ → Requiere 200 después de redirect
- https://tu-dominio/api/health → Requiere 200
- docker stats | grep -i "cpu\|mem" → Alertar si CPU >80%
```

---

## Verificación Final

```bash
# Checklist de validación post-despliegue
echo "▶ Validación Final Post-Despliegue"
echo ""

# 1. Todos los servicios running
docker compose ps | grep "Up" | wc -l
# Esperado: 7 (mlflow, model-trainer, inference-api, drift-detector, nginx, keycloak, oauth2-proxy, test-runner)

# 2. Logs sin errores
docker compose logs | grep -i "error\|critical\|failed" | head -5
# Esperado: (vacío o solo warnings)

# 3. Endpoints accesibles
echo "P0-1 (Auth): $(curl -s -k https://localhost/mlflow/ | grep -c 'keycloak' || echo '0')"
echo "P0-2 (CORS): $(curl -s -k -H 'Origin: https://blocked.com' https://localhost/api/health | grep -c 'access-control' | grep -v '1' && echo '✓' || echo '✗')"
echo "P0-3 (HTTPS): $(curl -s -k -I https://localhost/nginx-health | grep -c '200' && echo '✓' || echo '✗')"
echo "P0-4 (Localhost): $(netstat -tlnp 2>/dev/null | grep -c '127.0.0.1' && echo '✓' || echo '✗')"
echo "P0-5 (Perms): $(docker compose exec mlflow ls -la /mlflow/mlflow.db | grep -c '700' && echo '✓' || echo '✗')"

echo ""
echo "✓ Despliegue completado"
```

---

**Documento preparado por:** DevOps Team  
**Fecha:** 2026-06-11  
**Duración estimada de despliegue:** 10-15 minutos (sin issues)
