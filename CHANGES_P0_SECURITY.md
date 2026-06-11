# Documento de Cambios P0 (Críticos) - Seguridad

**Fecha:** 2026-06-11  
**Versión:** 1.0  
**Estado:** Implementado (Pendiente de validación)

---

## Resumen Ejecutivo

Se han implementado **5 cambios críticos de seguridad** (P0) para proteger el pipeline MLOps:

| # | Cambio | Severidad | Estado | Archivos |
|---|--------|-----------|--------|----------|
| P0-1 | Autenticación (Keycloak + OAuth2) | **Crítico** | ✓ Implementado | docker-compose.yml, nginx.conf, setup-keycloak.sh |
| P0-2 | CORS restricción | **Crítico** | ✓ Implementado | inference-api/app.py |
| P0-3 | HTTPS con TLS | **Crítico** | ✓ Implementado | nginx.conf, docker-compose.yml, generate-certs.sh |
| P0-4 | Puertos localhost-only | **Crítico** | ✓ Implementado | docker-compose.yml |
| P0-5 | Permisos de volumen | **Crítico** | ✓ Implementado | mlflow/entrypoint.sh |

---

## 1. P0-1: Autenticación con Keycloak + OAuth2 Proxy

### Descripción
Implementa autenticación OIDC requerida para acceder a MLFlow y API de inferencia.

### Cambios Realizados

#### 1.1 docker-compose.yml
**Cambios:**
- ✅ Agregado volumen `keycloak-data` para persistencia
- ✅ Agregado servicio `keycloak` (puerto 8080, localhost-only)
- ✅ Agregado servicio `oauth2-proxy` (puerto 4180, localhost-only)
- ✅ Ambos servicios reinician automáticamente (`restart: unless-stopped`)

**Nuevos servicios:**
```yaml
keycloak:
  image: quay.io/keycloak/keycloak:23.0.7
  environment:
    KEYCLOAK_ADMIN: admin
    KEYCLOAK_ADMIN_PASSWORD: admin123  # ⚠️ DEBE CAMBIAR en producción
  ports:
    - "127.0.0.1:8080:8080"  # localhost-only

oauth2-proxy:
  image: quay.io/oauth2-proxy/oauth2-proxy:v7.5.1
  environment:
    OAUTH2_PROXY_PROVIDER: oidc
    OAUTH2_PROXY_OIDC_ISSUER_URL: http://keycloak:8080/realms/mlops
    OAUTH2_PROXY_CLIENT_ID: mlops-client
    OAUTH2_PROXY_CLIENT_SECRET: mlops-client-secret  # ⚠️ DEBE CAMBIAR
```

**Ubicación:** `mlops-stack/docker-compose.yml` (líneas 135-188)

#### 1.2 nginx.conf
**Cambios:**
- ✅ Agregado upstream para oauth2-proxy
- ✅ Agregado endpoint `/oauth2/` para manejo de autenticación
- ✅ Agregado endpoint `/oauth2/auth` para verificación
- ✅ Rutas `/mlflow/` y `/api/` ahora requieren `auth_request`
- ✅ Headers de usuario (`X-User`, `X-Email`) pasan a upstream services

**Cambios clave:**
```nginx
upstream oauth2_proxy { server oauth2-proxy:4180; }

location = /oauth2/auth {
    proxy_pass http://oauth2_proxy;
    # Verificación de token sin body
}

location /mlflow/ {
    auth_request /oauth2/auth;  # ← Protegido
    auth_request_set $user $upstream_http_x_auth_request_user;
    # ...
}
```

**Ubicación:** `mlops-stack/nginx/nginx.conf` (líneas 1-73)

#### 1.3 setup-keycloak.sh (Nuevo)
**Propósito:** Configurar automáticamente Keycloak post-despliegue

**Acciones:**
- Espera a que Keycloak esté disponible
- Obtiene token de admin
- Crea realm `mlops`
- Crea cliente OIDC `mlops-client`
- Crea usuario de prueba `mlops-user` / `mlops123`
- Imprime client secret para actualizar docker-compose

**Ubicación:** `mlops-stack/setup-keycloak.sh`

### Flujo de Autenticación

```
Cliente → Nginx (HTTPS)
           ↓
        auth_request → OAuth2-Proxy
           ↓
           ├─ Token inválido/ausente → Redirige a Keycloak
           │
           ├─ Keycloak: Usuario ingresa credenciales
           │
           ├─ Keycloak genera token OIDC
           │
           └─ OAuth2-Proxy verifica token → Permite acceso
                ↓
           Headers (X-User, X-Email) → MLFlow/API
```

### Configuración para Producción
```yaml
# En docker-compose.yml, cambiar:
KEYCLOAK_ADMIN_PASSWORD: <generar contraseña fuerte>
OAUTH2_PROXY_CLIENT_SECRET: <obtener de setup-keycloak.sh>
OAUTH2_PROXY_REDIRECT_URL: https://tu-dominio.com/oauth2/callback
```

---

## 2. P0-2: CORS Restricción

### Descripción
Cambiar de CORS permisivo (`["*"]`) a dominios específicos.

### Cambios Realizados

#### 2.1 inference-api/app.py

**Antes:**
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

**Después:**
```python
# Variable de entorno con valores por defecto (desarrollo)
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS", 
    "http://localhost,http://localhost:3000,http://localhost:8080"
).split(",")

# Configuración restrictiva
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],              # Solo GET/POST
    allow_headers=["Content-Type", "Accept"],   # Headers limitados
    max_age=3600,                               # Cache 1 hora
    allow_credentials=True,
)
```

**Cambios clave:**
- ✅ Dominios específicos (no `*`)
- ✅ Métodos limitados (GET, POST)
- ✅ Headers limitados (Content-Type, Accept)
- ✅ Credenciales permitidas (cookies, auth)

**Ubicación:** `mlops-stack/inference-api/app.py` (líneas 24-86)

**Configuración para Producción:**
```bash
# docker-compose.yml o .env
CORS_ORIGINS=https://tu-app.com,https://tu-dashboard.com
```

---

## 3. P0-3: HTTPS con Certificado TLS

### Descripción
Cambiar de HTTP a HTTPS requerido, redirigir HTTP → HTTPS.

### Cambios Realizados

#### 3.1 generate-certs.sh (Nuevo)
**Propósito:** Generar certificados autofirmados

**Acciones:**
- Genera clave privada RSA 4096-bit
- Genera certificate signing request (CSR)
- Genera certificado autofirmado válido por 365 días
- Establece permisos correctos (600 para key, 644 para cert)

**Uso:**
```bash
bash generate-certs.sh localhost
# O: bash generate-certs.sh tu-dominio.com
```

**Output:**
```
./nginx/certs/mlops-key.pem    (clave privada)
./nginx/certs/mlops-cert.pem   (certificado)
./nginx/certs/mlops.csr        (CSR - puede eliminarse)
```

**Ubicación:** `mlops-stack/generate-certs.sh`

#### 3.2 nginx.conf

**Antes:**
```nginx
server {
    listen 80;
    # HTTP sin encriptación
}
```

**Después:**
```nginx
# Servidor HTTP: redirige todo a HTTPS
server {
    listen 80;
    return 301 https://$host$request_uri;
}

# Servidor HTTPS: con certificados TLS
server {
    listen 443 ssl;
    
    ssl_certificate /etc/nginx/certs/mlops-cert.pem;
    ssl_certificate_key /etc/nginx/certs/mlops-key.pem;
    
    # Configuración segura de TLS
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
}
```

**Cambios clave:**
- ✅ HTTP → HTTPS redirect automático
- ✅ TLS 1.2 y 1.3 (sin SSL 3.0, TLS 1.0, 1.1)
- ✅ Ciphers fuertes (HIGH, no-NULL, no-MD5)
- ✅ Session caching para performance

**Ubicación:** `mlops-stack/nginx/nginx.conf` (líneas 1-73)

#### 3.3 docker-compose.yml

**Cambios:**
- ✅ Nginx puertos: `80` y `443`
- ✅ Volumen para certificados: `./nginx/certs:/etc/nginx/certs:ro`

```yaml
nginx:
  ports:
    - "80:80"      # HTTP (redirige)
    - "443:443"    # HTTPS (principal)
  volumes:
    - ./nginx/certs:/etc/nginx/certs:ro  # Certificados (read-only)
```

**Ubicación:** `mlops-stack/docker-compose.yml` (líneas 119-133)

### Configuración para Producción (Let's Encrypt)
```bash
# En servidor de producción, usar certbot:
sudo apt install certbot python3-certbot-nginx
sudo certbot certonly --nginx -d tu-dominio.com

# Copiar certificados a:
/opt/mlops-stack/nginx/certs/mlops-cert.pem
/opt/mlops-stack/nginx/certs/mlops-key.pem

# Auto-renovación:
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

---

## 4. P0-4: Puertos Localhost-Only

### Descripción
Cambiar puertos de `0.0.0.0` (accesible desde cualquier red) a `127.0.0.1` (localhost-only).

### Cambios Realizados

#### 4.1 docker-compose.yml

**Antes:**
```yaml
services:
  mlflow:
    ports:
      - "5001:5001"          # Accesible desde cualquier IP

  inference-api:
    ports:
      - "8000:8000"          # Accesible desde cualquier IP

  keycloak:
    ports:
      - "8080:8080"          # Accesible desde cualquier IP

  oauth2-proxy:
    ports:
      - "4180:4180"          # Accesible desde cualquier IP

  nginx:
    ports:
      - "80:80"              # HTTP (único puerto público)
      - "443:443"            # HTTPS (único puerto público)
```

**Después:**
```yaml
services:
  mlflow:
    ports:
      - "127.0.0.1:5001:5001"    # localhost-only

  inference-api:
    ports:
      - "127.0.0.1:8000:8000"    # localhost-only

  keycloak:
    ports:
      - "127.0.0.1:8080:8080"    # localhost-only

  oauth2-proxy:
    ports:
      - "127.0.0.1:4180:4180"    # localhost-only

  nginx:
    ports:
      - "80:80"                  # HTTP (redirige a HTTPS)
      - "443:443"                # HTTPS (único público, requiere auth)
```

**Cambios clave:**
- ✅ Microservicios internos (`127.0.0.1`) — acceso solo local
- ✅ Nginx es único punto de entrada público
- ✅ Nginx redirige HTTP → HTTPS + require OAuth2

**Ubicación:** `mlops-stack/docker-compose.yml`
- mlflow: línea 23
- inference-api: línea 54
- keycloak: línea 150
- oauth2-proxy: línea 176
- nginx: línea 125

### Impacto
```
Antes:
  localhost:5001 → MLFlow (público, sin auth)
  localhost:8000 → API (público, sin auth, CORS abierto)
  Internet:5001  → MLFlow (¡GRAVE RIESGO!)
  Internet:8000  → API (¡GRAVE RIESGO!)

Después:
  localhost:5001 → MLFlow (local, auth via nginx)
  localhost:8000 → API (local, auth via nginx)
  Internet       → Nginx (HTTPS, requiere OAuth2)
  Internet       → MLFlow (bloqueado, solo via nginx autenticado)
  Internet       → API (bloqueado, solo via nginx autenticado)
```

---

## 5. P0-5: Permisos de Volumen

### Descripción
Cambiar de `chmod -R 777` (todos pueden leer/escribir) a permisos restrictivos.

### Cambios Realizados

#### 5.1 mlflow/entrypoint.sh

**Antes:**
```bash
mkdir -p /mlflow/artifacts
chmod -R 777 /mlflow  # ← INSEGURO: todos pueden leer/escribir TODO
```

**Después:**
```bash
mkdir -p /mlflow/artifacts

# SECURITY FIX P0-5: Restringir permisos
# - Base de datos: 700 (solo dueño puede leer/escribir)
# - Artifacts: 755 (dueño lee/escribe, otros solo leen)
chmod 700 /mlflow/mlflow.db 2>/dev/null || true
chmod 755 /mlflow/artifacts
chown -R mlflow:mlflow /mlflow 2>/dev/null || true
```

**Cambios clave:**
- ✅ `mlflow.db`: `700` — Solo dueño (rwx), sin acceso para otros
- ✅ `artifacts/`: `755` — Dueño rwx, otros rx (lectura y ejecución)
- ✅ `chown`: Establece propietario (mlflow:mlflow)

**Impacto:**
```
Antes (777):
  -rwxrwxrwx  mlflow.db      ← Cualquiera puede modificar BD
  -rwxrwxrwx  modelo.pkl     ← Cualquiera puede reemplazar modelo
  
Después (700/755):
  -rwx------  mlflow.db      ← Solo mlflow puede acceder
  -rwxr-xr-x  artifacts/     ← mlflow escribe, otros leen
  -rwxr-xr-x  modelo.pkl     ← mlflow escribe, otros leen
```

**Ubicación:** `mlops-stack/mlflow/entrypoint.sh` (líneas 4-11)

---

## Archivos Modificados y Creados

### Modificados:
| Archivo | Cambios | Líneas |
|---------|---------|--------|
| `docker-compose.yml` | P0-1, P0-3, P0-4 (Keycloak, oauth2, nginx, volúmenes) | +70 |
| `inference-api/app.py` | P0-2 (CORS restricción) | +4 |
| `nginx/nginx.conf` | P0-1, P0-3 (Auth + HTTPS) | ~45 (reescrito) |
| `mlflow/entrypoint.sh` | P0-5 (Permisos) | +5 |

### Creados (Nuevos):
| Archivo | Propósito |
|---------|-----------|
| `generate-certs.sh` | P0-3 — Script para generar certificados autofirmados |
| `setup-keycloak.sh` | P0-1 — Script para configurar Keycloak realm + cliente |

### Total: 4 archivos modificados + 2 nuevos = **6 cambios**

---

## Checklist de Implementación

- [x] P0-1: Keycloak + OAuth2 proxy agregado a docker-compose.yml
- [x] P0-1: nginx.conf actualizado con auth_request
- [x] P0-1: setup-keycloak.sh creado para auto-configuración
- [x] P0-2: CORS restricción implementada en app.py
- [x] P0-2: Variable de entorno CORS_ORIGINS añadida
- [x] P0-3: generate-certs.sh creado
- [x] P0-3: nginx.conf actualizado con HTTPS + HTTP redirect
- [x] P0-3: docker-compose.yml con volumen de certs
- [x] P0-4: Todos los puertos internos a localhost-only (127.0.0.1)
- [x] P0-4: nginx puertos públicos (80, 443) sin restricción local
- [x] P0-5: mlflow/entrypoint.sh actualizado con permisos 700/755
- [x] P0-5: chown para establecer propietario correcto

---

## Verificación Inicial

### Pre-despliegue:
```bash
# Validar sintaxis YAML
docker-compose config > /dev/null

# Validar sintaxis nginx
docker run --rm -v $(pwd)/nginx:/etc/nginx:ro nginx:latest nginx -t

# Generar certificados
bash generate-certs.sh localhost
```

### Post-despliegue:
```bash
# Verificar servicios activos
docker compose ps

# Verificar certificados
docker compose exec nginx openssl x509 -in /etc/nginx/certs/mlops-cert.pem -text -noout

# Verificar auth_request
curl -k https://localhost/mlflow/  # Debe redirigir a Keycloak

# Verificar CORS
curl -k -H "Origin: https://blocked.com" https://localhost/api/health
# Debe retornar error 403 (no CORS header)
```

---

## Notas Importantes

⚠️ **Credenciales por defecto (DEBEN CAMBIAR en producción):**
```
Keycloak Admin:
  - Usuario: admin
  - Contraseña: admin123

Keycloak Test User:
  - Usuario: mlops-user
  - Contraseña: mlops123

OAuth2 Client Secret:
  - Obtener de: bash setup-keycloak.sh (post-despliegue)
```

⚠️ **Certificados autofirmados (solo desarrollo):**
```
generate-certs.sh crea certificados autofirmados válidos por 365 días.
Para producción, usar Let's Encrypt:
  - certbot certonly --nginx -d tu-dominio.com
```

⚠️ **CORS Origins (configurar para tu dominio):**
```
Desarrollo: http://localhost,http://localhost:3000
Producción: https://tu-app.com,https://tu-dashboard.com
```

---

## Próximos Pasos

Ver: `DEPLOYMENT_GUIDE_P0.md` para instrucciones de despliegue y validación.

---

**Documento preparado por:** Security Implementation Team  
**Fecha:** 2026-06-11  
**Versión:** 1.0
