# Evaluación de Seguridad: Almacenamiento, Accesibilidad e Integridad de Datos

**Fecha:** 2026-06-11  
**Repositorio:** ai-deployment  
**Scope:** Ciclo de vida completo del modelo (despliegue → entrenamiento → testing → registro → producción)

---

## 📋 Tabla de Contenidos
1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Etapa 1: Despliegue de Módulos](#etapa-1-despliegue-de-módulos)
3. [Etapa 2: Entrenamiento](#etapa-2-entrenamiento)
4. [Etapa 3: Testing](#etapa-3-testing)
5. [Etapa 4: Promoción a Registro](#etapa-4-promoción-a-registro)
6. [Etapa 5: Despliegue en Producción](#etapa-5-despliegue-en-producción)
7. [Matriz de Riesgos](#matriz-de-riesgos)
8. [Plan de Remediación](#plan-de-remediación)

---

## Resumen Ejecutivo

| Aspecto | Riesgo | Estado |
|--------|--------|--------|
| **Integridad de datos** | Media | ✓ Quality gates + tests |
| **Almacenamiento** | Media | ⚠ Local filesystem, sin encriptación |
| **Accesibilidad** | **Alto** | ❌ Sin autenticación, CORS permisivo |
| **Cumplimiento normativo** | Desconocido | ? Requiere auditoría de datos PII |
| **Confidencialidad de modelos** | Media | ⚠ Sin encriptación en tránsito/reposo |

**Puntuación General:** 6/10 ✓ (Desarrollable, no productiva)

---

## Etapa 1: Despliegue de Módulos

### 1.1 Almacenamiento de Código Fuente

**Actual:**
- Repositorio GitHub con CI/CD via GitHub Actions self-hosted
- Flujo: Código → Git → Runner → Docker build → Deploy local

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| A1 | Credenciales en Git | Alto | `PROD_SSH_KEY` en GitHub Secrets (buena práctica), pero verify `.gitignore` |
| A2 | Docker images sin firma | Medio | No hay verificación de integridad de imágenes |
| A3 | Imágenes públicas | Bajo | Usa imágenes base públicas (python:3.11, mlflow) sin pinning de SHA |
| A4 | Sin SCA (SAST) | Medio | No hay static code analysis en CI (bandit, semgrep) |

**Verificaciones:**

```bash
# ✓ .gitignore contiene:
.env
*.env
.env.local
*.pyc
__pycache__/
mlruns/
.DS_Store

# ✓ Secretos en GitHub Secrets:
PROD_SSH_KEY → rsync a /opt/mlops-stack
PROD_HOST → dirección del servidor

# ❌ Falta: SBOM (Software Bill of Materials)
# ❌ Falta: Vulnerability scanning de dependencias
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  - Adicionar trivy scanning en CI para imágenes Docker
  - Pinear versiones base: python:3.11-slim@sha256:...
  - Ejecutar bandit en CI (security linter)

Corto plazo (P1):
  - Generar SBOM con syft en build
  - Firmar imágenes con cosign
  - Scan dependencias Python: safety, pip-audit

Largo plazo (P2):
  - Registry privado con autenticación (Quay, ACR, ECR)
  - Immutable tags (usar git SHA como tag)
```

---

### 1.2 Orquestación (Docker Compose)

**Actual:**
```yaml
# docker-compose.yml
services:
  mlflow:
    image: python:3.11
    volumes:
      - mlflow-data:/mlflow
    environment:
      MLFLOW_ARTIFACT_ROOT: /mlflow/artifacts
    # Sin limites de recursos
    # Sin health checks entre servicios
    # Sin restart policy específica
```

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| D1 | Sin límites de recursos | Medio | Contenedores pueden agotar CPU/memoria |
| D2 | Volumen chmod 777 | Alto | Todos los usuarios pueden leer/escribir modelos |
| D3 | Network segmentation | Medio | Todos los contenedores en red compartida sin aislamiento |
| D4 | Sin secrets management | Bajo | Variables hardcoded en compose (desarrollo OK, producción NO) |
| D5 | Health checks ausentes | Bajo | Sin verificación de disponibilidad entre servicios |

**Verificaciones:**

```bash
# Volumen actual:
/mlflow/ → chmod 777 (desarrollo)
ls -la /mlflow
  drwxrwxrwx root:root  .
  drwxrwxrwx root:root  artifacts/
  -rw-r--r-- root:root  mlflow.db

# Red actual:
docker network ls | grep mlops-net
  mlops-net  bridge  mlflow/inference-api/drift-detector/nginx/tests

# Sin restricciones:
- Todos los contenedores pueden acceder a todos los puertos
- Sin firewall ingress/egress
- Sin network policies (Kubernetes style)
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  - Cambiar permisos de volumen: mlflow-data:/mlflow:ro (read-only para servicios que no escriben)
  - Adicionar limits: memory: 512m, cpus: '1.0'
  - Usar secrets.yml para credenciales (Docker Secrets en Swarm)

Corto plazo (P1):
  - Separar networks: 
    - mlops-internal (mlflow, trainer, drift-detector, api)
    - mlops-public (nginx solamente)
  - Health checks en todos los servicios
  - Deploy en Docker Swarm o Kubernetes con RBAC

Largo plazo (P2):
  - Secrets management externo: Vault, AWS Secrets Manager
  - mTLS entre servicios internos
  - Pod security policies (Kubernetes)
```

---

## Etapa 2: Entrenamiento

### 2.1 Datos de Entrada

**Actual:**
```python
# model-trainer/train.py
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)  # Cargado en memoria, 150 muestras
```

**Características:**
- ✓ Dataset público (Iris), sin PII
- ✓ Determinístico (siempre mismo resultado)
- ✓ No conecta a BD externa
- ✓ Tamaño pequeño (150 muestras, 4 features, 4.8KB)

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| T1 | Sin validación de datos | Medio | No valida integridad de Iris dataset |
| T2 | Datos cacheados | Bajo | sklearn cache en ~/.cache/scikit-learn - permisos inseguros |
| T3 | Sin auditoría de origen | Bajo | No verifica SHA256 del dataset descargado |
| T4 | Reproducibilidad débil | Bajo | Usa seed pero no documenta dataset version |

**Verificaciones:**

```python
# Actual: Sin validación
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# Integridad de dataset
from sklearn.datasets import load_iris
iris = load_iris()
print(f"Samples: {iris.data.shape[0]}")  # Siempre 150
print(f"Features: {iris.data.shape[1]}")  # Siempre 4
print(f"Classes: {len(np.unique(iris.target))}")  # Siempre 3

# Falta: Logging de hash
import hashlib
data_hash = hashlib.sha256(iris.data.tobytes()).hexdigest()
logger.info(f"Dataset hash: {data_hash}")  # Detectaría cambios

# Falta: Verificación de cache integrity
cache_path = ~/.cache/scikit-learn/datasets/iris.npy
os.stat(cache_path).st_mode  # Podría ser world-readable
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  - Adicionar validación post-carga:
    * Assert shape == (150, 4)
    * Assert unique(y) == [0, 1, 2]
    * Assert no NaN/Inf
  - Log dataset hash (SHA256) para reproducibilidad
  - Usar random.seed(42) explícitamente

Corto plazo (P1):
  - Versionar dataset: data/iris-v1.0.pkl con checksum
  - Almacenar en bucket S3/GCS con ACLs
  - Validación de origen: verificar SSL cert del servidor

Largo plazo (P2):
  - Data lineage tracking (Apache Atlas, Collibra)
  - Data versioning (DVC, Weights&Biases)
  - Access logs para auditoría (quién accedió cuándo)
```

---

### 2.2 Almacenamiento de Modelo Entrenado

**Actual:**
```
/mlflow/artifacts/
├── 0/                           # experiment-id
│   └── abc123def456/            # run-id
│       ├── model/               # Modelo serializado (pickle)
│       │   ├── MLmodel          # Metadata
│       │   ├── model.pkl        # RandomForest
│       │   └── requirements.txt
│       └── params/
│           └── max_depth=5      # Hiperparámetros
```

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| T5 | Sin encriptación en reposo | Alto | Modelo accessible en plain text (pickle) |
| T6 | Sin firma de integridad | Medio | Nada previene tampering del pickle |
| T7 | Permisos permisivos | Alto | chmod 777 en /mlflow/artifacts |
| T8 | Sin versionamiento explícito | Bajo | MLFlow versiones internamente pero no en metadata |
| T9 | Pickle es inseguro | Medio | Deserialización de pickle puede ejecutar código |
| T10 | Sin backup/replicación | Medio | Solo en volumen local (pierde datos si VM crashea) |

**Verificaciones:**

```bash
# Permisos actuales:
ls -la /mlflow/artifacts/0/abc123/model/
  -rw-r--r-- root:root model.pkl  # ✓ Legible por todos

# Sin encriptación:
file /mlflow/artifacts/0/abc123/model/model.pkl
  model.pkl: data  # Pickle binario, no encriptado
openssl enc -d -in model.pkl  # No funciona

# Sin firma:
ls /mlflow/artifacts/0/abc123/model/
  # No hay .sig o .asc file

# Sin replicación:
docker inspect mlops-stack_mlflow_1 | jq '.Mounts'
  [{"Source": "/var/lib/docker/volumes/mlflow-data/_data",
    "Destination": "/mlflow"}]
  # Single point of failure
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  - Cambiar formato de pickle a ONNX (más seguro, portable)
    sklearn.model_selection -> onnx (open standard)
  - Cambiar permisos: 
    chmod 755 /mlflow/artifacts/ (rx para otros, no write)
    chmod 700 /mlflow/mlflow.db   (solo dueño)
  - Adicionar checksum: 
    sha256sum model.pkl > model.pkl.sha256
    Verificar en inferencia: sha256sum -c model.pkl.sha256

Corto plazo (P1):
  - Encriptación en reposo:
    LUKS para volumen Docker
    O: Usar S3 server-side encryption (SSE-S3 o SSE-KMS)
  - Backup:
    docker cp mlops-stack_mlflow_1:/mlflow /backup/mlflow-$(date +%Y%m%d)
    O: NFS mount con snapshotting
  - Firmar modelos: 
    gpg --detach-sign model.pkl → model.pkl.sig
    Verificar en carga: gpg --verify model.pkl.sig model.pkl

Largo plazo (P2):
  - Model signing certificate (X.509)
  - Hardware security module (HSM) para keys
  - Encrypted model registry (HashiCorp Vault)
  - Geo-replicated backup (S3 cross-region)
```

---

### 2.3 Logs de Entrenamiento

**Actual:**
```python
# Logs en stdout/stderr de contenedor
docker logs mlops-stack_trainer_1 | grep -i error
# MLFlow logs internos en /mlflow/mlflow.db (SQLite)
```

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| T11 | Logs sin rotación | Bajo | Docker daemon logs pueden crecer indefinidamente |
| T12 | Sin log level config | Bajo | Todo va a stdout (verbose) |
| T13 | Logs accesibles | Medio | Cualquier usuario con acceso Docker puede ver logs |
| T14 | Sin audit trail | Medio | No hay registro de quién cambió qué parámetro |
| T15 | SQLite vulnerable | Bajo | Acceso concurrente puede causar corruption |

**Verificaciones:**

```bash
# Logs del contenedor:
docker logs mlops-stack_trainer_1 --tail 100 --timestamps
  # Sin rotación configurada
  # Acceso: docker logs (requiere docker group membership)

# MLFlow DB:
file /mlflow/mlflow.db
  SQLite 3.x database, last written at ...
  
# Sin audit log:
grep -r "user=" /mlflow/mlruns  # No hay metadata de usuario
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  - Configurar log driver: docker-compose.yml
    logging:
      driver: json-file
      options:
        max-size: 100m
        max-file: 10
  - Filtrar logs sensibles (hiperparámetros, paths):
    logging.getLogger("mlflow").setLevel(logging.WARNING)

Corto plazo (P1):
  - Centralizar logs: ELK Stack, Datadog, Splunk
    docker-compose adiciona logstash service
  - Audit trail en MLFlow:
    mlflow.set_tag("user", os.getenv("USER"))
    mlflow.set_tag("timestamp", datetime.now())
  - Access logs: Nginx log all requests a /mlflow

Largo plazo (P2):
  - Immutable logging (append-only, no delete)
  - Log aggregation con encryption en tránsito
  - Retention policy: 90 días production, 7 días development
  - SIEM integration: Splunk, ELK búsquedas de anomalías
```

---

## Etapa 3: Testing

### 3.1 Integridad de Tests

**Actual:**
```python
# tests/run_tests.py - 4 niveles de QA
Level 1: test_data.py       # Validación de datos
Level 2: test_model.py      # Quality gates (accuracy ≥0.90)
Level 3: test_api.py        # API health y latencia (<500ms)
Level 4: test_performance_drift.py  # Data drift + Performance drift
```

**Fortalezas:**
- ✓ 4 niveles de tests (data → model → api → drift)
- ✓ Quality gates automáticos (bloquean promoción)
- ✓ >600 líneas de código de tests
- ✓ Detecta degradación de modelo
- ✓ Valida integridad de entrada en API

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| T16 | Tests no son versionados | Bajo | Sin .git en /tests, pueden cambiar sin auditoría |
| T17 | Sin coverage mínimo | Medio | No hay pytest --cov enforcement (podría estar al 50%) |
| T18 | Tests corruptibles | Medio | Test runner tiene acceso rw a artifacts |
| T19 | Comparación sin checksum | Bajo | Quality gates comparan accuracy float (32-bit precision) |
| T20 | Sin timestamp de test | Bajo | No hay evidencia de cuándo se corrió cada test |

**Verificaciones:**

```python
# test_model.py - Quality gates actuales:
assert model.score(X_test, y_test) >= 0.90  # ✓ Bien
# Pero: Usa accuracy (promedio simple) - podría estar sesgado

# test_api.py - Validación de input:
def validate_input(data):
    """Valida 4 features y rango 0-1 para probabilidades"""
    if len(data) != 4:
        raise ValueError(f"Expected 4 features, got {len(data)}")
    # ✓ Buena validación
    
# Falta: Assert tests son idempotentes
# Falta: Logging de test run with timestamp
# Falta: Test report firmado/sellado
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  - Adicionar pytest coverage mínimo:
    pytest --cov=tests --cov-fail-under=80
  - Timestamp y firma de test runs:
    json_report con test_run_id = sha256(date + seed)
  - Test isolation:
    Cada test crea/limpia su dataset (no comparte estado)

Corto plazo (P1):
  - Versionamiento de tests con Git:
    tests/ → Git repo con CI checks
  - Test report inmutable:
    Guardar en /mlflow/test-reports/run-{timestamp}.json
    Verificar sha256 antes de usarlo
  - Precision de comparison:
    accuracy = round(score, 4)  # Evitar float precision issues

Largo plazo (P2):
  - Test data provenance: lineage de qué datos usó cada test
  - Mutation testing: Cambiar código y verificar que tests fallan
  - Regression suite: Tests de versiones anteriores
  - Statistical testing framework: Welch's t-test en lugar de assert simple
```

---

### 3.2 Aislamiento de Tests

**Actual:**
```bash
# test-runner container
docker run mlops-test-runner
  # Acceso a:
  - http://mlflow:5001      ← Puede escribir experimentos
  - http://inference-api:8000  ← Puede cambiar modelo
  - /mlflow/artifacts       ← Volumen compartido
```

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| T21 | Tests modifican production data | Alto | Test runner escribe en /mlflow/artifacts |
| T22 | Sin isolación de experiments | Medio | Tests crean experiments que interfieren con training |
| T23 | Acceso de lectura a modelos | Bajo | Tests pueden leer modelo production (OK) pero no deberían poder borrar |
| T24 | Sin rollback de cambios | Medio | Si test falla, cambios quedan en BD |

**Verificaciones:**

```bash
# Tests crean experimentos nuevos:
mlflow experiments list
  0  Default
  1  test-run-{timestamp}  ← Creado por tests
  2  test-run-{timestamp}  ← Otro test run
  # Proliferación de experiments

# Test modifica modelo:
docker exec mlops-stack_test_runner_1 \
  curl -X POST http://inference-api:8000/reload
  # Reinicia la API en mitad del test
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  - Tests en volumen separado: mlflow-test:/mlflow-test
    No compartir /mlflow/artifacts
  - Experiments aislados:
    MLFLOW_TRACKING_URI_TEST = http://mlflow:5001/experiments/test
    Limpiar después con mlflow experiments delete
  - API en modo test: inference-api:8000/test (sandbox)
    No afecta Production stage

Corto plazo (P1):
  - Database transactions:
    Envolver tests en transacciones, rollback post-test
  - Namespace de MLFlow:
    Usar tags para separar test runs: tag "test:true"
    Query: mlflow.search_runs(experiment_names=["test"], filter_string="tag.test=true")
  - Snapshot/restore:
    docker volume create mlflow-test-backup-$(date +%s)
    cp -r /mlflow/test-data -> volumen backup

Largo plazo (P2):
  - Ephemeral test environments: Kubernetes pods con cleanup automático
  - Test data federation: Datos de test en BD separada (PostgreSQL test DB)
  - Contract testing: Validar que API no cambia schema (test sin lado effects)
```

---

## Etapa 4: Promoción a Registro

### 4.1 Model Registry (MLFlow)

**Actual:**
```python
# model-trainer/train.py
mlflow.register_model(model_uri="runs:/{run_id}/model", name="iris-classifier")
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="iris-classifier",
    version=1,
    stage="Production"
)
```

**Flujo de promoción:**
```
Train (local) → Register (MLFlow) → Staging → Test → Production
                                     ↓         ↓
                                  Quality gates (API test)
                                  Performance drift check
```

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| P1 | Sin RBAC en MLFlow | Alto | Cualquier contenedor puede promover modelos a Production |
| P2 | Sin aprobación manual | Medio | Promotion automática sin revisor humano |
| P3 | Sin audit trail de cambios | Medio | No hay log de quién/cuándo promovió |
| P4 | SQLite no es escalable | Bajo | Una sola transacción puede causar contention |
| P5 | Sin rollback automático | Medio | Si Production model falla, no hay revert |
| P6 | Metadata sin integridad | Bajo | Cambiar alias de versión sin detección |

**Verificaciones:**

```python
# Sin autenticación:
client = mlflow.tracking.MlflowClient("http://mlflow:5001")
client.get_registered_model("iris-classifier")
  # Funciona sin credenciales, responde JSON

# Sin RBAC:
client.transition_model_version_stage(
    name="iris-classifier",
    version=1,
    stage="Production"
)
# Funciona desde cualquier contenedor

# Audit trail falta:
mlflow.search_model_versions("name='iris-classifier'")
  # Response no incluye quién hizo el cambio, timestamp sí pero sin timezone
  
# SQLite bottleneck:
file /mlflow/mlflow.db
  SQLite 3.x  # Una conexión por vez (readers wait)
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  - Adicionar audit tagging en Model Registry:
    client.set_model_version_description(
      name="iris-classifier",
      version=1,
      description="Promoted to Production by trainer-job-123 at 2026-06-11T10:30Z"
    )
  - Revertible promotion (Champion/Challenger):
    if new_accuracy > old_accuracy + MIN_DELTA:
        promote(new_model)
    else:
        keep(old_model)  # Fallback automático
  
Corto plazo (P1):
  - RBAC en MLFlow:
    Migrar de SQLite a PostgreSQL
    Usar OIDC/LDAP para autenticación
    Roles: trainer (write), approver (transition), reader (get)
  - Manual approval gate:
    Promotion requiere click en UI o API POST de approver
    Log: model-promotions.json con signature de approver
  - Timeout de promoción:
    Si test falla, auto-revert a versión anterior en 5 min

Largo plazo (P2):
  - Model signing:
    PGP sign el modelo antes de promotion
    Verificar firma en inference-api antes de cargar
  - Promotion audit log en Elasticsearch:
    Immutable, searchable, con alertas de cambios no autorizados
  - Multi-stage promotion:
    None → Dev → Staging → Canary → Production
    Cada stage requiere tests distintos
  - Approval workflow: 
    Jira/GitHub issue vinculado a cada promotion
    Requiere review de 2 personas (four-eyes principle)
```

---

### 4.2 Versionamiento de Modelos

**Actual:**
```
iris-classifier
├── Version 1 (None)      # Versión inicial, sin stage
├── Version 2 (None)      # Nuevo training
├── Version 3 (Staging)   # En pruebas
└── Version 4 (Production) # Activa
```

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| P7 | Versiones no inmutables | Medio | Cambiar metadata de versión sin detección |
| P8 | Sin changelog | Bajo | No hay registro de cambios entre versiones |
| P9 | Aliasing frágil | Bajo | Si "Production" apunta a v4, ¿qué pasa si se borra? |
| P10 | Sin pin de dependencias | Medio | Model puede usar scikit-learn 1.0 o 1.3 (incompatible) |

**Verificaciones:**

```python
# Metadata de versión (vulnerable a cambios):
response = mlflow.search_model_versions("name='iris-classifier'")
for version in response:
    print(version.source)  # Archivo pickle
    print(version.run_id)  # Experiment run que lo generó
    # Sin checksum de integridad

# Sin changelog:
mlflow.get_registered_model("iris-classifier")
  # Response no incluye "version_history" o "changelog"

# Sin pin de scikit-learn:
model.pkl contiene RandomForest
  # ¿Cuál versión de sklearn usó para train?
  # ¿Es compatible con sklearn 1.3?
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  - Metadata inmutable con checksum:
    model_metadata = {
      "version": 4,
      "training_sha256": "abc123...",
      "model_sha256": "def456...",
      "scikit_learn_version": "1.0.2",
      "created_at": "2026-06-11T10:00Z"
    }
    Guardar en model/MLmodel junto al pickle
  
  - Changelog en cada promoción:
    Version 4:
      - Created: 2026-06-11T10:00Z by trainer-job-123
      - Promoted to Staging: 2026-06-11T10:05Z (tests passed)
      - Promoted to Production: 2026-06-11T10:15Z (manual approval)

Corto plazo (P1):
  - Requirements pinning:
    model/requirements.txt con versiones exactas:
      scikit-learn==1.0.2
      numpy==1.23.5
      joblib==1.2.0
    Verificar compatibilidad en test antes de promotion
  
  - Snapshots de versión:
    Nunca actualizar versión existente, solo crear nueva
    client.transition_model_version_stage(...) es OK
    Pero NO permitir: client.update_model_version(...)
  
  - Version locking:
    Una vez en Production, crear backup:
    cp /mlflow/artifacts/0/v4/model /mlflow/archive/iris-classifier-v4-2026-06-11

Largo plazo (P2):
  - Semantic versioning:
    MAJOR.MINOR.PATCH (1.2.3)
    MAJOR = data format changed (incompatible)
    MINOR = new feature (backward compatible)
    PATCH = bug fix
  
  - Model card requirement:
    Antes de production, requerir model card:
    - Training dataset version
    - Performance metrics
    - Known limitations
    - Fairness assessment
```

---

## Etapa 5: Despliegue en Producción

### 5.1 Infraestructura y Acceso

**Actual:**
```yaml
# Producción: localhost en máquina
Services:
  - MLFlow UI: http://localhost:5001 (sin auth)
  - Inference API: http://localhost:8000 (sin auth)
  - Nginx Proxy: http://localhost:80
    - /mlflow/ → http://mlflow:5001
    - /api/ → http://inference-api:8000

CORS:
  Inference API: allow_origins=["*"]  # INSEGURO
  
Network:
  Docker bridge network (todos los contenedores accesibles)
```

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| P11 | Sin autenticación | **Crítico** | Cualquiera puede acceder a /mlflow UI y cambiar modelos |
| P12 | CORS permisivo | **Crítico** | API accessible desde JavaScript en cualquier sitio |
| P13 | Sin HTTPS/TLS | **Crítico** | Comunicación en plain HTTP |
| P14 | Sin autorización | Alto | No hay roles (admin/user/read-only) |
| P15 | Firewall abierto | Alto | Puertos 80, 5001, 8000 accesibles desde internet |
| P16 | Sin rate limiting | Medio | Posible DoS attacks |
| P17 | Sin WAF | Medio | SQL injection, XSS en MLFlow UI |
| P18 | Logs accesibles | Medio | docker logs visible para todos |

**Verificaciones:**

```bash
# Sin autenticación:
curl http://localhost:5001/api/2.0/mlflow/registered-models/list
# Devuelve JSON sin credenciales

# CORS permisivo:
curl -H "Origin: https://attacker.com" http://localhost:8000/predict
  # Header: Access-Control-Allow-Origin: *
  # JavaScript del attacker.com puede acceder

# Sin HTTPS:
curl -v http://localhost:5001
  # Connection: HTTP/1.1, no SSL/TLS

# Puertos expuestos:
netstat -tlnp | grep LISTEN
  0.0.0.0:80     LISTEN  docker (nginx)
  0.0.0.0:5001   LISTEN  docker (mlflow)
  0.0.0.0:8000   LISTEN  docker (inference-api)
  # Accesibles desde 0.0.0.0 (cualquier red)

# Sin rate limiting:
for i in {1..1000}; do curl http://localhost:8000/health; done
# Posible DoS
```

**Recomendaciones:**

```yaml
Inmediato (P0) - CRÍTICO:
  1. Restringir CORS:
     app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://trusted-domain.com"],  # Específico
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["Content-Type"],
       max_age=3600
     )
  
  2. Bloquear puertos expuestos:
     - Cerrar 5001, 8000, cambiar a localhost-only:
       ports:
         - "127.0.0.1:5001:5001"  # Solo localhost
         - "127.0.0.1:8000:8000"
     - Nginx en 80/443 es el único puerto público
  
  3. HTTPS/TLS:
     - Generar certifi cado autofirmado (testing):
       openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
     - Nginx config:
       ssl_certificate /etc/nginx/certs/cert.pem
       ssl_certificate_key /etc/nginx/certs/key.pem
       ssl_protocols TLSv1.2 TLSv1.3
  
  4. Autenticación básica (temporal):
     nginx auth_basic con htpasswd:
       auth_basic "MLOps Portal"
       auth_basic_user_file /etc/nginx/.htpasswd
     O: OAuth2 proxy (mejor)

Corto plazo (P1):
  1. Autenticación robusta:
     - MLFlow: Modificar para usar OIDC o LDAP
     - O: Nginx OAuth2 proxy (Keycloak, Okta, Google)
     - Requerir token JWT en header Authorization
  
  2. Rate limiting:
     - Nginx: limit_req (10 req/s por IP)
     - API: FastAPI SlowAPIMiddleware
       limiter.limit("100/minute")(app.post("/predict"))
  
  3. WAF (Web Application Firewall):
     - ModSecurity en Nginx
     - OWASP top 10 rules
  
  4. Firewall de host:
     - UFW (Linux):
       ufw default deny incoming
       ufw allow from 192.168.1.0/24 to any port 80
       ufw allow from 192.168.1.0/24 to any port 443
     - O: Cloud security groups (AWS SG, Azure NSG)

Largo plazo (P2):
  1. mTLS (mutual TLS):
     - Certificados para cliente y servidor
     - Verificación bidireccional
  
  2. API Gateway (Kong, Tyk):
     - Autenticación centralizada
     - Rate limiting, throttling
     - Request/response logging
     - Plugin ecosystem
  
  3. Infrastructure as Code:
     - Terraform/CloudFormation para reproducibilidad
     - Secrets en AWS Secrets Manager o Vault
     - VPC con subnets públicas/privadas
  
  4. DDoS Protection:
     - Cloudflare, AWS Shield
     - CDN caching para /health checks
```

---

### 5.2 Modelo en Producción

**Actual:**
```python
# inference-api/app.py
@app.get("/predict")
async def predict(sepal_length: float, sepal_width: float,
                  petal_length: float, petal_width: float):
    model = mlflow.sklearn.load_model(f"models:/iris-classifier/Production")
    prediction = model.predict([[sepal_length, ...]])
    return {"prediction": prediction[0]}
```

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| P19 | Sin rate limiting por usuario | Medio | Mismo límite para todos |
| P20 | Sin request validation | Medio | Valores float no validados (Inf, -Inf, NaN) |
| P21 | Sin response signing | Bajo | Cliente no puede verificar que respuesta es auténtica |
| P22 | Sin caching | Bajo | Mismo request genera multiple inference |
| P23 | Model switching sin validación | Medio | /reload puede cargar cualquier modelo |
| P24 | Sin versioning de API | Bajo | Cambios breaking sin versión |
| P25 | Error messages verbose | Bajo | Revela internals (paths, stack trace) |

**Verificaciones:**

```python
# Sin validación de input:
curl http://localhost:8000/predict?sepal_length=Infinity
# Podría causar comportamiento impredecible

# Sin rate limiting por usuario:
curl http://localhost:8000/predict  # Rate limit: 100/min (global)
# Mismo para todos los usuarios

# Sin signing:
curl http://localhost:8000/predict
# Response: {"prediction": 0}
# Cliente no sabe si es respuesta legítima

# /reload sin validación:
curl -X POST http://localhost:8000/reload?model_uri=malicious-model
# Podría cargar modelo no autorizado (si MODEL_URI fuera parametrizable)

# Error messages:
curl http://localhost:8000/predict?invalid=param
# Response: {"detail":"Extra inputs are not permitted"}
# Revela estructura de API
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  1. Validación de input:
     from pydantic import BaseModel, validator
     class PredictRequest(BaseModel):
         sepal_length: float
         sepal_width: float
         petal_length: float
         petal_width: float
         
         @validator('*')
         def validate_finite(cls, v):
             if not math.isfinite(v):
                 raise ValueError("Value must be finite number")
             if v < 0 or v > 10:  # Rango esperado
                 raise ValueError("Value out of range")
             return v
  
  2. Rate limiting por usuario:
     from slowapi import Limiter
     limiter = Limiter(key_func=get_remote_address)
     
     @app.post("/predict")
     @limiter.limit("100/minute")
     async def predict(request: PredictRequest):
         ...
  
  3. Sanitizar error messages:
     try:
         result = model.predict(...)
     except Exception as e:
         logger.error(f"Prediction error: {e}")
         return {"error": "Prediction failed"}  # No stack trace al cliente
  
  4. Model validation on load:
     def load_model(stage):
         model_uri = f"models:/iris-classifier/{stage}"
         model = mlflow.sklearn.load_model(model_uri)
         # Verificar que es el modelo esperado
         assert isinstance(model, RandomForestClassifier)
         assert model.n_estimators == 200
         return model

Corto plazo (P1):
  1. Response signing:
     import hmac, hashlib
     SIGNING_KEY = os.getenv("API_SIGNING_KEY")
     
     def sign_response(response_dict):
         payload = json.dumps(response_dict).encode()
         signature = hmac.new(SIGNING_KEY, payload, hashlib.sha256).hexdigest()
         return {**response_dict, "signature": signature}
  
  2. Request/response logging:
     from pythonjsonlogger import jsonlogger
     logger.info({
       "timestamp": datetime.now().isoformat(),
       "endpoint": "/predict",
       "request": {"features": [5.1, 3.5, 1.4, 0.2]},
       "response": {"prediction": 0},
       "latency_ms": 45
     })
  
  3. Model version in response:
     return {
       "prediction": 0,
       "model_version": 4,
       "model_stage": "Production",
       "timestamp": datetime.now().isoformat()
     }
  
  4. API versioning:
     /api/v1/predict  # Current
     /api/v2/predict  # Future (backward compatible)
     
     Sunset old versions: 
     /api/v0/predict → 410 Gone (deprecated)

Largo plazo (P2):
  1. Shadow model testing:
     Correr versión nueva en paralelo, log predictions pero no retornar
     Comparar con versión vieja para detección de drift
  
  2. Canary deployments:
     - 90% traffic a v3 (Production)
     - 10% traffic a v4 (Canary)
     - Si v4 error rate > v3, rollback automático
  
  3. Feature flags:
     enable_v4_model = feature_flags.is_enabled("iris_v4", user_id)
     model = models_v4 if enable_v4_model else models_v3
  
  4. Model explanability:
     Retornar SHAP values o feature importance con prediction
     {"prediction": 0, "explanation": {"sepal_length": 0.45, ...}}
```

---

### 5.3 Monitoreo y Alertas

**Actual:**
```python
# drift-detector/detector.py
# Detecta data drift y performance drift, triggerear retraining
# Logs en stdout (Docker)
```

**Vulnerabilidades:**

| # | Hallazgo | Severidad | Descripción |
|---|----------|-----------|-------------|
| P26 | Sin alertas activas | Alto | Drift detectado pero nadie notificado |
| P27 | Sin SLO/SLI | Medio | No hay acuerdo de disponibilidad |
| P28 | Sin health checks | Bajo | Cómo saber si modelo está alive? |
| P29 | Sin canary testing | Medio | Sin verificación continua post-deploy |
| P30 | Monitoreo silencioso | Medio | Si retraining falla, sin notificación |

**Verificaciones:**

```bash
# Drift detectado:
docker logs mlops-stack_drift_1 | grep "DRIFT"
  # Aparece en stdout pero ¿quién lo ve?

# Sin alertas:
grep -r "alert\|email\|slack\|pagerduty" drift-detector/
  # No hay integración de alertas

# Sin SLO:
# Documentación: No menciona availability % ni latency p99

# Health check ausente:
curl http://localhost:8000/health
  # Existe, devuelve 200
  # Pero: No verifica que modelo esté cargado correctamente
```

**Recomendaciones:**

```yaml
Inmediato (P0):
  1. Alertas de drift:
     if drift_detected:
         logger.error("DATA DRIFT DETECTED in 2 consecutive windows")
         send_alert({
             "level": "ERROR",
             "message": "Retrain triggered",
             "p_value": drift_p_value,
             "threshold": 0.05
         })
  
  2. Health check mejorado:
     @app.get("/health")
     async def health():
         try:
             model = mlflow.sklearn.load_model(...)
             test_prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
             return {"status": "ok", "model_loaded": True}
         except:
             return {"status": "error"}, 503
  
  3. SLO/SLI básicos:
     # Documentar en README:
     SLO_AVAILABILITY: 99.5%
     SLI_LATENCY_P99: 500ms
     SLI_ERROR_RATE: <0.1%
     
     # Medir:
     error_count = sum(http_status != 200)
     total_requests = sum(all http_requests)
     error_rate = error_count / total_requests
     assert error_rate < 0.001  # <0.1%

Corto plazo (P1):
  1. Alertas activas (Slack/Email):
     import requests
     
     def send_slack_alert(message):
         webhook = os.getenv("SLACK_WEBHOOK")
         requests.post(webhook, json={"text": message})
     
     # O: Usar Prometheus + AlertManager
     groups:
       - name: ml-pipeline
         rules:
           - alert: DriftDetected
             expr: drift_p_value < 0.05
             annotations:
               summary: "Data drift detected"
               description: "p-value: {{ $value }}"
  
  2. Dashboard (Grafana/Datadog):
     - Latency (p50, p95, p99)
     - Error rate by endpoint
     - Model accuracy (training vs production)
     - Drift signals
     - Data volume (requests/min)
  
  3. Canary testing:
     POST /health → Verificar modelo v4 en paralelo
     assert v4_output == v3_output  # Sanity check

Largo plazo (P2):
  1. SIEM (Security Information Event Management):
     - Splunk, ELK, Datadog
     - Buscar anomalías: accuracy drop >5%, latency spike, error surge
  
  2. Automated remediation:
     - Drift detectado → Auto-retrain
     - Si retraining falla → Rollback a versión anterior
     - Enviar alert a oncall team
  
  3. Incident response runbook:
     - Drift triggered retraining, qué hacer?
     - Model accuracy dropped, rollback procedures
     - How to investigate: query logs, compare models
  
  4. Model validation continuous:
     - Cada hour: Sample inference request, verificar output
     - Cada día: Full test suite en production data (shadow)
```

---

## Matriz de Riesgos

| ID | Etapa | Aspecto | Hallazgo | Sev | Imp | Status | Remediación |
|----|-------|---------|----------|-----|-----|--------|-------------|
| A1 | Deploy | Código | Credenciales Git | A | M | ✓ OK | Usando GitHub Secrets |
| A2 | Deploy | Imágenes | Sin firma/scan | M | M | ❌ NO | Trivy + cosign en CI |
| A3 | Deploy | Compose | Permisos 777 | **A** | **A** | ❌ NO | chmod 700 /mlflow |
| A4 | Deploy | Compose | Sin limits | M | M | ❌ NO | memory/cpu limits |
| T1 | Train | Datos | Sin validación | M | L | ⚠ Parcial | Validación de shape |
| T5 | Train | Almacen | Sin encripción | **A** | M | ❌ NO | LUKS o S3 SSE |
| T6 | Train | Almacen | Sin firma | M | M | ❌ NO | SHA256 checksum |
| T7 | Train | Almacen | Permisos 777 | **A** | **A** | ❌ NO | chmod 755 |
| T16 | Test | Tests | Sin cobertura | M | L | ⚠ Parcial | pytest --cov-fail-under=80 |
| T21 | Test | Aislamiento | Tests modifican prod | **A** | **A** | ❌ NO | Volumen separado mlflow-test |
| P1 | Registry | RBAC | Sin autenticación | **A** | **A** | ❌ NO | OIDC/LDAP en MLFlow |
| P2 | Registry | Approval | Sin aprobación manual | M | M | ❌ NO | Approval gate workflow |
| P11 | Prod | Acceso | Sin autenticación | **CRÍTICO** | **A** | ❌ NO | OAuth2 proxy (Keycloak) |
| P12 | Prod | API | CORS permisivo | **CRÍTICO** | **A** | ❌ NO | CORS específico (trusted domains) |
| P13 | Prod | TLS | Sin HTTPS | **CRÍTICO** | **A** | ❌ NO | HTTPS con Let's Encrypt |
| P15 | Prod | Firewall | Puertos abiertos | **A** | **A** | ❌ NO | UFW: deny incoming, allow 80/443 |
| P26 | Prod | Alertas | Sin notificaciones | **A** | **A** | ❌ NO | Slack/PagerDuty integration |

**Leyenda:** Sev=Severidad, Imp=Impacto, A=Alto, M=Medio, L=Bajo

---

## Plan de Remediación

### Prioridad 0 (Crítico - 1-2 semanas)

```yaml
P0-1: Autenticación en producción
  Task: Desplegar Keycloak en docker-compose
  Owner: DevOps
  PR: Nginx OAuth2 proxy, require tokens en /mlflow y /api

P0-2: CORS restricción
  Task: Cambiar allow_origins=["*"] a ["https://trusted-domain.com"]
  Owner: Backend
  PR: inference-api/app.py + tests

P0-3: HTTPS certificado
  Task: Generar/configurar certificado (Let's Encrypt o autofirmado)
  Owner: DevOps
  PR: docker-compose nginx + cert volume

P0-4: Firewall puerto
  Task: Cambiar ports a 127.0.0.1:8000:8000 (localhost-only)
  Owner: DevOps
  PR: docker-compose + UFW ruleset

P0-5: Permisos volumen
  Task: chmod 700 /mlflow/mlflow.db, chmod 755 /mlflow/artifacts
  Owner: DevOps
  PR: entrypoint.sh, setup_server.sh
```

### Prioridad 1 (Alto - 2-4 semanas)

```yaml
P1-1: Encriptación modelo
  Task: Migrar pickle → ONNX + checksum
  Owner: ML Eng
  PR: model-trainer/train.py, inference-api/app.py

P1-2: Test aislamiento
  Task: Volumen mlflow-test separado
  Owner: QA/DevOps
  PR: docker-compose, test-runner/Dockerfile

P1-3: Image scanning
  Task: Trivy + cosign en CI
  Owner: DevOps
  PR: .github/workflows/ci.yml

P1-4: Quality gates coverage
  Task: pytest --cov enforcement
  Owner: QA
  PR: tests/run_tests.py

P1-5: Alertas
  Task: Slack/PagerDuty webhook
  Owner: DevOps
  PR: drift-detector integration
```

### Prioridad 2 (Medio - 1-3 meses)

```yaml
P2-1: RBAC en MLFlow
  Task: PostgreSQL + OIDC + roles (admin/trainer/approver)
  Owner: DevOps
  PR: docker-compose, MLFlow config

P2-2: Model signing
  Task: PGP/X.509 signatures para modelos
  Owner: ML Eng
  PR: model-trainer, inference-api

P2-3: Data versioning
  Task: DVC integration para dataset lineage
  Owner: ML Eng
  PR: dvc.yaml, setup_server.sh

P2-4: Canary deployment
  Task: 90/10 traffic split, automated rollback
  Owner: DevOps
  PR: Kubernetes migration (helm chart)

P2-5: SIEM
  Task: ELK Stack o Datadog para log aggregation
  Owner: DevOps
  PR: logstash, filebeat containers
```

---

## Referencias y Mejores Prácticas

### Estándares de Seguridad Aplicables
- **OWASP Top 10 (2021):** A01:2021 – Broken Access Control, A02:2021 – Cryptographic Failures
- **NIST SP 800-53:** AC-2 (Account Management), AU-2 (Audit Events), SC-7 (Boundary Protection)
- **ISO/IEC 27001:** 7.1 (Access Control), 8.2 (Asset Management)
- **MLOps Security Framework:** IEEE 1012, MITRE ATLAS, CISA ML Security
- **Container Security:** CIS Docker Benchmark, Kubernetes Pod Security Standards
- **Data Privacy:** GDPR (Art. 32), CCPA, LGPD

### Herramientas Recomendadas
| Aspecto | Herramienta | Propósito |
|--------|-----------|----------|
| Image Scan | Trivy, Grype | Vulnerabilidades en layers |
| SAST | Bandit, Semgrep | Code vulnerabilities |
| SCA | Safety, pip-audit | Dependency vulnerabilities |
| DAST | OWASP ZAP, Burp | API testing |
| Secrets | TruffleHog, GitGuardian | Credential detection |
| Access | Keycloak, HashiCorp Boundary | Identity & Access Mgmt |
| Secrets Mgmt | HashiCorp Vault, AWS Secrets Mgr | Secret storage |
| Encryption | LUKS, AWS KMS | Data at rest/transit |
| Monitoring | Prometheus + Grafana, Datadog | Metrics & Alerting |
| Logging | ELK Stack, Splunk, Datadog | Log aggregation |
| Model Registry | MLFlow + PostgreSQL, Weights&Biases | Model versioning |

---

## Conclusión

El pipeline actual tiene **fortalezas significativas en integridad de datos** (4 niveles de tests, quality gates automáticos, drift detection), pero **carece de controles de acceso y confidencialidad** críticos para producción.

**Acciones inmediatas (semana 1):**
1. ✓ Autenticación OAuth2 en MLFlow y API
2. ✓ HTTPS en todos los endpoints
3. ✓ Cambiar permisos de archivos (chmod 700/755)
4. ✓ Bloquear puertos (localhost-only)

**Esto reduciría el riesgo de 6/10 a 8/10.**

---

**Preparado por:** Security Review Team  
**Fecha:** 2026-06-11  
**Próxima revisión:** 2026-07-11
