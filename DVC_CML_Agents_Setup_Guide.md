# Guía de configuración del entorno MLOps con Codex, DVC, Google Drive, CML y GitHub Actions

**Proyecto base:** `cml-example-base`  
**Objetivo:** configurar un nuevo entorno de trabajo para que un agente como Codex pueda trabajar sobre el repositorio, recuperar datos versionados con DVC desde Google Drive, ejecutar entrenamiento y validar el pipeline de CML en GitHub Actions.

> **Nota de seguridad:** este documento usa placeholders como `<CLIENT_ID>`, `<CLIENT_SECRET>`, `<FOLDER_ID>` y `<SERVICE_ACCOUNT_EMAIL>`. No se deben guardar claves reales, archivos JSON privados ni secretos directamente en el repositorio.

---

## 1. Estructura esperada del repositorio

El repositorio debe tener una estructura similar a:

```text
cml-example-base/
│
├── data.dvc
├── train.py
├── requirements.txt
├── metrics.txt
├── plot.png
│
├── .dvc/
│   ├── config
│   └── .gitignore
│
└── .github/
    └── workflows/
        └── cml.yaml
```

La carpeta `data/` no debe estar versionada directamente por Git. Debe ser recuperada mediante DVC con:

```bash
dvc pull
```

---

## 2. Requisitos del entorno local

Instalar:

- Git
- Python 3.10 o superior
- DVC con soporte para Google Drive
- Dependencias del proyecto
- Acceso al repositorio GitHub
- Acceso al Google Drive remote usado por DVC

Comandos base:

```bash
python -m pip install --upgrade pip
pip install "dvc[gdrive]"
pip install -r requirements.txt
```

Verificar instalación:

```bash
git --version
python --version
dvc --version
```

---

## 3. Clonar el repositorio

```bash
git clone https://github.com/<USUARIO>/<REPOSITORIO>.git
cd <REPOSITORIO>
```

Ejemplo:

```bash
git clone https://github.com/Ivan1721/cml-example-base.git
cd cml-example-base
```

Verificar estado:

```bash
git status
```

---

## 4. Configuración de DVC

### 4.1 Verificar que DVC ya esté inicializado

```bash
ls .dvc
```

Debe existir al menos:

```text
.dvc/config
.dvc/.gitignore
```

También puedes verificar:

```bash
dvc status
```

---

### 4.2 Si el repositorio no tiene DVC inicializado

Solo usar si partes desde cero:

```bash
dvc init
git add .dvc/.gitignore .dvc/config
git commit -m "Initialize DVC"
```

---

## 5. Migrar datos desde Git hacia DVC

Si `data/` está siendo seguido por Git:

```bash
git rm -r --cached data
git commit -m "stop tracking data with Git"
```

Luego agregar los datos a DVC:

```bash
dvc add data/
git add data.dvc .gitignore
git commit -m "track data with DVC"
```

Subir los datos al remote:

```bash
dvc push
```

---

## 6. Configurar Google Drive como remote de DVC

Crear una carpeta en Google Drive y obtener su ID desde la URL.

Ejemplo:

```text
https://drive.google.com/drive/folders/<FOLDER_ID>
```

Configurar el remote:

```bash
dvc remote add --default gdriveremote gdrive://<FOLDER_ID>
```

Verificar:

```bash
dvc remote list
```

Salida esperada:

```text
gdriveremote  gdrive://<FOLDER_ID> (default)
```

Guardar configuración versionable:

```bash
git add .dvc/config
git commit -m "configure DVC Google Drive remote"
```

---

## 7. Configuración de Google Cloud Console

### 7.1 Crear proyecto

Ir a:

```text
https://console.cloud.google.com/
```

Crear o seleccionar un proyecto, por ejemplo:

```text
cm-geonosis
```

---

### 7.2 Habilitar Google Drive API

Ruta:

```text
APIs & Services > Library > Google Drive API > Enable
```

---

### 7.3 Crear OAuth Client ID para pruebas locales

Ruta:

```text
APIs & Services > Credentials > Create Credentials > OAuth client ID
```

Tipo:

```text
Desktop application
```

Nombre sugerido:

```text
cm-geonosis
```

Guardar:

```text
CLIENT_ID
CLIENT_SECRET
```

Configurar localmente con DVC usando `--local`:

```bash
dvc remote modify --local gdriveremote gdrive_client_id "<CLIENT_ID>"
dvc remote modify --local gdriveremote gdrive_client_secret "<CLIENT_SECRET>"
```

Esto guarda las credenciales en:

```text
.dvc/config.local
```

Ese archivo no debe subirse a Git.

---

### 7.4 Crear Service Account para GitHub Actions

Ruta:

```text
APIs & Services > Credentials > Create Credentials > Service Account
```

Nombre sugerido:

```text
dvc-gdrive
```

Luego generar una clave:

```text
Service Account > Keys > Add Key > Create new key > JSON
```

Esto descarga un archivo `.json`.

---

### 7.5 Compartir carpeta de Google Drive

Abrir la carpeta de Google Drive usada como remote de DVC y compartirla con el email de la service account.

El email aparece dentro del JSON:

```json
"client_email": "<SERVICE_ACCOUNT_EMAIL>"
```

Permiso requerido:

```text
Editor
```

---

## 8. Configurar GitHub Secrets

Ir al repositorio:

```text
Settings > Secrets and variables > Actions > New repository secret
```

Crear el secret:

```text
GDRIVE_CREDENTIALS_DATA
```

Como valor, pegar el contenido completo del JSON de la service account.

Opcionalmente, si se mantienen credenciales OAuth para pruebas locales:

```text
GDRIVE_CLIENT_ID
GDRIVE_CLIENT_SECRET
```

> Recomendación: para GitHub Actions usar principalmente `GDRIVE_CREDENTIALS_DATA` con Service Account.

---

## 9. Archivo `requirements.txt`

Ejemplo mínimo:

```text
numpy
matplotlib
scikit-learn
dvc[gdrive]
```

Si el workflow instala DVC por separado, también se puede dejar `dvc[gdrive]` solo en el workflow.

---

## 10. Workflow de GitHub Actions con CML

Archivo:

```text
.github/workflows/cml.yaml
```

Contenido recomendado:

```yaml
name: model-training

on:
  push:
    branches:
      - main

jobs:
  run:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Setup CML
        uses: iterative/setup-cml@v2

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install "dvc[gdrive]"
          pip install -r requirements.txt

      - name: Configure DVC Google Drive service account
        env:
          GDRIVE_CREDENTIALS_DATA: ${{ secrets.GDRIVE_CREDENTIALS_DATA }}
        run: |
          mkdir -p .dvc/tmp
          echo "$GDRIVE_CREDENTIALS_DATA" > .dvc/tmp/gdrive-sa.json
          dvc remote modify --local gdriveremote gdrive_use_service_account true
          dvc remote modify --local gdriveremote gdrive_service_account_json_file_path .dvc/tmp/gdrive-sa.json

      - name: Pull data from DVC remote
        run: |
          dvc pull -v

      - name: Train model
        run: |
          python train.py

      - name: Create CML report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          echo "## Model training report" > report.md
          echo "" >> report.md
          echo "### Metrics" >> report.md
          cat metrics.txt >> report.md
          echo "" >> report.md
          echo "### Confusion Matrix" >> report.md
          echo "![](./plot.png)" >> report.md
          cml comment create report.md
```

---

## 11. Verificar el workflow

Hacer un commit y push:

```bash
git add .
git commit -m "test CML DVC workflow"
git push
```

Luego revisar:

```text
GitHub > Actions > model-training
```

El workflow debe completar estos pasos:

```text
Checkout repository
Setup Python
Setup CML
Install dependencies
Configure DVC Google Drive service account
Pull data from DVC remote
Train model
Create CML report
```

---

## 12. Validaciones locales

### 12.1 Verificar remote

```bash
dvc remote list
```

### 12.2 Verificar estado

```bash
dvc status
```

### 12.3 Descargar datos

```bash
dvc pull
```

### 12.4 Entrenar localmente

```bash
python train.py
```

Debe generar:

```text
metrics.txt
plot.png
```

---

## 13. Uso del repositorio con Codex como agente

Cuando Codex trabaje sobre el repositorio, debe seguir este flujo:

### 13.1 Inspección inicial

```bash
pwd
ls
git status
```

### 13.2 Revisar estructura

```bash
ls .github/workflows
ls .dvc
cat .dvc/config
cat requirements.txt
```

### 13.3 Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install "dvc[gdrive]"
pip install -r requirements.txt
```

### 13.4 Recuperar datos

En local, si existe configuración OAuth local:

```bash
dvc pull
```

En GitHub Actions, se usa la service account configurada en secrets.

### 13.5 Ejecutar entrenamiento

```bash
python train.py
```

### 13.6 Revisar salidas

```bash
cat metrics.txt
ls plot.png
```

### 13.7 Preparar cambios

```bash
git status
git add .
git commit -m "update training pipeline"
git push
```

---

## 14. Instrucciones importantes para Codex

Codex debe respetar estas reglas:

1. No subir archivos reales de datos (`data/`) a Git.
2. No subir `.dvc/config.local`.
3. No subir archivos `.json` de Google Cloud.
4. No exponer `CLIENT_SECRET`, `private_key`, `client_email` completo si no es necesario.
5. Si modifica el dataset, debe ejecutar:

```bash
dvc add data/
dvc push
git add data.dvc .gitignore
git commit -m "update DVC data version"
```

6. Si modifica `train.py`, debe ejecutar:

```bash
python train.py
```

7. Si modifica el workflow, debe validar sintaxis YAML antes de hacer push.

---

## 15. Archivos que NO deben subirse a Git

Agregar o verificar en `.gitignore`:

```gitignore
/data
.dvc/config.local
.dvc/tmp/
*.json
.env
__pycache__/
*.pyc
```

DVC normalmente agrega `/data` automáticamente al `.gitignore` al ejecutar:

```bash
dvc add data/
```

---

## 16. Errores comunes y solución

### Error: `dvc: command not found`

Solución:

```bash
pip install "dvc[gdrive]"
```

---

### Error: `output 'data' is already tracked by SCM`

Solución:

```bash
git rm -r --cached data
git commit -m "stop tracking data with Git"
dvc add data/
git add data.dvc .gitignore
git commit -m "track data with DVC"
```

---

### Error: Google Drive `File not found`

Causa probable:

- La service account no tiene acceso a la carpeta.
- El `<FOLDER_ID>` está incorrecto.
- No se compartió la carpeta como Editor.

Solución:

1. Copiar `client_email` desde el JSON.
2. Compartir la carpeta de Google Drive con ese email.
3. Dar permiso Editor.
4. Reintentar:

```bash
dvc pull -v
```

---

### Error: Google bloquea la aplicación

Causa:

- Se está intentando usar OAuth interactivo en GitHub Actions.

Solución:

- Usar Service Account.
- Guardar el JSON en `GDRIVE_CREDENTIALS_DATA`.
- Configurar DVC en el workflow con:

```bash
dvc remote modify --local gdriveremote gdrive_use_service_account true
dvc remote modify --local gdriveremote gdrive_service_account_json_file_path .dvc/tmp/gdrive-sa.json
```

---

### Error: Push rechazado por credenciales

Causa:

- Se intentó subir un secreto al repositorio.

Solución:

1. Revocar o regenerar credenciales.
2. Eliminar credenciales de `.dvc/config`.
3. Usar `--local` o GitHub Secrets.
4. Si quedó en commits, limpiar historial o rehacer commits.

---

## 17. Flujo final recomendado

### Trabajo local

```bash
git pull
dvc pull
python train.py
git status
```

### Si se modifica código

```bash
git add train.py requirements.txt
git commit -m "update model training code"
git push
```

### Si se modifica data

```bash
dvc add data/
dvc push
git add data.dvc .gitignore
git commit -m "update dataset version"
git push
```

### En GitHub Actions

```text
push
↓
checkout
↓
install dependencies
↓
configure DVC service account
↓
dvc pull
↓
python train.py
↓
create CML report
```

---

## 18. Checklist de configuración

- [ ] Repositorio clonado.
- [ ] Python instalado.
- [ ] `requirements.txt` instalado.
- [ ] DVC instalado con soporte Google Drive.
- [ ] `.dvc/config` contiene el remote `gdriveremote`.
- [ ] Carpeta de Google Drive creada.
- [ ] Google Drive API habilitada.
- [ ] Service Account creada.
- [ ] JSON de Service Account descargado.
- [ ] Carpeta de Drive compartida con la Service Account.
- [ ] GitHub Secret `GDRIVE_CREDENTIALS_DATA` creado.
- [ ] Workflow `cml.yaml` configurado.
- [ ] `dvc push` ejecutado localmente.
- [ ] `dvc pull` funciona en GitHub Actions.
- [ ] `python train.py` genera `metrics.txt` y `plot.png`.
- [ ] CML publica reporte en GitHub.
