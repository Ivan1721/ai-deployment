# Guión de demostración — desde el computador recién encendido

Tiempo total de preparación: ~15 min. Hacerlo **antes** de que empiece la presentación.

---

## FASE 0 — Encender y levantar (≈5 min)

1. Enciende el PC e inicia sesión en Windows.
2. Abre **Docker Desktop** (menú Inicio) y espera a que el motor quede corriendo
   (ballena estable / "Engine running", 1–2 min). Minimízalo, no lo cierres.
3. Abre la terminal **Ubuntu (WSL)** y ve al proyecto:

```bash
cd ~/ai-deployment/mlops-stack
docker compose ps
```

4. Dos casos según lo que muestre:
   - **Aparecen los 9 servicios "Up"** → los contenedores se auto-levantaron
     (`restart: unless-stopped`). Sigue a la Fase 1.
   - **Lista vacía** (quedó apagado con `docker compose down`) → levántalos:

```bash
docker compose up -d --no-build postgres mlflow inference-api drift-detector keycloak oauth2-proxy nginx prometheus grafana
```

> ⚠️ Siempre con `--no-build` y la lista explícita: un `up -d` a secas
> dispararía el `model-trainer` (reentrenamiento) y el `test-runner`.

## FASE 1 — Verificación (≈2 min)

```bash
# Esperar a que los 8 modelos estén cargados:
until curl -s http://localhost:8000/health | grep -q '"models_loaded":8'; do sleep 5; echo esperando...; done && echo "OK: 8 modelos"

# Calentar Grafana con tráfico real (los paneles de latencia tendrán datos):
for i in $(seq 1 20); do curl -sk -X POST https://localhost/api/predict -H "Content-Type: application/json" -H "X-API-Key: $(grep ^API_KEY .env | cut -d= -f2)" -d "{\"scenario\":$((i%2)),\"workers\":$((RANDOM%12+1)),\"crop_row\":2,\"rand_pos\":0,\"activity\":\"harv_ground\"}" -o /dev/null; done && echo "OK: 20 predicciones"
```

## FASE 2 — Credenciales y pestañas (≈5 min)

```bash
grep ^API_KEY .env | cut -d= -f2 | clip.exe      # API key → portapapeles de Windows
grep ^GRAFANA_ADMIN_PASSWORD .env                 # anotar esta contraseña
```

Abre estas 4 pestañas **en este orden** y déjalas listas (acepta el certificado
autofirmado con "Avanzado → Continuar" AHORA, para que no aparezca la
advertencia en vivo):

| # | URL | Preparación |
|---|-----|-------------|
| 1 | http://localhost:8000/docs | Botón **Authorize** (candado) → pegar la API key del portapapeles → Close |
| 2 | https://localhost/mlflow/ | Login SSO: `mlops-user` / `mlops123` → dejar abierto el experimento **hri-harvesting** |
| 3 | https://localhost/grafana/ | Login: `admin` / (contraseña anotada) → abrir el dashboard **MLOps** |
| 4 | http://localhost:9090/targets | Sin login |

Deja **pegados (sin ejecutar)** en la terminal:

```bash
# Paso 5 de la demo — DVC doble remoto:
~/.venvs/dvc-paper/bin/dvc remote list && ~/.venvs/dvc-paper/bin/dvc pull -r gdriveremote

# Paso 6 (opcional) — pirámide QA en vivo (~2 min):
MLIP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlflow-server); APIIP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' inference-api); docker compose run --rm -e MLFLOW_TRACKING_URI=http://$MLIP:5001 -e INFERENCE_API_URL=http://$APIIP:8000 test-runner
```

---

## FASE 3 — La demo en vivo (lámina 15, ≈4 min)

1. **Swagger** (1 min) — `POST /predict` → *Try it out* → cambiar body a:
   `{"scenario":1,"workers":7,"crop_row":2,"rand_pos":0,"activity":"harv_ground"}`
   → *Execute*. **Narrar**: "7 trabajadores nunca se simuló — el modelo
   interpola; responde los 4 targets en una llamada, con TLS y API key, en
   milisegundos".
2. **MLflow** (1 min) — pestaña *Experiments* → runs con métricas `ho_r2`;
   pestaña *Models* → 8 modelos con **miles de versiones**. **Narrar**: "cada
   versión es un ciclo champion/challenger automático del detector de drift".
3. **Grafana** (30 s) — panel de latencia p50/p95/p99 y tasa de requests (con
   el tráfico del calentamiento) + panel de drift.
4. **Prometheus** (15 s) — 2 targets `up` (inference-api y drift-detector).
5. **DVC** (30 s) — ejecutar el comando preparado → "Everything is up to
   date". **Narrar**: "doble remoto: local para CI/CD sin credenciales, Google
   Drive para sincronizar desde cualquier máquina". (Si hay tiempo: mostrar la
   carpeta en drive.google.com.)
6. **Pirámide QA** (opcional, 2 min) — ejecutar el segundo comando; mientras
   corre, narrar los 6 niveles; el resumen final muestra los 3 hallazgos
   conocidos. **Narrar**: "el sistema se audita a sí mismo: estas fallas son
   los hallazgos documentados del informe, no sorpresas".

## PLAN B

Si algo falla en vivo → lámina 16 (capturas de respaldo) y la carpeta
`Escritorio\capturas_ppt\` con las 10 imágenes.

## Qué NO hacer / problemas conocidos

- **Nunca** `docker compose down -v` (borra los modelos y el realm de Keycloak).
- **No reconstruir imágenes** ese día (`docker compose build` puede fallar por
  el credential helper de Docker Desktop; no hace falta ningún build).
- `oauth2-proxy` puede aparecer "(unhealthy)": es cosmético, el SSO funciona.
- Si nginx/oauth2-proxy entran en crash loop (solo pasa si se borraron
  volúmenes): receta del realm en `mlops-stack/CLAUDE.md` → Troubleshooting.
- Si Docker Desktop no arranca: cerrarlo, `wsl --shutdown` en PowerShell,
  y abrir Docker Desktop de nuevo.
