# Guía: Performance Drift Detection

## Introducción

Este módulo detecta degradación en el desempeño de los modelos de regresión HRI en producción,
monitoreando 5 métricas continuas con tests estadísticos.

**Métricas monitoreadas:**
- **R²**: Varianza explicada (↓ malo)
- **RMSE**: Error cuadrático medio (↑ malo)
- **MAE**: Error absoluto medio (↑ malo)
- **sMAPE**: Error porcentual simétrico (↑ malo, robusto a valores cercanos a cero)
- **Max Error**: Peor predicción individual (↑ malo)

## Cómo Funciona

### 1. Baseline (Referencia)

Se calcula al iniciar el servicio drift-detector usando las métricas del modelo registrado en MLflow:

```python
baseline_metrics = {
    "r2":        0.92,
    "rmse":      10.5,
    "mae":        7.8,
    "smape":      4.5,
    "max_error": 28.0,
}
```

### 2. Detección Deslizante

Cada ventana de tiempo (30 predicciones), se calculan las métricas actuales y se comparan con baseline:

```
┌─ Ventana 1: r2=0.91, rmse=11.0, mae=8.1  ✓
├─ Ventana 2: r2=0.85, rmse=16.0, mae=12.0 ⚠ (2 drifts)
├─ Ventana 3: r2=0.78, rmse=22.0, mae=17.0 ⚠ (3 drifts)
└─ TRIGGER: Retraining (2 ventanas consecutivas con drift)
```

### 3. Métodos Estadísticos

#### a) Effect Size (Cambio Absoluto)
```
Si |actual - baseline| > 0.05 → Potencial drift
Ejemplo: R² cae de 0.92 a 0.85 → cambio de 0.07 → DRIFT
```

#### b) T-Test (Significancia)
```
Compara últimas 5 observaciones contra baseline
H0: media(últimas_5) = baseline
Si p_value < 0.05 AND cambio > 0.02 → Drift estadísticamente significativo
```

#### c) EWMA (Tendencia Sostenida)
```
EWMA_actual = 0.3 * valor_actual + 0.7 * EWMA_anterior
Si |EWMA - baseline| > 0.07 → Divergencia sostenida
Detecta problemas que no son puntuales sino tendencias
```

**Drift declarado** cuando ≥ 2 métricas disparan cualquier test.

## Configuración

### Variables de Entorno

```bash
ENABLE_PERFORMANCE_DRIFT=true
PERF_DRIFT_EFFECT_SIZE=0.05       # Cambio mínimo (5%)
PERF_DRIFT_P_VALUE=0.05           # p-value para t-test
PERF_DRIFT_CONSECUTIVE=2          # Ventanas consecutivas para trigger
```

### En docker-compose.yml

```yaml
drift-detector:
  environment:
    ENABLE_PERFORMANCE_DRIFT: "true"
    PERF_DRIFT_EFFECT_SIZE: "0.05"
    PERF_DRIFT_P_VALUE: "0.05"
    PERF_DRIFT_CONSECUTIVE: "2"
```

## Interpretación de Resultados

### Logs del Drift Detector

```
--- Performance Drift Cycle 42 ---
Baseline: {'r2': 0.92, 'rmse': 10.5, 'mae': 7.8, 'smape': 4.5, 'max_error': 28.0}
Current:  {'r2': 0.78, 'rmse': 22.0, 'mae': 17.0, 'smape': 14.2, 'max_error': 61.0}

  Performance Status: DRIFT ⚠  consecutive=1/2
  Drifted metrics: r2 (effect_size=0.14), rmse (effect_size=11.5), mae (effect_size=9.2)
```

### Métricas en MLFlow

Experimento: `performance-drift-monitoring`

Cada ventana registra:
```
current_r2:           0.78
baseline_r2:          0.92
change_pct_r2:       -15.22
current_rmse:         22.0
perf_drift_detected:  1
perf_consec_drifts:   1
```

## Escenarios Reales

### Escenario 1: Degradación Gradual
El modelo ya no refleja condiciones de cosecha actuales:
- **Síntomas**: R² cae de 0.92 → 0.87 → 0.80 en ventanas sucesivas
- **Detección**: EWMA detecta tendencia sostenida después de varios ciclos
- **Acción**: Retraining automático disparado

### Escenario 2: Ruido Puntual
Una ventana tiene predicciones erróneas por datos corruptos:
- **Síntomas**: RMSE sube a 45 en 1 ventana, luego vuelve a 11
- **Detección**: Effect size dispara, pero no hay tendencia (EWMA se recupera)
- **Acción**: Sin trigger (solo 1 ventana con drift, necesita 2 consecutivas)

### Escenario 3: Shift de Distribución
Nuevo tipo de actividad no visto en entrenamiento:
- **Síntomas**: R² global OK (0.88), pero sMAPE sube a 35% y Max Error a 120
- **Detección**: Si ≥ 2 métricas están fuera de threshold
- **Acción**: Performance drift trigger + retraining

## Pruebas Automatizadas

```bash
# Tests unitarios
pytest tests/test_performance_drift.py -v

# Tests de integración
pytest tests/test_performance_drift_integration.py -v -s

# Suite QA completa
docker compose run --rm test-runner
```

## Ajuste de Umbrales

### Más Sensible (más alertas)
```bash
PERF_DRIFT_EFFECT_SIZE=0.02    # Detecta cambios pequeños
PERF_DRIFT_P_VALUE=0.10        # p-value más lenient
PERF_DRIFT_CONSECUTIVE=1       # Trigger en 1 ventana
```

### Más Conservador (menos alertas)
```bash
PERF_DRIFT_EFFECT_SIZE=0.10    # Solo cambios grandes
PERF_DRIFT_P_VALUE=0.01        # p-value muy estricto
PERF_DRIFT_CONSECUTIVE=3       # Trigger en 3 ventanas
```

## Por qué sMAPE en lugar de MAPE

Algunos targets del dataset HRI tienen valores cercanos a 0
(`TotalWorkload` en escenario WithRobot puede ser 0.0).
MAPE produce `inf` cuando `y_true → 0`. sMAPE usa denominador simétrico:

```
sMAPE = mean(|y_pred - y_true| / ((|y_true| + |y_pred|) / 2)) × 100
```

Rango acotado: 0–200%. Gate de producción: `sMAPE < 20%`.

## Troubleshooting

### Muchos falsos positivos
**Causa**: Umbrales muy sensibles  
**Solución**: Aumentar `PERF_DRIFT_EFFECT_SIZE` a 0.08–0.10

### Nunca detecta drift real
**Causa**: Umbrales muy altos  
**Solución**: Reducir `PERF_DRIFT_EFFECT_SIZE` a 0.02–0.03

### Métricas no aparecen en MLFlow
**Causa**: MLFlow no estaba listo al iniciar drift-detector  
**Solución**: Aumentar `retries` en `wait_for()` en `detector.py`

## Referencias

- **Effect Size**: Estándar en estadística (Cohen's d)
- **T-test**: Prueba si hay diferencia significativa de medias
- **EWMA**: Control de calidad industrial (SPC)
- **sMAPE**: Kim & Kim (2016), robusto a denominadores cercanos a cero
- **Dataset**: Vasconez & Auat Cheein (2022), *Biosystems Engineering* Vol. 223

---

**Última actualización**: 2026-06-12  
**Versión**: 2.0.0
