# Guía: Performance Drift Detection - v4

## Introducción

Este módulo detecta degradación en el desempeño del modelo en producción monitoreando 4 métricas clave de clasificación:
- **Accuracy**: Exactitud general
- **Precision**: Precisión (weighted/macro/micro)
- **Recall**: Exhaustividad (weighted/macro/micro)
- **F1-score**: Balance entre precision y recall (weighted/macro/micro)

## Cómo Funciona

### 1. Baseline (Referencia)
Se calcula al iniciar el servicio drift-detector usando el conjunto de datos original (Iris):
```python
baseline_metrics = {
    'accuracy': 0.96,
    'precision': 0.96,
    'recall': 0.96,
    'f1': 0.96
}
```

### 2. Detección Deslizante
Cada ventana de tiempo (e.g., 30 predicciones), se calculan las métricas actuales y se comparan con baseline:

```
┌─ Ventana 1: acc=0.95, prec=0.95, rec=0.95, f1=0.95 ✓
├─ Ventana 2: acc=0.92, prec=0.90, rec=0.93, f1=0.91 ⚠ (2 drifts)
├─ Ventana 3: acc=0.88, prec=0.85, rec=0.87, f1=0.86 ⚠ (4 drifts)
└─ TRIGGER: Retraining (2 ventanas consecutivas con drift)
```

### 3. Métodos Estadísticos

#### a) Effect Size (Cambio Absoluto)
```
Si |actual - baseline| > 0.05 (5%) → Potencial drift
Ejemplo: accuracy cae de 0.96 a 0.90 → 6% de caída → DRIFT
```

#### b) T-Test (Significancia)
```
Compara últimas 5 observaciones contra baseline
H0: media(últimas_5) = baseline
Si p_value < 0.05 → Cambio estadísticamente significativo
```

#### c) EWMA (Tendencia Sostenida)
```
EWMA_actual = 0.3 * valor_actual + 0.7 * EWMA_anterior
Si |EWMA - baseline| > 0.07 → Divergencia sostenida
Detecta problemas que no son puntuales sino tendencias
```

## Configuración

### Variables de Entorno

```bash
# Habilitar/deshabilitar
ENABLE_PERFORMANCE_DRIFT=true

# Umbrales
PERF_DRIFT_EFFECT_SIZE=0.05          # Cambio mínimo (5%)
PERF_DRIFT_P_VALUE=0.05              # p-value para t-test
PERF_DRIFT_CONSECUTIVE=2             # Ventanas consecutivas para trigger

# Parámetro de suavizado EWMA
EWMA_ALPHA=0.3                       # (no configurable, hardcoded)
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
Baseline metrics: {'accuracy': 0.96, 'precision': 0.96, 'recall': 0.96, 'f1': 0.96}
Current metrics: {'accuracy': 0.88, 'precision': 0.87, 'recall': 0.89, 'f1': 0.87}

  Performance Status: DRIFT ⚠  consecutive=1/2
```

### Métricas en MLFlow

Experimento: `performance-drift-monitoring`

Cada ventana registra:
```
current_accuracy:      0.88
baseline_accuracy:     0.96
change_pct_accuracy:   -8.33
ttest_pval_accuracy:   0.0234
perf_drift_detected:   1
perf_consec_drifts:    1
```

## Escenarios Reales

### Escenario 1: Degradación Gradual
El modelo entrenado con datos de hace 6 meses ya no refleja la distribución actual:
- **Síntomas**: Caída gradual en accuracy de 0.96 → 0.92 → 0.88
- **Detección**: EWMA detectará tendencia sostenida después de varios ciclos
- **Acción**: Retraining automático disparado

### Escenario 2: Ruido Puntual
Una ventana tiene predicciones erróneas por datos corruptos:
- **Síntomas**: Accuracy cae a 0.80 en 1 ventana, luego vuelve a 0.95
- **Detección**: Effect size dispara, pero no hay tendencia sostenida (EWMA se recupera)
- **Acción**: Sin trigger (sólo 1 ventana con drift, necesita 2 consecutivas)

### Escenario 3: Shift de Distribución
El modelo ve clases diferentes a las que aprendió:
- **Síntomas**: Accuracy global OK (0.94), pero Recall de clase X baja a 0.60
- **Detección**: Si 2+ métricas están fuera de threshold
- **Acción**: Performance drift trigger + retraining

## Pruebas Automatizadas

### Tests Unitarios
```bash
pytest test_performance_drift.py -v
```

Verifica:
- Cálculo correcto de métricas
- Detección de drift real vs falsos positivos
- Comportamiento de t-test y EWMA
- Manejo de edge cases

### Tests de Integración
```bash
pytest test_performance_drift_integration.py -v -s
```

Verifica:
- Escenarios realistas (degradación gradual, shift de distribución)
- Logging a MLFlow
- Recovery después de drift
- Sensibilidad de umbrales

### Suite QA Completa
```bash
docker compose run --rm test-runner
```

Niveles:
1. **Level 1**: Data validation
2. **Level 2**: Model quality gates
3. **Level 3**: API tests
4. **Level 4**: Performance drift tests

## Ajuste de Umbrales

### Más Sensible (más alertas)
```bash
PERF_DRIFT_EFFECT_SIZE=0.02          # Detecta cambios de 2%
PERF_DRIFT_P_VALUE=0.10              # p-value más lenient
PERF_DRIFT_CONSECUTIVE=1             # Trigger en 1 ventana
```

### Más Conservador (menos alertas)
```bash
PERF_DRIFT_EFFECT_SIZE=0.10          # Detecta cambios de 10%
PERF_DRIFT_P_VALUE=0.01              # p-value muy estricto
PERF_DRIFT_CONSECUTIVE=3             # Trigger en 3 ventanas
```

## Troubleshooting

### "Performance metrics logged to MLFlow ✓" pero no aparecen runs

**Causa**: MLFlow no está listo al iniciar drift-detector

**Solución**: Aumentar `retries` en `wait_for()` en detector.py

```python
wait_for(f"{MLFLOW_URI}/", "MLFlow", retries=30, delay=4)
```

### Muchos falsos positivos

**Causa**: Umbrales muy sensibles

**Solución**: Aumentar PERF_DRIFT_EFFECT_SIZE a 0.08-0.10

### Nunca detecta drift real

**Causa**: Umbrales muy altos

**Solución**: Reducir PERF_DRIFT_EFFECT_SIZE a 0.02-0.03

## Referencias

- **Effect Size**: Estándar en estadística (Cohen's d)
- **T-test**: Prueba si hay diferencia significativa de medias
- **EWMA**: Usado en control de calidad industrial (SPC)
- **Performance Drift**: Concepto de Model Drift Detection (Eken et al., 2025)

## Ejemplo: Monitoreo Completo

```bash
# 1. Iniciar stack
docker compose up -d

# 2. Ver logs del drift detector
docker compose logs -f drift-detector

# 3. Visualizar en MLFlow
# Ir a http://localhost:5000 → Experimento "performance-drift-monitoring"

# 4. Simular degradación (reemplazo en train.py)
# Reducir n_estimators a 10 para intentar degradación
docker compose run --rm model-trainer

# 5. Esperar ciclos y observar
# - performance_drift_detected = 1 cuando detecte drift
# - perf_consec_drifts incrementa
# - Si alcanza threshold → retraining trigger

# 6. Revisar logs de retraining
docker compose logs model-trainer
```

---

**Última actualización**: 2026-06-04  
**Versión**: 1.0.0
