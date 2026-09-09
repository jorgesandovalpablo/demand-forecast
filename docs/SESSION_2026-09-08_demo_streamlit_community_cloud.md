# Sesión 2026-09-08 — Demo Streamlit Community Cloud

## Contexto

Implementación de demo autocontenida para despliegue en Streamlit Community Cloud
(share.streamlit.io). La demo usa assets pre-calculados (predicciones parquet,
métricas, backtest) para ser independiente de `src/` en runtime.

**Decisión de plataforma:** HuggingFace Spaces deprecó Streamlit SDK (2025-04-30).
Gradio+ZeroGPU requiere PRO para esta cuenta. Streamlit Community Cloud es la
alternativa gratuita: deploy desde GitHub, 1GB RAM, auto-redeploy en push a main.

## Archivos creados

| Archivo | Descripción |
|---|---|
| `scripts/build_demo_bundle.py` | Genera assets pre-calculados (predicciones, metadata, backtest, métricas) |
| `deploy/app.py` | Dashboard Streamlit autocontenido (365 líneas, 0 imports de src/) |
| `deploy/requirements.txt` | Dependencias fijas para Community Cloud |
| `tests/test_deploy_dashboard.py` | 5 tests AppTest con skipif |

## Bugs encontrados y corregidos

### 1. categorical_feature mismatch por predict_by_store por tienda

**Problema:** `build_demo_bundle.py` llamaba `predict_by_store(historical, h, store)`
para cada tienda. Esto causaba un DataFrame de test vacío en
`prepare_prediction_data()` → `holiday_impact_type` sin categorías →
LightGBM lanzaba `categorical_feature mismatch`.

**Causa raíz:** `predict_by_store()` filtra el historial por tienda antes de pasar a
`predict()`. Con pocas filas (1 tienda × 33 familias × ~365 días), el subset
generado por `prepare_prediction_data()` no contiene todas las categorías de
`holiday_impact_type` → LightGBM detecta que las categorías del modelo difieren
del DataFrame de predicción.

**Fix:** `build_demo_bundle.py` ahora llama `predict(historical, h)` UNA vez por
horizonte (todas las tiendas juntas), luego filtra por store para escribir los
parquets individuales. El historial completo garantiza que todas las categorías
estén presentes.

### 2. holiday_impact_type con categorías inconsistentes

**Problema:** En `build_features.py`, `holiday_impact_type` se creaba con
`.astype('category')` que derivaba categorías automáticamente del subset. Cada
subset (train vs predict, h7 vs h30) podía tener categorías diferentes →
`pandas_categorical` del modelo no coincidía con los datos de predicción.

**Fix:** Hardcoded `IMPACT_CATEGORIES = ['atypical', 'negative', 'neutral', 'none', 'positive']`
usado en `pd.Categorical()` en `_build_holiday_features`. Las categorías son
las mismas independientemente del subset.

### 3. holiday_locale como object en predicción

**Problema:** Ambos modelos (h7, h30) tienen `pandas_categorical` con 2 columnas:
col 0 = `holiday_locale` ['Local','National','Regional'], col 1 = `holiday_impact_type`.
El encoding loop en `build_features.py:99-102` convertía `holiday_locale` de
category a int16 (códigos), eliminando su dtype category. LightGBM veía int16
en vez de category → mismatch.

**Fix:** Después del encoding loop, se reconstruye `holiday_locale` como
`pd.Categorical` con categorías explícitas:
```python
if 'holiday_locale' in df.columns:
    df['holiday_locale'] = pd.Categorical(
        df['holiday_locale'],
        categories=['Local', 'National', 'Regional'],
    )
```

### 4. Modelos .pkl innecesarios en deploy/assets (41MB → ~5MB)

**Problema:** `build_demo_bundle.py` copiaba 8 archivos .pkl (35MB) a
`deploy/assets/models/`. El dashboard `deploy/app.py` nunca los usa — solo
lee JSONs y parquets. El bundle completo pesaba 41MB; Community Cloud tiene
1GB de RAM y el repo no debería transportar binarios innecesarios.

**Fix:** Eliminada `_copy_model_artifacts()` de `build_demo_bundle.py`. Eliminada
referencia `MODELS_DIR` de `deploy/app.py`. Agregado `deploy/assets/models/`
a `.gitignore`. El bundle se redujo de 41MB a ~5MB (solo parquets de
predicciones + backtest + métricas + JSONs).

## Build result (post-fix)

- 108 parquet files (54 stores × 2 horizons): ~1.8MB
- 4 backtest/metrics parquets: ~3MB
- 2 JSON metadata: ~10KB
- **Total: ~5MB** (antes: 41MB con modelos)

## Tests

Suite completa: 104/104 passed (incluye 5 nuevos en test_deploy_dashboard.py).

## Deploy

- **Plataforma:** Streamlit Community Cloud (share.streamlit.io)
- **Repo:** jorgesandovalpablo/demand-forecast
- **Main file:** deploy/app.py
- **Python:** 3.11 o 3.12
- **URL desplegada:** https://demand-forecast-minimarket.streamlit.app (en vivo)
- **Ciclo:** retrain → build_demo_bundle.py → commit → push → auto-redeploy

## Estado (post-deploy)

- ✅ **URL desplegada:** https://demand-forecast-minimarket.streamlit.app
- ✅ **README.md:** placeholders `<SPACE_URL>` reemplazados, heading L1 corregido
- ✅ **dashboard/README.md:** referencias de HuggingFace Spaces migradas a Community Cloud
