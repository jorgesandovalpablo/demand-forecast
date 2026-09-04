# Sesión 2026-09-03 — Ventana backtest en dashboard y corrección del intervalo de confianza (li/ls)

## Contexto

Dos objetivos en esta sesión:

1. **Ventana backtest en el dashboard:** graficar el desempeño del modelo
   sobre las últimas 8 semanas (test set) mostrando el real vs la predicción
   backtest junto a la predicción futura, tanto por familia como en la vista
   agregada "(Todas las familias)".
2. **Corregir el intervalo de confianza (li/ls)** del dashboard, que era
   ancho y asimétrico. Ejemplo tienda 1, familia GROCERY, 2017-08-16:
   `predict=3179, li=970, ls=18851` (ls/pred ≈ 5.9, li/pred ≈ 0.31).

### Causa raíz del IC ancho/asimétrico (diagnóstico)

El intervalo se construye en escala log y se revierte con `expm1`, lo que lo
vuelve **multiplicativo en escala real** (con σ≈0.605: upper ×5.9, lower ×0.31
respecto al punto). Factores que lo agravaban:

- `upper_factor = 1.5` asimétrico → el extremo superior usaba 2.94σ vs 1.96σ.
- El punto `exp(μ)` es la **mediana** de una lognormal → siempre pegado al li.
- `std_sales = venta_std_historica` (volatilidad histórica total, no error
  predictivo) y constante en todo el horizonte.

El IC es **exclusivo de `predict.py`**; `evaluate.py` no genera intervalos.

## Cambios por archivo

### `src/models/evaluate.py`
- **`build_backtest_df(test_df, y_pred, target) -> DataFrame`**: helper puro
  que construye `backtest_predictions_h{h}` con columnas
  `[date, store_nbr, family, real_sales, y_pred_real, y_pred_log]`.
  `real_sales = clip(expm1(target), 0, None)`,
  `y_pred_real = clip(expm1(y_pred), 0, None)` (clip por paridad con
  `predict.py`).
- En `run_evaluation` persiste
  `data/predictions/backtest_predictions_h{horizon}.parquet` + log_artifact
  MLflow.
- **`compute_residual_std(backtest_df) -> {'global': float, 'df': DataFrame}`**:
  helper puro. `resid = y_pred_log - log1p(real_sales)`; std por
  `(store_nbr, family)` y std global. Persistido en
  `models/residual_std_h{horizon}.pkl` + log_artifact MLflow (h7 global≈0.3916,
  h30 global≈0.4020).

### `src/models/predict.py`
- `ModelRegistry`: nuevo atributo `_std`; `_load_residual_std(horizon,
  pipeline_path)` carga `models/residual_std_h{h}.pkl` (fallback a
  `pipeline.store_stats.venta_std_historica` si falta); `get_residual_std` y
  limpieza en `clear_cache`.
- `_build_confidence_intervals(..., upper_factor=1.0)`: default de
  `upper_factor` cambiado `1.5 → 1.0` (simétrico en log).
- En `predict()` se reemplaza el std por grupo (`venta_std_historica`) por el
  σ residual por `(store_nbr, family)` con `fillna(std global)`.

### `dashboard/app.py`
- `load_assets()` carga `backtest[horizon]`.
- Vista familia: trazas `Real (test)` y `Backtest` + vline "Predicción inicia".
- Vista "(Todas las familias)": serie temporal agregada por fecha
  (real_sales, y_pred_real, predicted_sales) + vline.

### Tests
- `tests/test_evaluate_backtest.py` (nuevo): columnas/escalas de
  `build_backtest_df`, claves de agrupación, parquet generado (skipif), y
  tests de `compute_residual_std`.
- `tests/test_dashboard.py` (nuevo): AppTest de las vistas "todas" y familia.
- `tests/test_predict_intervals.py`: test de simetría por defecto
  (d_low ≈ d_up en escala log).

## Resultado

- **IC corregido** (tienda 1 GROCERY): `[970, 3179, 18851]` →
  `[2326, 3044, 3983]` (ls/pred 5.9 → 1.31, li/pred 0.31 → 0.76).
- Suite de tests: **99 passed** (incluye AppTest del dashboard).
- `evaluate.py` h7/h30 re-ejecutado → regenera `residual_std` y parquets de
  backtest.

## Pendientes

- Retrain h7/h30 (fix de transacciones de la sesión 09-01 sigue pendiente).
- Publicación del Space HF.

---

## 2ª parte: rediseño visual del dashboard y z configurable

Se continuó el trabajo de la sesión del mismo día con un rediseño visual
del dashboard, la configuración del multiplicador del IC y varios fixes.

### Diagnóstico del salto 16-ago (~14K)

Se analizó el salto de ~10K (15-ago) a ~14K (16-ago) en la predicción
de tienda 1. Conclusiones:

- **No es un offset inducido** por el placeholder `target=0` de
  `predict()`: el forecast es single-shot, `min_lag == horizon` (7/30),
  por lo que todos los lags de la ventana pronosticada referencian
  fechas reales.
- **Es una alza real predicha**, impulsada por la columna
  `onpromotion` del dataset. El 16-ago tiene un pico promocional
  extremo (524 unidades en promoción en tienda 1, frente a la media
  reciente de ~6.4/familia). El modelo responde legítimamente a esa
  feature.
- El forecast **no es plano a 14K** — oscila 8K–18.6K con
  estacionalidad semanal (domingos ~8K, 01-sep ≈18.6K). Media h30
  ≈13.5K.

### Cambios implementados

#### `configs/config.yaml`
- **`confidence.z: 1.1`** (nueva sección): el multiplicador del IC ya
  no está hardcodeado en 1.96 en `predict.py`; ahora se lee desde
  config. Con z=1.1, el IC se estrecha ~44% (≈86% confianza vs el
  anterior 95%).

#### `src/models/predict.py`
- `_build_confidence_intervals(..., z: float = 1.96)`: sigue
  aceptando z como parámetro con default 1.96.
- En `predict()`: lee `config['confidence']['z']` y lo pasa al
  helper.

#### `dashboard/app.py`
- **IC como área semi-transparente:** sustituidas las dos trazas
  `li`/`ls` rojas por relleno `fill="tonexty"` con
  `fillcolor="rgba(158,202,225,0.20)"` (sombra azul claro 20%).
  Sin leyenda para el IC.
- **Colores actualizados:**
  - Real (test): `#2ca02c` (verde) → `#37474f` (azul marino/gris
    oscuro).
  - Predicción: `#1f77b4` → `#1565c0` (azul vibrante).
- **Layout full-width:** eliminadas las columnas `[3, 2]`; la gráfica
  ocupa todo el ancho.
- **Tabla "Detalle" debajo:** la tabla se movió de la columna derecha
  a debajo de la gráfica.
- **Altura de gráficas:** de 420px a 560px.
- **Leyendas más grandes:** `font=dict(size=16)` (antes 12).
- **Fondo gris en backtest:** `fig.add_vrect(x0=bt_min, x1=pred_start,
  fillcolor="gray", opacity=0.10, layer="below")` en ambas vistas.
- **IC en vista agregada "Todas las familias":** `groupby` ahora suma
  `["predicted_sales", "lower_bound", "upper_bound"]` + trazas de
  relleno (upper invisible + lower con fill). Antes solo mostraba la
  predicción.
- **Caption de onpromotion:** bajo la gráfica, explicando que los
  picos del forecast se alinean con la columna `onpromotion`.

#### `tests/test_dashboard.py`
- Eliminado `import pandas as pd` sin usar (F401, fix CI).

### Resultado
- 99/99 tests verdes, flake8 limpio (F).
- `z=1.1` reduce el ancho del IC ~44% vs el anterior (1.96).