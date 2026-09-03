# Sesión 2026-09-01 — Dashboard Streamlit, HuggingFace Space y fixes de predicción

## Contexto

Fase de portafolio. El objetivo fue añadir una demo interactiva en vivo
(HuggingFace Spaces), robustecer el pipeline de predicción y alinear las
métricas que muestra el dashboard con la métrica de negocio de referencia.
Se implementaron optimizaciones de rendimiento, un dashboard Streamlit
autocontenido, y se corrigieron dos bugs reales en la capa de predicción
(intervalo de confianza negativo y colapso de la feature de transacciones).

## Cambios por archivo

### `src/models/predict.py`
- **Cache de datos raw:** nueva función `_load_raw_predict_cached()` con
  cache a nivel módulo de `load_raw_data(predict=True)` (los 4 CSVs de apoyo
  son estáticos y determinísticos). Evita releerlos en cada llamada.
  Predicción pasa a ~0.3-0.4s por tienda.
- **Fix intervalo de confianza (punto 1):** extraída la lógica de intervalos
  a la función pura tipada `_build_confidence_intervals(y_pred_log,
  std_sales, z=1.96, upper_factor=1.5)`. Ahora `upper_bound` también se
  recorta a ≥0 y se fuerza `upper >= lower`. Antes, en familias esporádicas
  (predicted≈0) `upper_bound` podía ser negativo (p.ej. -0.01, -0.02, -0.05)
  e incoherente (`upper < lower = 0`).
- `predict()` usa el helper para asignar `lower_bound`/`upper_bound`.

### `src/features/build_features.py`
- **Fix feature de transacciones (causa raíz del colapso):**
  `_build_transaction_features` ahora alinea **por fecha**, no por posición.
  Causa del bug: hay 33 filas por `(store, date)` (una por familia) y el
  `shift(horizon)` posicional retrocedía ~1 día en vez de `horizon` días.
  En la frontera futura (dataset termina 2017-08-15, filas futuras con
  `transactions=0`) esto colapsaba `trans_lag_{h}` y
  `trans_rolling_mean_{h}` a 0 → como `transactions` es feature top-3, el
  modelo predecía ~0 desde el día 2 (p.ej. store 1, CLEANING, h30:
  Aug16 pred=887 → Aug17 pred=339 → Aug18 pred=0.53) y la banda colapsaba.
  Ahora se construye una serie store-level única `(store, date,
  transactions)`, se aplica `shift(horizon)` en días reales y se mapea de
  vuelta a las 33 filas. Para fechas futuras `trans_lag_30` = transacciones
  reales de hace 30 días (existen hasta Aug15) → no colapsa. **Requiere
  retrain.**

### `src/models/evaluate.py`
- **Persistir métricas globales:** en `run_evaluation` se guarda
  `data/predictions/global_metrics_h{horizon}.parquet` (1 fila con rmse,
  mae, mape, rmsle, wape) junto a family_metrics/store_metrics.

### `src/api/` (sin cambios funcionales en esta sesión)

### `dashboard/app.py` (nuevo)
- Dashboard Streamlit standalone que importa `src/models/predict.py`
  directamente (paridad total con la API).
- Sidebar: selector de tienda (1-54), horizonte (7/30) y familia (por nombre
  desde `feature_pipeline_h7.pkl`).
- KPIs por horizonte leídos de `global_metrics_h{h}.parquet` (métricas
  globales reales de test): WAPE 10.51% / 12.34%, MAE 49.73 / 58.41,
  RMSE 194.35 / 218.19.
- Gráfico interactivo (Plotly): predicción futura con banda de intervalo de
  confianza al 95%, o top 8 familias por volumen.
- Tabla de detalle con predicción por fecha y límites.
- Cache con `st.cache_resource(load_assets)` y `st.cache_data(get_predictions)`.

### `dashboard/requirements.txt` (nuevo)
- streamlit, plotly, pandas, numpy, pyarrow, joblib, pyyaml, lightgbm,
  mlflow, python-dotenv, scikit-learn.

### `dashboard/README.md` (nuevo)
- Guía para ejecutar el dashboard en local y publicarlo en HuggingFace Spaces.

### `scripts/sync_hf_space.sh` (nuevo)
- Prepara un repositorio de HuggingFace Space autocontenido: copia `src/`,
  `dashboard/`, `configs/`, los artefactos (modelos, datos) y pone `app.py` +
  `requirements.txt` en la raíz. Versiona los binarios con git LFS
  (`models/*.pkl`, `data/**/*.parquet`, `data/raw/*.csv`). Excluye egg-info.

### `tests/test_predict_intervals.py` (nuevo)
- 5 tests de `_build_confidence_intervals` (caso esporádico, caso normal,
  redondeo a 2 decimales, upper>=lower, e invariante end-to-end con
  `predict_by_store` con skip si faltan artefactos).

### `tests/test_features.py`
- `make_synthetic_df` corrige `transactions` para que sea por tienda.
- Helper `_make_future_rows`.
- Nueva clase `TestTransactionFeaturesDateAligned` (3 tests: alineación por
  fecha, no colapso en frontera futura, store-level).

### `README.md` (raíz)
- Sección "Demo en vivo", badge de HuggingFace, `dashboard/` y `scripts/` en
  estructura, CHANGELOG v0.5.0 y nueva entrada de fixes.

## Verificación ejecutada

```bash
./venv/bin/python -m pytest tests/ -q          # 81 passed
./venv/bin/python -m pytest tests/test_features.py -q   # 12 passed
./venv/bin/python -m pytest tests/test_predict_intervals.py -q  # 5 passed
```

- Dashboard validado con `streamlit.testing.v1.AppTest`: sin excepciones,
  KPIs correctos (WAPE 10.51%, MAE 49.73, RMSE 194.35 en h7).
- Fix intervalo: con datos reales (store 1, family 1) ya no hay `upper < 0`
  ni `upper < lower`.
- Staging del Space `/tmp/hf_staging` probado punta a punta (sync + dashboard
  autocontenido).

## Pendiente / fuera de alcance

- **Retrain h7 y h30 (REQUERIDO):** el fix de `_build_transaction_features`
  cambia las features de entrenamiento. Hay que reentrenar para que
  producción use las features corregidas:
  ```bash
  ./venv/bin/python src/models/retrain.py --horizon 7 --params-file reports/optuna/best_params_h7.json
  ./venv/bin/python src/models/retrain.py --horizon 30 --params-file reports/optuna/best_params_h30.json
  ./venv/bin/python src/models/evaluate.py --horizon 7
  ./venv/bin/python src/models/evaluate.py --horizon 30
  ```
  (NO es necesario re-ejecutar Optuna: los hiperparámetros ya están
  optimizados y el cambio de feature engineering no los altera.)
- Re-verificar `predict_by_store(df, 30, 1)` CLEANING tras el retrain: debe
  dar predicciones estables (sin colapso) y banda coherente.
- Publicar el Space real de HuggingFace (falta credenciales).
- Plan C del dashboard (toggle "Futuro / Backtest") aún no implementado.