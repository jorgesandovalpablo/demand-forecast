# Guía de Validación End-to-End — v0.4.1

Objetivo: verificar el pipeline completo con el `DemandFeatureEngineer`
stateful y la promoción segura de modelos, y capturar las métricas
definitivas para el README.

Requisitos previos:
- Dataset en `data/raw/` (Kaggle: *Store Sales - Time Series Forecasting*).
- `.env` con credenciales DagsHub (`MLFLOW_TRACKING_USERNAME/PASSWORD`).
- Ejecutar desde la raíz del proyecto con el venv activado.

---

## Paso 0 — Tests y linting (sin datos)

```bash
pytest tests/ -v --tb=short          # 54 tests esperados en verde
flake8 --select=F src/ tests/        # sin salida = limpio
```

## Paso 1 — Preprocessing

```bash
python src/data/preprocessing.py
```

Esperado:
- `data/processed/train_processed.parquet` generado.
- Log final: `train: (3000888, 17)` aprox. (post merge corregido).
- Sin error de duplicados en el merge de holidays.

## Paso 2 — Entrenamiento por horizonte

```bash
python src/models/train.py --horizon 7
python src/models/train.py --horizon 30
```

Esperado:
- Logs `Ajustando DemandFeatureEngineer` al inicio.
- CV walk-forward con métricas RMSE/MAE/MAPE/RMSLE/WAPE por fold.
- Artefactos regenerados en `models/`:
  `lgbm_h{h}.pkl`, `features_h{h}.pkl`, `feature_pipeline_h{h}.pkl`.
- Run visible en DagsHub con params, métricas CV y modelo logueado.

> ⚠️ Los artefactos anteriores a este punto quedan obsoletos; no mezclar.

## Paso 3 — Evaluación sobre test set

```bash
python src/models/evaluate.py --horizon 7
python src/models/evaluate.py --horizon 30
```

Esperado:
- Carga `feature_pipeline_h{h}.pkl` (log: `Cargando historial procesado...`).
- Métricas globales + por familia + por tienda + temporales.
- Parquets en `data/predictions/family_metrics_h{h}.parquet` y
  `store_metrics_h{h}.parquet`.
- 📸 **Capturar los valores de métricas globales → pegarlas en el README.**

## Paso 4 — Inferencia batch

```bash
python src/models/predict.py --horizon 7
```

Esperado:
- Tabla con `date, store_nbr, family, predicted_sales, lower_bound, upper_bound`.
- `lower_bound <= predicted_sales <= upper_bound` en todas las filas.
- Parquet en `data/predictions/predictions_daily_YYYYMMDD.parquet`.
- **Sin** `predict_df.parquet` generado (side-effect eliminado).

## Paso 5 — Promoción segura + Model Registry

```bash
# Primera pasada: decisión por métricas (probablemente acepte: inf baseline)
python src/models/retrain.py --horizon 7

# Segunda pasada: forzar para validar rotación y backups
python src/models/retrain.py --horizon 7 --force
ls models/*backup*
```

Esperado:
- Primera pasada: `No hay modelo previo → aceptado` o comparación vs
  `family_metrics`; producción intacta hasta la promoción.
- Segunda pasada (`--force`): backups timestamped de los 3 artefactos
  (`*_backup_*.pkl`) y staging `_new` eliminado tras rotar.
- Run `retrain_h7_YYYYMMDD` en DagsHub con `model_updated=true`.
- **Registry (requiere credenciales DagsHub en `.env`):**
  - Log `Modelo registrado en el registry: demand-forecast-daily@production (vN)`.
  - En DagsHub → Models: versión nueva con tags `horizon`, `test_mae`,
    `test_wape`, `promoted_at` y alias `production`.
- Si el registro falla: warning en logs, promoción local intacta.

### Verificar recuperación desde el registry

```bash
mv models/lgbm_h7.pkl /tmp/lgbm_backup.pkl
python -c "from src.models.predict import ModelRegistry; m = ModelRegistry.load(7); print(type(m))"
mv /tmp/lgbm_backup.pkl models/lgbm_h7.pkl
```

Esperado: warning `Artefactos locales incompletos... consultando el
registry`, descarga de la versión `@production` a `models/` y carga OK.

### Rollback manual

```bash
python -c "from src.models.registry import rollback_production; rollback_production(7, <version_anterior>)"
```

## Paso 6 — API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

```bash
# Health
curl http://localhost:8000/health
# → {"status":"healthy","models_loaded":[7,30],...}

# Predicción completa de una tienda
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"store_nbr": 1, "horizon": 7}'

# Filtro por familia inexistente → 404 (regresión del fix)
curl -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"store_nbr": 1, "horizon": 7, "family": 99}'
# → 404
```

---

## Checklist final

- [ ] 54 tests en verde + flake8 limpio
- [ ] Artefactos v0.2 regenerados para h7 y h30
- [ ] Métricas capturadas y pegadas en README
- [ ] Backups de retraining presentes y staging limpio
- [ ] API responde 200 / 404 correctamente
- [ ] Runs de train, evaluate y retrain visibles en DagsHub
