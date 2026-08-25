# demand-
# 🛒 Demand Forecast — Retail Time Series

> Pipeline de ML end-to-end para predicción de demanda en minimercados de Ecuador.
> Modelos LightGBM con horizontes de 7 y 30 días, MLflow tracking en DagsHub,
> FastAPI deployment y retraining con promoción segura (cron semanal vía GitHub Actions).

![CI](https://github.com/jorgesandovalpablo/demand-forecast/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-green)
![FastAPI](https://img.shields.io/badge/api-FastAPI-teal)
![MLflow](https://img.shields.io/badge/tracking-MLflow-orange)
![Docker](https://img.shields.io/badge/deploy-Docker-blue)

---

## 📋 Tabla de contenidos

- [Problema de negocio](#-problema-de-negocio)
- [Solución](#-solución)
- [Arquitectura del pipeline](#-arquitectura-del-pipeline)
- [Stack tecnológico](#-stack-tecnológico)
- [Resultados](#-resultados)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Reproducir el proyecto](#-reproducir-el-proyecto)
- [API](#-api)
- [Decisiones de diseño](#-decisiones-de-diseño)
- [Hallazgos del EDA](#-hallazgos-del-eda)
- [Autor](#-autor)

---

## 🎯 Problema de negocio

Una cadena de **54 minimercados en Ecuador** necesita anticipar la demanda
de **33 familias de productos** para optimizar sus compras y evitar
problemas de inventario.

El equipo de cadena de suministro requiere dos horizontes de predicción:

| Horizonte | Granularidad | Uso operativo |
|---|---|---|
| Próximos 7 días | Día a día | Compras operativas semanales |
| Próximo meses | Mes a mes | Planificación de inventario |

**Costo del error:**

| Error | Consecuencia |
|---|---|
| Stockout (falta de stock) | Pérdida de venta + cliente insatisfecho |
| Overstock (exceso de stock) | Capital inmovilizado + merma en perecederos |

---

## 💡 Solución

Pipeline de ML end-to-end con dos modelos LightGBM especializados por horizonte,
entrenados sobre **1,782 series temporales simultáneas** (54 tiendas × 33 familias):

```
Datos históricos (4.5 años — 3,000,888 registros)
        │
        ▼
Feature Engineering (~50 features por horizonte)
  ├── Lags temporales respetando horizonte
  ├── Rolling statistics con shift correcto
  ├── Festivos clasificados por impacto
  ├── Promociones y transacciones
  └── Precio del petróleo (correlación -0.75)
        │
        ▼
LightGBM Global — modelo único para todas las series
  ├── Modelo diario  (horizon=7)  → D+1 a D+7
  └── Modelo mensual (horizon=30) → M+1 a M+3
        │
        ▼
FastAPI REST API → Equipo de cadena de suministro
```

---

## 🏗️ Arquitectura del pipeline

```
data/raw/ (CSV originales)
        │
        ▼
ingestion.py          → Carga + validación de esquema
        │
        ▼
preprocessing.py      → Merge correcto de 6 archivos
                         Tratamiento de nulos
                         log1p al target
                         Optimización de memoria
        │
        ▼
data/processed/train_processed.parquet
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
DemandFeatureEngineer(horizon=7)   DemandFeatureEngineer(horizon=30)
  fit(): categorías, store_stats,    fit(): ídem
         ranking global
  transform(): lags, rolling,
  festivos, promo, transacciones
  (en memoria — sin parquet intermedio)
        │                                      │
        ▼                                      ▼
validation.py                    validation.py
walk-forward CV (5 folds)        walk-forward CV (5 folds)
        │                                      │
        ▼                                      ▼
train.py + MLflow                train.py + MLflow
lgbm_h7.pkl                      lgbm_h30.pkl
features_h7.pkl                  features_h30.pkl
feature_pipeline_h7.pkl          feature_pipeline_h30.pkl
        │                                      │
        └─────────────────┬────────────────────┘
                          ▼
                  predict.py / evaluate.py
                  (cargan el pipeline serializado
                   con .transform() → paridad garantizada)
                          │
                          ▼
                  FastAPI (main.py)
                  POST /predict
```

> **Cambio clave:** las features ya no se persisten en parquets
> intermedios (`train_features_d*.parquet`). El `DemandFeatureEngineer`
> aprende su estado en `fit()` durante el entrenamiento y se serializa
> junto al modelo; serving y evaluación reconstruyen features con
> `.transform()` sobre ese estado congelado.


**Pipeline de reentrenamiento automático:**

```
Disparo manual / programado (⚠️ cron pendiente)
        │
        ▼
retrain.py
  ├── Ejecuta pipeline completo a STAGING (_new)
  ├── Producción NUNCA se toca durante el entrenamiento
  ├── Evalúa el nuevo modelo sobre el test set (8 semanas)
  ├── Compara métricas (threshold 1% mejora en MAE)
  ├── Si mejora → promueve los 3 artefactos con backup timestamped
  └── Si no mejora → descarta staging, mantiene producción
              │
              ▼
        MLflow registra versión
        Backups en models/*_backup_*.pkl (retención: 3)
```

---

## 🛠️ Stack tecnológico

| Categoría | Tecnología | Uso |
|---|---|---|
| **Modelado** | LightGBM | Modelo principal de forecast |
| **Tracking** | MLflow + DagsHub | Experimentos, métricas y modelos |
| **API** | FastAPI + Uvicorn | Serving de predicciones |
| **Validación** | Pydantic v2 | Schemas de entrada/salida |
| **Container** | Docker + docker-compose | Deployment reproducible |
| **CI** | GitHub Actions | Tests automáticos en cada push |
| **Optimización** | Optuna | ⚠️ Pendiente — declarado en requirements pero sin uso en src/ |
| **Versionado datos** | DVC | ⚠️ Pendiente — declarado en requirements pero sin uso en src/ |
| **Dependencias** | pip-tools | Versiones exactas y reproducibles |
| **Calidad** | Black + Flake8 + isort | Estilo y linting automático |
| **Tests** | pytest | 16 tests unitarios/de integración (features y API) |

---

## 📊 Resultados

### Modelo Diario (horizon=7 días)

> ⚠️ **Pendiente:** ejecutar `evaluate.py` con el pipeline actual para
> poblar estas tablas. Las métricas de MLflow históricas corresponden a
> versiones anteriores del feature engineering.

| Métrica | CV Mean | CV Std | Test Set |
|---|---|---|---|
| RMSE | - | - | - |
| MAE | - | - | - |
| MAPE | - | - | - |
| WAPE | - | - | - |
| RMSLE | - | - | - |

### Modelo Mensual (horizon=30 días)

| Métrica | CV Mean | CV Std | Test Set |
|---|---|---|---|
| RMSE | - | - | - |
| MAE | - | - | - |
| MAPE | - | - | - |
| WAPE | - | - | - |
| RMSLE | - | - | - |

> 📌 Métricas se actualizarán tras ejecutar `evaluate.py` con los artefactos vigentes.

### Top features más importantes

| Rank | Feature | Grupo |
|---|---|---|
| 1 | lag_7 / lag_30 | Lags |
| 2 | rolling_mean_7d / rolling_mean_30d | Rolling |
| 3 | transactions | Externo |
| 4 | lag_14 / lag_60 | Lags |
| 5 | dcoilwtico | Externo |

### Familias con mayor error (baseline)

Las familias con alto porcentaje de ceros presentan mayor error:

| Familia | % Ceros | Estrategia |
|---|---|---|
| BOOKS | 97% | Modelo global + flag esporádica |
| BABY CARE | 94% | Modelo global + flag esporádica |
| SCHOOL/OFFICE | 74% | Modelo global + flag esporádica |
| GROCERY I | 8% | Modelo global — alta precisión |
| BEVERAGES | 8% | Modelo global — alta precisión |

---

## 📁 Estructura del proyecto

```
demand-forecast/
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint + tests en cada push/PR
│       └── retrain.yml         # Retraining semanal (cron) o manual
│
├── configs/
│   └── config.yaml              # Fuente de verdad única del proyecto
│
├── data/
│   ├── raw/                     # CSV originales — nunca se modifican
│   ├── processed/               # Parquet generados por el pipeline
│   └── predictions/            # Outputs del modelo y métricas por familia/tienda
│
├── docs/
│   └── SESSION_*.md             # Bitácora de sesiones de desarrollo
│
├── models/                      # Artefactos serializados por horizonte
│   ├── lgbm_h{7,30}.pkl             # Booster LightGBM
│   ├── features_h{7,30}.pkl         # Lista exacta de features de entrenamiento
│   └── feature_pipeline_h{7,30}.pkl # DemandFeatureEngineer stateful (fit/transform)
│   └── *_backup_*.pkl               # Backups del retraining (retención 3)
│
├── notebooks/
│   └── 01_eda.ipynb            # Análisis exploratorio completo
│
├── src/
│   ├── data/
│   │   ├── ingestion.py        # Carga y validación de esquema
│   │   └── preprocessing.py    # Merge, nulos, log1p, memoria
│   ├── features/
│   │   └── build_features.py   # DemandFeatureEngineer (fit/transform)
│   ├── models/
│   │   ├── validation.py      # Walk-forward cross validation + métricas
│   │   ├── train.py           # LightGBM + MLflow + pipeline a staging
│   │   ├── evaluate.py        # Métricas y análisis de errores
│   │   ├── predict.py         # Inferencia + ModelRegistry con caché
│   │   └── retrain.py         # Promoción segura: staging → comparación → producción
│   ├── api/
│   │   ├── main.py            # FastAPI endpoints
│   │   └── schemas.py         # Pydantic validation
│   └── utils/
│       ├── config.py          # Cargador de config.yaml
│       ├── logger.py          # Logger centralizado
│       └── seed.py            # Reproducibilidad global
│
├── tests/
│   ├── test_features.py       # Paridad train/serving del feature engineering
│   └── test_api.py            # Contrato HTTP de la API (mockeado)
│
├── logs/                      # Logs operacionales (no versionados)
├── .env.example               # Variables de entorno de ejemplo
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml             # Build system + config de herramientas
├── requirements.in            # Dependencias principales
├── requirements.txt           # Dependencias exactas (pip-compile)
└── README.md
```

---

## 🚀 Reproducir el proyecto

### Requisitos previos

```
Python 3.10+
Git
Docker (opcional, para deployment)
Cuenta en DagsHub (para MLflow remoto)
Cuenta en Kaggle (para el dataset)
```

### 1. Clonar el repositorio

```bash
git clone https://github.com/jorgesandovalpablo/demand-forecast.git
cd demand-forecast
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install -e .
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env con tus credenciales de DagsHub
```

```bash
# .env
MLFLOW_TRACKING_USERNAME=tu_usuario_dagshub
MLFLOW_TRACKING_PASSWORD=tu_token_dagshub
```

### 4. Descargar el dataset

Dataset disponible en Kaggle:
[Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

```bash
# Coloca los archivos en data/raw/
data/raw/
  ├── train.csv
  ├── test.csv
  ├── stores.csv
  ├── oil.csv
  ├── holidays_events.csv
  └── transactions.csv
```

### 5. Ejecutar el pipeline completo

> **Importante (v0.2):** el feature engineering ya NO es un paso
> independiente. `DemandFeatureEngineer` se ajusta (`fit()`) y aplica
> (`transform()`) dentro del entrenamiento, y se serializa junto al modelo.
> No ejecutes `build_features.py` manualmente — existe solo como puente
> deprecated.

```bash
# 1. Preprocessing (merge + nulos + log1p → train_processed.parquet)
python src/data/preprocessing.py

# 2. Entrenamiento (por horizonte)
#    - Ajusta y serializa DemandFeatureEngineer
#    - Walk-forward CV + modelo final
#    - Genera: lgbm_h{h}.pkl, features_h{h}.pkl, feature_pipeline_h{h}.pkl
#    Recomendado en Kaggle/Colab por RAM (16GB gratis)
python src/models/train.py --horizon 7
python src/models/train.py --horizon 30

# 3. Evaluación sobre el test set (últimas 8 semanas)
#    Reconstruye features vía el pipeline serializado
#    Genera métricas globales/por familia/por tienda en data/predictions/
python src/models/evaluate.py --horizon 7
python src/models/evaluate.py --horizon 30

# 4. Inferencia batch (simulación de forecast futuro)
#    Usa el pipeline serializado para calcular lags con paridad garantizada
python src/models/predict.py --horizon 7

# 5. Reentrenamiento con promoción segura (opcional)
#    Entrena a staging (_new), compara MAE vs producción:
#    - Mejora ≥ 1% → promueve los 3 artefactos (backup automático)
#    - No mejora   → descarta staging
python src/models/retrain.py --horizon 7            # decisión por métricas
python src/models/retrain.py --horizon 7 --force    # promoción forzada

# 6. Iniciar API
uvicorn src.api.main:app --reload --port 8000
```

**Orden obligatorio:** `preprocessing` → `train` → (`evaluate` | `predict` | `retrain`) → `API`.
Los modelos `.pkl` anteriores al pipeline stateful son **incompatibles**;
si actualizas desde v0.1 reentrena desde el paso 2.

### 6. Con Docker

```bash
# Construir imagen
docker build -t demand-forecast:latest .

# Levantar con docker-compose
docker-compose up -d

# Verificar
curl http://localhost:8000/health
```

### 7. Ejecutar tests

```bash
pytest tests/ -v --tb=short
```

---

## 🌐 API

Documentación interactiva disponible en:

```
http://localhost:8000/docs
```

### Endpoints

#### `GET /health`
Verifica el estado de la API y qué modelos están cargados.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "models_loaded": [7, 30],
  "version": "0.1.0"
}
```

#### `POST /predict`
Genera predicciones de demanda para una tienda y horizonte.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_nbr": 1,
    "horizon": 7,
    "family": 3
  }'
```

> `family` es el **código entero** del Label Encoding aprendido en
> entrenamiento (ver `categories_mapping` dentro del pipeline serializado).
> Si se omite, retorna todas las familias de la tienda.

```json
{
  "store_nbr": 1,
  "horizon": 7,
  "n_predictions": 7,
  "predictions": [
    {
      "date": "2017-09-01",
      "store_nbr": 1,
      "family": 3,
      "predicted_sales": 245.30,
      "lower_bound": 198.20,
      "upper_bound": 292.40
    }
  ]
}
```

#### `GET /metrics/{horizon}`
Retorna métricas del modelo en producción.

```bash
curl http://localhost:8000/metrics/7
```

#### `POST /retrain`
Dispara el reentrenamiento del modelo en background.

```bash
curl -X POST http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"horizon": 7, "force": false}'
```

---

## 🧠 Decisiones de diseño

### ¿Por qué dos modelos separados por horizonte?

Los horizontes de 7 y 30 días requieren features completamente distintas.
El lag mínimo seguro para el modelo diario es `lag_7`, mientras que para
el mensual es `lag_30`. Un modelo único mezclaría señales de corto y largo
plazo que confunden al algoritmo y producen data leakage en el horizonte mensual.

### ¿Por qué LightGBM global sobre Prophet o ARIMA?

Con 1,782 series temporales simultáneas los modelos univariados son inviables
en producción — requerirían entrenar y mantener 1,782 modelos individuales.
LightGBM con features de lag permite un modelo global que aprende patrones
compartidos entre tiendas y categorías, con mejor generalización.
El ganador de la competencia M5 (Walmart) usó exactamente esta arquitectura.

### ¿Por qué log1p en el target?

La distribución de ventas es fuertemente sesgada a la derecha.
Sin transformación el modelo dedica su capacidad de aprendizaje a los outliers
y falla en los casos comunes. `log1p` estabiliza la varianza, las predicciones
se revierten con `expm1()` al momento de servir resultados.

### ¿Por qué walk-forward CV y no K-Fold?

K-Fold aleatorio en series temporales introduce data leakage — el modelo
entrena con datos del futuro y valida con el pasado, produciendo métricas
artificialmente optimistas. Walk-forward respeta la causalidad temporal:
siempre entrenamos con el pasado y validamos con el futuro.

### ¿Por qué merge diferenciado de holidays?

El dataset de holidays tiene festivos nacionales, regionales y locales.
Un merge simple por fecha genera duplicados (53,460 filas detectadas en EDA).
La corrección usa merge diferenciado: nacional por `date`, regional por
`date + state`, local por `date + city`, con prioridad local > regional > nacional.

### ¿Por qué un pipeline de features stateful (fit/transform)?

En la v0.1 el feature engineering se recalculaba en cada etapa con
funciones independientes. Eso rompía la paridad train/serving:

- El Label Encoding recodificaba categorías según el subset recibido
  (al filtrar una tienda en la API, todo colapsaba a `0`).
- El ranking de tiendas se recalculaba sobre el subset (siempre rank 1).
- Las stats por tienda se computaban sobre la ventana de inferencia,
  no sobre el historial completo.

La solución es un único componente (`DemandFeatureEngineer`) que en
`fit()` congela vocabularios, stats y rankings sobre el histórico
completo, y en `transform()` los aplica rígidamente. Se serializa junto
al modelo, garantizando que serving y evaluación usen exactamente el
mismo estado que vio el entrenamiento. Cubierto por tests de paridad
(`tests/test_features.py`).

### ¿Por qué promoción segura en el retraining?

Entrenar directamente sobre producción implicaba que un modelo peor
igual sobrescribía los artefactos vigentes antes de comparar métricas.
Ahora `retrain.py` entrena a staging (`*_new.pkl`), evalúa sobre el
test set reservado, y solo si supera el modelo vigente (≥1% de mejora
en MAE) promueve los 3 artefactos con backup timestamped para rollback.

---

## 🔍 Hallazgos del EDA

### Dataset
- **3,000,888 filas** × 17 columnas (post corrección de merge)
- **54 tiendas**, **33 familias**, **1,782 series temporales**
- Período: `2013-01-01` → `2017-08-15` (4.5 años)

### Target (sales)
- Distribución fuertemente sesgada → se aplica `log1p`
- **31% de ceros** — concentrados en familias no core (BOOKS 97%, BABY CARE 94%)
- Tendencia creciente sostenida año sobre año
- Pico semanal: **Domingo**
- Pico anual: **Diciembre** (efecto navidad)

### Festivos
Los festivos NO impactan uniformemente las ventas:

| Grupo | Ejemplos | Impacto |
|---|---|---|
| Positivo | Navidad, Día de la Madre, Carnaval | +23% a +80% |
| Negativo | Año Nuevo, Traslados, Black Friday | -43% a -98% |
| Atípico | Terremoto Manabí 2016 | Comportamiento especial |

### Correlaciones externas
- **Precio del petróleo vs ventas:** `-0.75` (negativa fuerte)
- **Transacciones vs ventas:** `+0.837` (positiva fuerte)

### Bug corregido en EDA
Merge de holidays por `date` únicamente generaba **53,460 filas duplicadas**.
Corregido con merge diferenciado por locale (nacional/regional/local).

---

## 📈 Experimentos en MLflow

Experimentos trackeados en DagsHub:

```
https://dagshub.com/jorgesandovalpablo/demand-forecast
```

## ⚠️ Estado actual y limitaciones

Última actualización: 2026-08-25.

### Implementado y verificado
- Pipeline de features stateful con paridad train/serving testada (16 tests).
- Promoción segura en retraining (staging → comparación → producción con backups).
- `evaluate.py` reconstruyendo features desde el pipeline serializado.
- Intervalos de confianza calculados en escala log desde las stats históricas completas.

### Pendientes
| Ítem | Detalle |
|---|---|
| Métricas en README | Poblar tras ejecutar `evaluate.py` (guía: `docs/VALIDATION_GUIDE.md`) |
| Secrets de CI | Configurar `MLFLOW_TRACKING_USERNAME/PASSWORD` y opcionalmente `KAGGLE_USERNAME/KEY` + repo variable `KAGGLE_DOWNLOAD_ENABLED=true` para el retraining programado |
| Optuna | Declarado en requirements sin uso en `src/` |
| DVC | Declarado en requirements sin uso; datos/modelos fuera de git |
| SHAP | Import comentado en `evaluate.py`; reports/shap son de versiones anteriores |

### Limitaciones conocidas (deuda técnica aceptada)
- **OOM potencial en la API:** `main.py` carga el histórico completo
  (`train_processed.parquet`) en RAM. Válido como portafolio; en producción
  real requeriría una tienda de features externa (Redis/DB).
- **Inferencia por recálculo:** cada `/predict` reconstruye lags sobre el
  historial concatenado; no hay cache incremental de features.

---

## 📝 CHANGELOG

### v0.2.0 (2026-08-25)
- **Pipeline de features stateful:** `DemandFeatureEngineer` con patrón
  fit/transform; elimina la ruptura de paridad train/serving
  (label encoding destructivo, ranking colapsado, stats recalculadas).
- **Serialización del pipeline:** `feature_pipeline_h{h}.pkl` viaja junto al
  booster; serving y evaluación reconstruyen features con `.transform()`.
- **Promoción segura en retraining:** entrenamiento a staging (`_new`),
  comparación por MAE y rotación de los 3 artefactos con backups timestamped.
- **`evaluate.py` reparado:** ya no depende de parquets intermedios eliminados.
- **Intervalos de confianza corregidos:** cuantiles calculados en escala log
  a partir de `store_stats` completa (antes: std sobre ventana de 365 días).
- **Paridad de dtypes en serving:** `_reduce_memory` aplicado en rama predict.
- **Fix en API:** filtro por familia devolvía 500 en vez de 404.
- **Tests:** 16 tests (paridad de features + contrato HTTP) y linting limpio.
- Config externalizada: `model.top_families` movida a `config.yaml`.

### v0.1.0
- Pipeline completo de datos con corrección de merge de holidays
- Feature engineering diferenciado por horizonte (sin data leakage)
- Modelos diario y mensual con walk-forward CV
- MLflow tracking en DagsHub
- FastAPI con endpoints de predicción, métricas y retraining
- CI/CD con GitHub Actions
- Retraining automático semanal con comparación de métricas
- Docker + docker-compose para deployment reproducible

---

## 👤 Autor

**Jorge Sandoval**
- GitHub: [@jorgesandovalpablo](https://github.com/jorgesandovalpablo)

---

## 📄 Licencia

MIT License — ver [LICENSE](LICENSE)