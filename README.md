# demand-
# 🛒 Demand Forecast — Retail Time Series

> Pipeline de ML end-to-end para predicción de demanda en minimercados de Ecuador.
> Modelos LightGBM con horizontes de 7 y 30 días, MLflow tracking en DagsHub,
> FastAPI deployment y retraining automático semanal vía GitHub Actions.

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
        ├─────────────────────────────────┐
        ▼                                 ▼
build_features(horizon=7)     build_features(horizon=30)
  lags: 7,14,21,28,364          lags: 30,60,90,364
  rolling: 7,14,28              rolling: 30,60,90
        │                                 │
        ▼                                 ▼
train_features_d7.parquet    train_features_m30.parquet
        │                                 │
        ▼                                 ▼
validation.py                  validation.py
walk-forward CV (5 folds)      walk-forward CV (5 folds)
        │                                 │
        ▼                                 ▼
train.py + MLflow              train.py + MLflow
lgbm_h7.pkl                    lgbm_h30.pkl
        │                                 │
        └─────────────┬───────────────────┘
                      ▼
              predict.py
              evaluate.py
                      │
                      ▼
              FastAPI (main.py)
              POST /predict
```

**Pipeline de reentrenamiento automático:**

```
GitHub Actions (cron semanal)
        │
        ▼
retrain.py
  ├── Ejecuta pipeline completo
  ├── Entrena nuevo modelo
  ├── Compara métricas (threshold 1% mejora)
  ├── Si mejora → reemplaza modelo en producción
  └── Si no mejora → mantiene modelo anterior
              │
              ▼
        MLflow registra versión
        Backup del modelo anterior
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
| **CI/CD** | GitHub Actions | Tests automáticos y retraining |
| **Optimización** | Optuna | Búsqueda de hiperparámetros |
| **Versionado datos** | DVC | Datos y modelos grandes |
| **Dependencias** | pip-tools | Versiones exactas y reproducibles |
| **Calidad** | Black + Flake8 + isort | Estilo y linting automático |
| **Tests** | pytest | Tests unitarios y de integración |

---

## 📊 Resultados

### Modelo Diario (horizon=7 días)

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

> 📌 Métricas se actualizarán tras el entrenamiento completo en Kaggle Notebooks.

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
│       ├── ci.yml              # Tests en cada push
│       └── retrain.yml         # Retraining automático semanal
│
├── configs/
│   └── config.yaml             # Fuente de verdad única del proyecto
│
├── data/
│   ├── raw/                    # CSV originales — nunca se modifican
│   ├── processed/              # Parquet generados por el pipeline
│   └── predictions/            # Outputs del modelo
│
├── models/                     # Modelos entrenados (.pkl)
│   ├── lgbm_h7.pkl
│   ├── lgbm_h30.pkl
│   ├── features_h7.pkl         # Features exactas de entrenamiento
│   ├── features_h30.pkl
│   ├── store_stats_h7.pkl      # Estadísticas históricas por tienda
│   └── store_stats_h30.pkl
│
├── notebooks/
│   └── 01_eda.ipynb            # Análisis exploratorio completo
│
├── src/
│   ├── data/
│   │   ├── ingestion.py        # Carga y validación de esquema
│   │   └── preprocessing.py   # Merge, nulos, log1p, memoria
│   ├── features/
│   │   └── build_features.py  # ~50 features por horizonte
│   ├── models/
│   │   ├── validation.py      # Walk-forward cross validation
│   │   ├── train.py           # LightGBM + MLflow tracking
│   │   ├── evaluate.py        # Métricas y análisis de errores
│   │   ├── predict.py         # Inferencia + test set oficial
│   │   └── retrain.py         # Pipeline de reentrenamiento
│   ├── api/
│   │   ├── main.py            # FastAPI endpoints
│   │   └── schemas.py         # Pydantic validation
│   └── utils/
│       ├── config.py          # Cargador de config.yaml
│       ├── logger.py          # Logger centralizado
│       └── seed.py            # Reproducibilidad global
│
├── tests/
│   ├── test_features.py       # Tests de feature engineering
│   └── test_api.py            # Tests de endpoints
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

```bash
# 1. Preprocessing
python src/data/preprocessing.py

# 2. Feature engineering
python src/features/build_features.py --horizon 7
python src/features/build_features.py --horizon 30

# 3. Entrenamiento
# Recomendado ejecutar en Kaggle Notebooks o Google Colab
# por capacidad de cómputo (16GB RAM disponibles gratis)
python src/models/train.py --horizon 7
python src/models/train.py --horizon 30

# 4. Evaluación
python src/models/evaluate.py --horizon 7
python src/models/evaluate.py --horizon 30

# 5. Predicción sobre test set oficial
python src/models/predict.py --horizon 7 --mode test

# 6. Iniciar API
uvicorn src.api.main:app --reload --port 8000
```

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
    "family": "GROCERY I"
  }'
```

```json
{
  "store_nbr": 1,
  "horizon": 7,
  "n_predictions": 7,
  "predictions": [
    {
      "date": "2017-09-01",
      "store_nbr": 1,
      "family": "GROCERY I",
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

### ¿Por qué guardar features y store stats junto al modelo?

Durante predicción el dataset nuevo no tiene el mismo historial que el de
entrenamiento. Las store stats (`venta_media_historica`, `venta_std_historica`)
deben calcularse sobre el historial completo de entrenamiento, no sobre los
datos de inferencia. Guardándolas junto al modelo se garantiza consistencia
entre entrenamiento y predicción.

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

---

## 📝 CHANGELOG

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