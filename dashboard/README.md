# 🖥️ Demand Forecast — Dashboard interactivo

Demo en vivo del sistema de forecasting de demanda para 54 minimercados en
Ecuador, publicada en **HuggingFace Spaces**.

El dashboard importa directamente la librería de predicción del pipeline
(`src/models/predict.py`), de modo que lo que se muestra es **exactamente el
mismo modelo** que sirve la API FastAPI. Paridad garantizada por diseño.

## ✨ Funcionalidades

- **KPIs por horizonte:** MAE, WAPE y RMSE promediados por familia, con el
  WAPE global de test (10.51% para h7 / 12.34% para h30) como referencia.
- **Selector de tienda / horizonte / familia:** explora las 54 tiendas, los
  horizontes de 7 y 30 días, y las 33 familias de productos.
- **Gráfico interactivo (Plotly):** serie temporal de la predicción con
  intervalo de confianza del 95%, o top 8 familias por volumen.
- **Tabla de detalle:** predicción por fecha con límites inferior y superior.

## 🚀 Ejecutar en local

```bash
python -m venv venv && source venv/bin/activate
pip install -r dashboard/requirements.txt

# Artefactos (modelos + datos) deben existir en el repo raíz:
#   models/*.pkl, data/processed/train_processed.parquet, data/raw/*.csv

streamlit run dashboard/app.py
```

Abre `http://localhost:8501`.

## ☁️ Publicar en HuggingFace Spaces

### 1. Crear el Space

- [huggingface.co/new-space](https://huggingface.co/new-space)
- SDK: **Streamlit** · Hardware: **CPU basic**
- Clónalo en local, p. ej. `~/demand-forecast-demo`.

### 2. Sincronizar código y artefactos

```bash
HF_SPACE_DIR=~/demand-forecast-demo bash scripts/sync_hf_space.sh
cd ~/demand-forecast-demo && git push
```

El script:
- Copia `src/`, `dashboard/`, `configs/` y los artefactos (modelos, datos).
- Pone `app.py` y `requirements.txt` en la raíz del Space (Streamlit lo espera así).
- Versiona los binarios grandes con **git LFS** (`models/*.pkl`, `*.parquet`, `*.csv`).

> Sin secretos: el Space es autocontenido, no necesita DagsHub/MLflow.

### 3. Alternativa: publicar con HF CLI

```bash
pip install -U huggingface_hub
export HF_TOKEN=tu_token
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id="TU_USUARIO/demand-forecast-demo",
    folder_path="~/demand-forecast-demo",
    repo_type="space",
)
EOF
```

> Los artefactos superan 10MB, por lo que conviene usar git LFS o
> `upload_large_files=True` en `upload_folder`.

## 🧩 Estructura

```
dashboard/
├── app.py             # Código Streamlit
└── requirements.txt   # Dependencias
scripts/
└── sync_hf_space.sh   # Sync del Space
```