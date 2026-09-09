# 🖥️ Demand Forecast — Dashboard interactivo

Demo interactiva del sistema de forecasting de demanda para 54 minimercados en
Ecuador.

**Dos versiones disponibles:**

| Versión | Archivo | Uso |
|---|---|---|
| **Local** | `dashboard/app.py` | Requiere `src/` instalado (paridad con API) |
| **Community Cloud** | `deploy/app.py` | Autocontenida, sin dependencias de `src/` |

## ✨ Funcionalidades

- **KPIs por horizonte:** MAE, WAPE y RMSE promediados por familia, con el
  WAPE global de test (10.51% para h7 / 12.34% para h30) como referencia.
- **Selector de tienda / horizonte / familia:** explora las 54 tiendas, los
  horizontes de 7 y 30 días, y las 33 familias de productos.
- **Gráfico interactivo (Plotly):** serie temporal de la predicción con
  intervalo de confianza (configurable via `confidence.z` en config.yaml),
  ventana de backtest (real vs predicción), o top 8 familias por volumen.
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

## ☁️ Streamlit Community Cloud

La demo en vivo corre en [Streamlit Community Cloud](https://share.streamlit.io).
El archivo `deploy/app.py` es autocontenido: lee assets pre-calculados
(parquets de predicciones, métricas, backtest) sin importar `src/`.

### Deploy

1. Push a `main` → auto-redeploy en ~1-2 min
2. URL: `https://demand-forecast-minimarket.streamlit.app`

### Regenerar assets

```bash
./venv/bin/python scripts/build_demo_bundle.py
git add deploy/assets/
git commit -m "chore: regenerar assets demo"
git push
```

## 🧩 Estructura

```
dashboard/
├── app.py             # Dashboard local (requiere src/)
└── requirements.txt   # Dependencias del dashboard local
deploy/
├── app.py             # Dashboard autocontenido (Community Cloud)
├── requirements.txt   # Dependencias fijas
└── assets/            # Assets pre-calculados (generados, no commitear modelos)
scripts/
└── build_demo_bundle.py  # Genera assets para deploy/
```
