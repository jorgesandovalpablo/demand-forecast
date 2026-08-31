# Baselines: Comparación Naive vs ML

Comparación del modelo LightGBM contra dos baselines triviales
para demostrar valor añadido del ML.

---

## Modelo Diario (h=7)

| Modelo | WAPE | MAE | RMSE |
|---|---|---|---|
| Naive (último valor) | 26.60% | 125.93 | 462.00 |
| Seasonal Naive | 25.39% | 120.20 | 445.03 |
| **LightGBM (Optuna)** | **10.51%** | **49.73** | **194.35** |

## Modelo Mensual (h=30)

| Modelo | WAPE | MAE | RMSE |
|---|---|---|---|
| Naive (último valor) | 26.60% | 125.93 | 462.00 |
| Seasonal Naive | 29.42% | 139.26 | 517.06 |
| **LightGBM (Optuna)** | **12.34%** | **58.41** | **218.19** |

---

## Resumen de mejora relativa

### h7
- ML vs Naive: **+60.5%** en WAPE (26.60% → 10.51%)
- ML vs SNaive: **+58.6%** en WAPE (25.39% → 10.51%)

### h30
- ML vs Naive: **+53.6%** en WAPE (26.60% → 12.34%)
- ML vs SNaive: **+58.1%** en WAPE (29.42% → 12.34%)

---

*Generado por `src/models/baselines.py`*