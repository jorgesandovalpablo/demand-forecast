# dashboard/app.py
"""Dashboard Streamlit para el sistema de forecasting de demanda.

Permite explorar predicciones por tienda, horizonte y familia,
con intervalos de confianza, métricas de modelo y gráficos interactivos.
Reutiliza la librería de predicción del pipeline (`src/models/predict.py`).
"""
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.predict import ModelRegistry, predict_by_store
from src.utils.config import config

DATA_PROCESSED = Path("data/processed/train_processed.parquet")
METRICS_PATTERN = Path("data/predictions/global_metrics_h{horizon}.parquet")
BACKTEST_PATTERN = Path("data/predictions/backtest_predictions_h{horizon}.parquet")
HORIZONS = [7, 30]

st.set_page_config(
    page_title="Demand Forecast — Retail Ecuador",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource(show_spinner="Cargando datos y modelos...")
def load_assets() -> dict:
    """Carga históricos, modelos y mapeos de familia una sola vez."""
    historical = pd.read_parquet(DATA_PROCESSED)
    max_history = config["lags"]["max_lag"]
    cutoff = historical["date"].max() - pd.Timedelta(days=max_history)
    historical = historical[historical["date"] >= cutoff].copy()

    family_map: dict[int, str] = {}
    pipeline_path = Path("models/feature_pipeline_h7.pkl")
    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)
        cats = pipeline.categories_mapping.get("family", [])
        family_map = {i: str(name) for i, name in enumerate(cats)}

    metrics = {}
    for horizon in HORIZONS:
        path = METRICS_PATTERN.as_posix().format(horizon=horizon)
        p = Path(path)
        metrics[horizon] = (
            pd.read_parquet(p) if p.exists() else None
        )

    backtest = {}
    for horizon in HORIZONS:
        path = BACKTEST_PATTERN.as_posix().format(horizon=horizon)
        p = Path(path)
        backtest[horizon] = (
            pd.read_parquet(p) if p.exists() else None
        )

    for horizon in HORIZONS:
        ModelRegistry.load(horizon)

    return {
        "historical": historical,
        "family_map": family_map,
        "metrics": metrics,
        "backtest": backtest,
        "stores": sorted(historical["store_nbr"].unique().tolist()),
    }


@st.cache_data(show_spinner="Generando predicciones...")
def get_predictions(store_nbr: int, horizon: int) -> pd.DataFrame:
    """Predicciones determinísticas de una tienda+horizonte (cacheadas)."""
    return predict_by_store(
        load_assets()["historical"], horizon=horizon, store_nbr=store_nbr
    )


assets = load_assets()
family_map = assets["family_map"]
stores = assets["stores"]
default_store = stores[0] if stores else 1

st.title("🛒 Demand Forecast — Minimercados Ecuador")
st.caption(
    "Predicción de demanda con LightGBM para horizontes de 7 y 30 días. "
    "Fuente: Store Sales (Kaggle) · Modelo: LightGBM global · "
    "Tracking: MLflow / DagsHub."
)

with st.sidebar:
    st.header("Configuración")
    store_nbr = st.selectbox(
        "Tienda", options=stores, format_func=lambda s: f"Tienda {s}"
    )
    horizon = st.selectbox(
        "Horizonte",
        options=HORIZONS,
        format_func=lambda h: f"{h} días",
    )
    family_names = ["(Todas las familias)"] + sorted(family_map.values())
    selected = st.selectbox("Familia", options=family_names)
    st.divider()
    st.caption(f"Modelos cargados: {HORIZONS}")
    st.caption(
        "Demo sobre datos históricos reales. "
        "El modelo reconstruye los lags desde el historial de la tienda."
    )

# ── Métricas (KPIs) ───────────────────────────────────────────────────────
metrics_df = assets["metrics"].get(horizon)
kpi_cols = st.columns(3)
if metrics_df is not None and not metrics_df.empty:
    row = metrics_df.iloc[0]
    kpi_cols[0].metric("WAPE global (test)", f"{row['wape']:.2f}%")
    kpi_cols[1].metric("MAE (test)", f"{row['mae']:.2f}")
    kpi_cols[2].metric("RMSE (test)", f"{row['rmse']:.2f}")
    st.caption(
        f"Métricas globales sobre el test set — "
        f"`src/models/evaluate.py --horizon {horizon}`."
    )
else:
    for c in kpi_cols:
        c.metric("—", "n/d")
    st.warning(
        f"No se encontró `data/predictions/global_metrics_h{horizon}.parquet`. "
        f"Ejecuta `src/models/evaluate.py --horizon {horizon}`."
    )

# ── Predicciones ──────────────────────────────────────────────────────────
st.subheader("Predicciones de demanda")

try:
    predictions = get_predictions(store_nbr, horizon)
except Exception as exc:  # noqa: BLE001
    st.error(f"No se pudo generar la predicción: {exc}")
    st.stop()

predictions["family_name"] = predictions["family"].map(family_map).fillna(
    predictions["family"].astype(str)
)

family_code = None
if selected != "(Todas las familias)":
    family_code = next(
        (c for c, n in family_map.items() if n == selected), None
    )
    subset = predictions[predictions["family"] == family_code]
    if subset.empty:
        st.info(
            f"Familia '{selected}' sin ventas predichas para tienda {store_nbr}."
        )
        st.stop()
else:
    subset = predictions.copy()

left, right = st.columns([3, 2])

with left:
    if family_code is not None:
        st.markdown(f"#### {selected}")
        display = subset.sort_values("date")
        fig = go.Figure()

        bt = assets["backtest"].get(horizon)
        if bt is not None and not bt.empty:
            bt_fam = bt[
                (bt["store_nbr"] == store_nbr) &
                (bt["family"] == family_code)
            ].sort_values("date")
            if not bt_fam.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bt_fam["date"], y=bt_fam["real_sales"],
                        mode="lines+markers", name="Real (test)",
                        line=dict(color="#2ca02c", width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=bt_fam["date"], y=bt_fam["y_pred_real"],
                        mode="lines+markers", name="Backtest",
                        line=dict(color="#ff7f0e", width=2, dash="dash"),
                    )
                )

        fig.add_trace(
            go.Scatter(
                x=display["date"], y=display["predicted_sales"],
                mode="lines+markers", name="Predicción",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=display["date"], y=display["upper_bound"],
                mode="lines", line=dict(width=0), showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=display["date"], y=display["lower_bound"],
                mode="lines", line=dict(width=0), showlegend=False,
                fill="tonexty", fillcolor="rgba(31,119,180,0.2)",
                name="Intervalo de confianza",
            )
        )
        fig.add_vline(
            x=display["date"].min(),
            line_dash="dot", line_color="gray",
            annotation_text="Predicción inicia",
            annotation_position="top left",
        )
        fig.update_layout(
            height=420, margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Ventas",
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.markdown(
            "#### Top familias por volumen predicho"
        )
        top = (
            subset.groupby("family_name")["predicted_sales"]
            .sum().nlargest(8).reset_index()
        )
        bar = go.Figure(
            go.Bar(
                x=top["family_name"], y=top["predicted_sales"],
                marker_color="#1f77b4",
            )
        )
        bar.update_layout(
            height=420, margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Ventas predichas (total horizonte)",
        )
        st.plotly_chart(bar, width="stretch")

        st.markdown("#### Series agregadas (todas las familias)")
        bt = assets["backtest"].get(horizon)
        if bt is not None and not bt.empty:
            bt_store = bt[bt["store_nbr"] == store_nbr]
            bt_daily = (
                bt_store.groupby("date", as_index=False)[
                    ["real_sales", "y_pred_real"]
                ].sum()
            )
            fut_daily = (
                subset.groupby("date", as_index=False)[
                    ["predicted_sales"]
                ].sum()
            )
            fig_ts = go.Figure()
            fig_ts.add_trace(
                go.Scatter(
                    x=bt_daily["date"], y=bt_daily["real_sales"],
                    mode="lines+markers", name="Real (test)",
                    line=dict(color="#2ca02c", width=2),
                )
            )
            fig_ts.add_trace(
                go.Scatter(
                    x=bt_daily["date"], y=bt_daily["y_pred_real"],
                    mode="lines+markers", name="Backtest",
                    line=dict(color="#ff7f0e", width=2, dash="dash"),
                )
            )
            fig_ts.add_trace(
                go.Scatter(
                    x=fut_daily["date"], y=fut_daily["predicted_sales"],
                    mode="lines+markers", name="Predicción",
                    line=dict(color="#1f77b4", width=2),
                )
            )
            fig_ts.add_vline(
                x=fut_daily["date"].min(),
                line_dash="dot", line_color="gray",
                annotation_text="Predicción inicia",
                annotation_position="top left",
            )
            fig_ts.update_layout(
                height=420, margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title="Ventas (agregado por día)",
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_ts, width="stretch")
        else:
            st.info(
                "No se encontró `data/predictions/"
                "backtest_predictions_h{horizon}.parquet`. "
                "Ejecuta `src/models/evaluate.py "
                "--horizon {horizon}` para generar la comparación."
            )

with right:
    st.markdown("#### Detalle")
    detail = subset[["date", "family_name", "predicted_sales",
                     "lower_bound", "upper_bound"]].copy()
    detail["date"] = detail["date"].dt.strftime("%Y-%m-%d")
    detail.columns = [
        "Fecha", "Familia", "Predicción", "Límite inf.", "Límite sup."
    ]
    st.dataframe(
        detail.sort_values("Fecha"),
        width="stretch", hide_index=True, height=420,
    )

st.divider()
st.caption(
    "La ventana de 8 semanas previa a la predicción corresponde al test set "
    "de `evaluate.py`: se muestran las ventas reales y la predicción backtest "
    "del modelo sobre ese mismo periodo. "
    "Los intervalos de confianza del 95% se calculan en escala log a partir "
    "de la desviación histórica de cada tienda-familia. "
    "La predicción es determinística dado el historial; "
    "los resultados se cachean por tienda y horizonte."
)
st.caption(
    "Los picos en la predicción (p. ej. 16-ago y días cercanos) se alinean "
    "con los valores reales de la columna `onpromotion` del dataset: "
    "días con muchas familias en promoción generan predicciones más altas, "
    "reflejando el patrón real del periodo."
)