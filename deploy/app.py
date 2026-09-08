# deploy/app.py
"""Dashboard Streamlit autocontenido para Streamlit Community Cloud.

Lee archivos pre-calculados de deploy/assets/ (sin imports de src/).
UI 1:1 con dashboard/app.py: sidebar, KPIs, vista familia con
backtest, vista agregada, dataframe detalle, captions en espanol.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
PRED_DIR = ASSETS / "predictions"
MODELS_DIR = ASSETS / "models"
HORIZONS = [7, 30]

st.set_page_config(
    page_title="Demand Forecast - Retail Ecuador",
    page_icon="\U0001f6d2",
    layout="wide",
)


@st.cache_data
def _load_stores() -> list[int]:
    with open(ASSETS / "stores.json") as f:
        return json.load(f)


@st.cache_data
def _load_families() -> dict[str, str]:
    with open(ASSETS / "families.json") as f:
        return json.load(f)


@st.cache_data
def _load_metrics(horizon: int) -> pd.DataFrame | None:
    p = ASSETS / f"global_metrics_h{horizon}.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def _load_backtest(horizon: int) -> pd.DataFrame | None:
    p = ASSETS / f"backtest_predictions_h{horizon}.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def _load_predictions(store_nbr: int, horizon: int) -> pd.DataFrame | None:
    p = PRED_DIR / f"store_{store_nbr}_h{horizon}.parquet"
    return pd.read_parquet(p) if p.exists() else None


stores = _load_stores()
family_map = _load_families()
default_store = stores[0] if stores else 1

st.title("\U0001f6d2 Demand Forecast - Minimercados Ecuador")
st.caption(
    "Prediccion de demanda con LightGBM para horizontes de 7 y 30 dias. "
    "Fuente: Store Sales (Kaggle) \u00b7 Modelo: LightGBM global \u00b7 "
    "Tracking: MLflow / DagsHub."
)

with st.sidebar:
    st.header("Configuracion")
    store_nbr = st.selectbox(
        "Tienda", options=stores, format_func=lambda s: f"Tienda {s}"
    )
    horizon = st.selectbox(
        "Horizonte",
        options=HORIZONS,
        format_func=lambda h: f"{h} dias",
    )
    family_names = ["(Todas las familias)"] + sorted(family_map.values())
    selected = st.selectbox("Familia", options=family_names)
    st.divider()
    st.caption(f"Modelos cargados: {len(HORIZONS)}")
    st.caption(
        "Demo sobre datos historicos reales. "
        "El modelo reconstruye los lags desde el historial de la tienda."
    )

# -- Metricas (KPIs) --
metrics_df = _load_metrics(horizon)
kpi_cols = st.columns(3)
if metrics_df is not None and not metrics_df.empty:
    row = metrics_df.iloc[0]
    kpi_cols[0].metric("WAPE global (test)", f"{row['wape']:.2f}%")
    kpi_cols[1].metric("MAE (test)", f"{row['mae']:.2f}")
    kpi_cols[2].metric("RMSE (test)", f"{row['rmse']:.2f}")
    st.caption(
        f"Metricas globales sobre el test set \u2014 "
        f"`src/models/evaluate.py --horizon {horizon}`."
    )
else:
    for c in kpi_cols:
        c.metric("\u2014", "n/d")
    st.warning(
        f"No se encontro `global_metrics_h{horizon}.parquet`. "
        f"Ejecuta `src/models/evaluate.py --horizon {horizon}`."
    )

# -- Predicciones --
st.subheader("Predicciones de demanda")

predictions = _load_predictions(store_nbr, horizon)
if predictions is None or predictions.empty:
    st.warning(
        f"No hay predicciones pre-calculadas para tienda {store_nbr}, "
        f"horizonte {horizon}. Ejecuta `scripts/build_demo_bundle.py`."
    )
    st.stop()

predictions["family_name"] = predictions["family"].map(family_map).fillna(
    predictions["family"].astype(str)
)

family_code = None
if selected != "(Todas las familias)":
    family_code = next(
        (int(c) for c, n in family_map.items() if n == selected), None
    )
    subset = predictions[predictions["family"] == family_code]
    if subset.empty:
        st.info(
            f"Familia '{selected}' sin ventas predichas para tienda {store_nbr}."
        )
        st.stop()
else:
    subset = predictions.copy()

if family_code is not None:
    st.markdown(f"#### {selected}")
    display = subset.sort_values("date")
    fig = go.Figure()

    bt = _load_backtest(horizon)
    if bt is not None and not bt.empty:
        bt_fam = bt[
            (bt["store_nbr"] == store_nbr)
            & (bt["family"] == family_code)
        ].sort_values("date")
        if not bt_fam.empty:
            fig.add_vrect(
                x0=bt_fam["date"].min(),
                x1=display["date"].min(),
                fillcolor="gray",
                opacity=0.10,
                layer="below",
                line_width=0,
            )
            fig.add_trace(
                go.Scatter(
                    x=bt_fam["date"],
                    y=bt_fam["real_sales"],
                    mode="lines+markers",
                    name="Real (test)",
                    line=dict(color="#37474f", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bt_fam["date"],
                    y=bt_fam["y_pred_real"],
                    mode="lines+markers",
                    name="Backtest",
                    line=dict(color="#ff7f0e", width=2, dash="dash"),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=display["date"],
            y=display["predicted_sales"],
            mode="lines+markers",
            name="Prediccion",
            line=dict(color="#1565c0", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=display["date"],
            y=display["upper_bound"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=display["date"],
            y=display["lower_bound"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(158,202,225,0.20)",
            hoverinfo="skip",
        )
    )
    fig.add_vline(
        x=display["date"].min(),
        line_dash="dot",
        line_color="gray",
        annotation_text="Prediccion inicia",
        annotation_position="top left",
    )
    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Ventas",
        legend=dict(orientation="h", y=1.02, font=dict(size=16)),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Detalle")
    detail = subset[
        ["date", "family_name", "predicted_sales", "lower_bound", "upper_bound"]
    ].copy()
    detail["date"] = detail["date"].dt.strftime("%Y-%m-%d")
    detail.columns = [
        "Fecha", "Familia", "Prediccion", "Limite inf.", "Limite sup."
    ]
    st.dataframe(
        detail.sort_values("Fecha"),
        width="stretch",
        hide_index=True,
        height=420,
    )
else:
    st.markdown("#### Top familias por volumen predicho")
    top = (
        subset.groupby("family_name")["predicted_sales"]
        .sum()
        .nlargest(8)
        .reset_index()
    )
    bar = go.Figure(
        go.Bar(
            x=top["family_name"],
            y=top["predicted_sales"],
            marker_color="#1565c0",
        )
    )
    bar.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Ventas predichas (total horizonte)",
    )
    st.plotly_chart(bar, width="stretch")

    st.markdown("#### Series agregadas (todas las familias)")
    bt = _load_backtest(horizon)
    if bt is not None and not bt.empty:
        bt_store = bt[bt["store_nbr"] == store_nbr]
        bt_daily = (
            bt_store.groupby("date", as_index=False)[
                ["real_sales", "y_pred_real"]
            ].sum()
        )
        fut_daily = (
            subset.groupby("date", as_index=False)[
                ["predicted_sales", "lower_bound", "upper_bound"]
            ].sum()
        )
        fig_ts = go.Figure()
        fig_ts.add_vrect(
            x0=bt_daily["date"].min(),
            x1=fut_daily["date"].min(),
            fillcolor="gray",
            opacity=0.10,
            layer="below",
            line_width=0,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=bt_daily["date"],
                y=bt_daily["real_sales"],
                mode="lines+markers",
                name="Real (test)",
                line=dict(color="#37474f", width=2),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=bt_daily["date"],
                y=bt_daily["y_pred_real"],
                mode="lines+markers",
                name="Backtest",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=fut_daily["date"],
                y=fut_daily["predicted_sales"],
                mode="lines+markers",
                name="Prediccion",
                line=dict(color="#1565c0", width=2),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=fut_daily["date"],
                y=fut_daily["upper_bound"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=fut_daily["date"],
                y=fut_daily["lower_bound"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                fill="tonexty",
                fillcolor="rgba(158,202,225,0.20)",
                hoverinfo="skip",
            )
        )
        fig_ts.add_vline(
            x=fut_daily["date"].min(),
            line_dash="dot",
            line_color="gray",
            annotation_text="Prediccion inicia",
            annotation_position="top left",
        )
        fig_ts.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Ventas (agregado por dia)",
            legend=dict(orientation="h", y=1.02, font=dict(size=16)),
        )
        st.plotly_chart(fig_ts, width="stretch")
    else:
        st.info(
            "No se encontro backtest_predictions para esta tienda. "
            "Ejecuta `src/models/evaluate.py --horizon {horizon}` "
            "para generar la comparacion."
        )

st.divider()
st.caption(
    "La ventana de 8 semanas previa a la prediccion corresponde al test set "
    "de `evaluate.py`: se muestran las ventas reales y la prediccion backtest "
    "del modelo sobre ese mismo periodo. "
    "Los intervalos de confianza se calculan en escala log a partir "
    "de la desviacion historica de cada tienda-familia. "
    "La prediccion es deterministica dado el historial; "
    "los resultados se cachean por tienda y horizonte."
)
st.caption(
    "Los picos en la prediccion (p. ej. 16-ago y dias cercanos) se alinean "
    "con los valores reales de la columna `onpromotion` del dataset: "
    "dias con muchas familias en promocion generan predicciones mas altas, "
    "reflejando el patron real del periodo."
)
