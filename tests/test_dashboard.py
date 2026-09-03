# tests/test_dashboard.py
"""Tests del dashboard Streamlit (ventana backtest).

Requiere los artefactos (modelos, histórico y parquet de backtest) y
`streamlit.testing.v1.AppTest`. Se saltan si faltan los artefactos.
"""
import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(
    Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
)


@pytest.mark.skipif(
    not Path("models/lgbm_h7.pkl").exists()
    or not Path("data/processed/train_processed.parquet").exists()
    or not Path("data/predictions/backtest_predictions_h7.parquet").exists(),
    reason="Artefactos de modelo/datos/backtest no presentes",
)
def test_dashboard_todas_sin_excepciones() -> None:
    """La vista '(Todas las familias)' renderiza y muestra la serie agregada."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    markdowns = [m.value for m in at.markdown]
    assert any("Series agregadas" in m for m in markdowns)


@pytest.mark.skipif(
    not Path("models/lgbm_h7.pkl").exists()
    or not Path("data/processed/train_processed.parquet").exists()
    or not Path("data/predictions/backtest_predictions_h7.parquet").exists(),
    reason="Artefactos de modelo/datos/backtest no presentes",
)
def test_dashboard_familia_sin_excepciones() -> None:
    """Seleccionar una familia renderiza el gráfico con ventana backtest."""
    family_map = _load_family_map()
    if not family_map:
        pytest.skip("No se pudo cargar el family_map")

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    # Sidebar: selectboxes tienda(0), horizonte(1), familia(2)
    first_family_name = sorted(family_map.values())[0]
    at.selectbox[2].select(first_family_name).run()
    assert not at.exception


def _load_family_map() -> dict:
    import joblib

    path = Path("models/feature_pipeline_h7.pkl")
    if not path.exists():
        return {}
    pipeline = joblib.load(path)
    cats = pipeline.categories_mapping.get("family", [])
    return {i: str(name) for i, name in enumerate(cats)}