"""
Tests de integración de la API FastAPI (serving).

Objetivo de MLOps: validar el CONTRATO HTTP sin depender de
modelos entrenados ni del dataset histórico (CI rápida y aislada).

Estrategia: se mockea `predict_by_store` (capa de inferencia) y el
estado en memoria `app_state`, ejercitando únicamente la lógica de
la API: validación de entrada, códigos de estado y shape de respuesta.
El lifespan no se ejecuta: los artefactos reales nunca se cargan.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.api.main import app, app_state


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """Cliente con estado simulado: datos disponibles y modelo h7 cargado."""
    monkeypatch.setitem(app_state, "historical_df", pd.DataFrame())
    monkeypatch.setitem(app_state, "models_loaded", [7])
    return TestClient(app)


def mock_predictions() -> pd.DataFrame:
    """
    Salida canónica de predict_by_store según el contrato interno:
    columnas [date, store_nbr, family, predicted_sales,
    lower_bound, upper_bound].
    """
    return pd.DataFrame({
        "date": [pd.Timestamp("2017-08-16"), pd.Timestamp("2017-08-17")],
        "store_nbr": [1, 1],
        "family": [3, 3],
        "predicted_sales": [10.0, 12.0],
        "lower_bound": [8.0, 9.0],
        "upper_bound": [12.0, 15.0],
    })


class TestHealth:
    """Endpoint de readiness para orquestación (Docker/K8s/CI)."""

    def test_health_ok(self, client):
        """Debe reportar healthy y reflejar los modelos precargados."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["models_loaded"] == [7]


class TestMetricsEndpoint:
    def test_invalid_horizon_rejected(self, client):
        """Solo se aceptan horizontes con modelo entrenado (7 o 30)."""
        assert client.get("/metrics/15").status_code == 400


class TestPredict:
    """Contrato del endpoint principal de inferencia."""

    def test_predict_ok(self, client, monkeypatch):
        """200 con payload completo cuando la inferencia es exitosa."""
        monkeypatch.setattr(
            "src.api.main.predict_by_store",
            lambda historical_df, horizon, store_nbr: mock_predictions(),
        )
        resp = client.post(
            "/predict",
            json={"store_nbr": 1, "horizon": 7},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_predictions"] == 2
        assert body["predictions"][0]["date"] == "2017-08-16"
        assert body["predictions"][0]["predicted_sales"] == 10.0

    def test_predict_without_data_returns_503(self, monkeypatch):
        """503 si el histórico no fue cargado en el startup."""
        monkeypatch.setitem(app_state, "historical_df", None)
        client = TestClient(app)
        resp = client.post(
            "/predict",
            json={"store_nbr": 1, "horizon": 7},
        )
        assert resp.status_code == 503

    def test_predict_with_unloaded_model_returns_503(self, monkeypatch):
        """503 si no hay modelo cargado para el horizonte solicitado."""
        monkeypatch.setitem(app_state, "historical_df", pd.DataFrame())
        monkeypatch.setitem(app_state, "models_loaded", [])
        client = TestClient(app)
        resp = client.post(
            "/predict",
            json={"store_nbr": 1, "horizon": 7},
        )
        assert resp.status_code == 503

    def test_family_filter_empty_returns_404(self, client, monkeypatch):
        """404 (no 500) si el filtro por familia no produce resultados.

        Regresión cubierta: HTTPException interna era capturada por el
        handler genérico y devuelta como error de servidor.
        """
        monkeypatch.setattr(
            "src.api.main.predict_by_store",
            lambda historical_df, horizon, store_nbr: mock_predictions(),
        )
        resp = client.post(
            "/predict",
            json={"store_nbr": 1, "horizon": 7, "family": 30},
        )
        assert resp.status_code == 404

    def test_unknown_store_raises_value_error_mapped_to_400(self, client, monkeypatch):
        """ValueError del dominio se mapea a 400 (petición inválida)."""
        def raise_missing(historical_df, horizon, store_nbr):
            raise ValueError(f"Tienda {store_nbr} no encontrada")

        monkeypatch.setattr("src.api.main.predict_by_store", raise_missing)
        resp = client.post(
            "/predict",
            json={"store_nbr": 3, "horizon": 7},
        )
        assert resp.status_code == 400


class TestLifespanCutoff:
    """El lifespan recorta el histórico a max_lag días para evitar OOM."""

    def test_lifespan_applies_temporal_cutoff(self, monkeypatch):
        """El parquet de ~730 días se reduce a ~365 tras el lifespan."""
        import asyncio
        from src.api.main import lifespan

        dates = pd.date_range("2015-01-01", "2017-08-15", freq="D")
        full_df = pd.DataFrame({
            "date": dates,
            "store_nbr": 1,
            "family": 1,
            "sales": 10.0,
        })

        saved = {
            "historical_df": app_state["historical_df"],
            "models_loaded": list(app_state["models_loaded"]),
        }

        try:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            monkeypatch.setattr(
                "src.api.main.Path", lambda x: mock_path_instance
            )
            monkeypatch.setattr(
                "src.api.main.pd.read_parquet", lambda p: full_df
            )
            monkeypatch.setattr(
                "src.api.main.ModelRegistry", MagicMock()
            )

            mock_app = MagicMock()

            async def run_lifespan():
                async with lifespan(mock_app):
                    pass

            asyncio.run(run_lifespan())

            result = app_state["historical_df"]
            assert result is not None
            assert len(result) < len(full_df)
            assert result["date"].min() >= (
                full_df["date"].max() - pd.Timedelta(days=365)
            )
        finally:
            app_state["historical_df"] = saved["historical_df"]
            app_state["models_loaded"] = saved["models_loaded"]
