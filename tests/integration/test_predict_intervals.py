"""Tests de integración: predicciones end-to-end por tienda.

Requiere modelo entrenado + datos procesados. Corren solo con
`pytest tests/integration/`.
"""
import pandas as pd

from src.models.predict import predict_by_store


def test_invariante_predict_by_store() -> None:
    """Invariante end-to-end: predicciones de tienda 1 con IC coherente."""
    historical = pd.read_parquet("data/processed/train_processed.parquet")
    for horizon in (7, 30):
        df = predict_by_store(historical, horizon=horizon, store_nbr=1)
        assert (df["upper_bound"] >= df["lower_bound"]).all()
        assert (df["lower_bound"] >= 0).all()
        assert (df["upper_bound"] >= 0).all()
