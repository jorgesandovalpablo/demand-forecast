# tests/test_predict_intervals.py
"""Tests unitarios del intervalo de confianza (fix punto 1).

El helper `_build_confidence_intervals` es una función pura, así que se
testea sin artefactos y corre en CI.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.models.predict import _build_confidence_intervals


def test_esporadica_upper_no_negativo() -> None:
    """Familia esporádica: predicción log muy negativa → 0 <= lower <= upper."""
    y_pred_log = np.array([-1.0, -4.0, -8.0])
    std = np.array([2.0, 2.0, 2.0])
    lower, upper = _build_confidence_intervals(y_pred_log, std)
    assert np.all(lower >= 0)
    assert np.all(upper >= 0)
    assert np.all(upper >= lower)
    # Con log tan negativo, el punto inferior colapsa a 0 y upper es 0
    assert upper[-1] == 0.0


def test_normal_lower_menor_upper() -> None:
    """Caso normal: 0 <= lower < upper."""
    y_pred_log = np.array([np.log1p(50.0)])
    std = np.array([0.5])
    lower, upper = _build_confidence_intervals(y_pred_log, std)
    assert 0 <= lower[0] < upper[0]


def test_redondeo_2_decimales() -> None:
    """Ambos extremos redondeados a 2 decimales."""
    y_pred_log = np.array([np.log1p(100.0)])
    std = np.array([0.3])
    lower, upper = _build_confidence_intervals(y_pred_log, std)
    for arr in (lower, upper):
        assert np.all(np.round(arr, 2) == arr)


def test_upper_respeto_lower_explícito() -> None:
    """upper nunca menor que lower aunque la predicción sea 0."""
    y_pred_log = np.array([-10.0])
    std = np.array([0.0])
    lower, upper = _build_confidence_intervals(y_pred_log, std)
    assert lower[0] == 0.0 and upper[0] == 0.0


def test_simetrico_por_defecto() -> None:
    """Por defecto (upper_factor=1.0) el intervalo es simétrico en log."""
    mu = np.log1p(100.0)
    std = np.array([0.5])
    lower, upper = _build_confidence_intervals(np.array([mu]), std)
    # En log, la distancia del punto a cada extremo debe ser la misma.
    d_low = mu - np.log1p(lower[0])
    d_up = np.log1p(upper[0]) - mu
    # Tolerancia por el redondeo a 2 decimales interno de la función.
    np.testing.assert_allclose(d_low, d_up, atol=0.01)


@pytest.mark.skipif(
    not Path("models/lgbm_h7.pkl").exists(),
    reason="Artefactos de modelo no presentes",
)
def test_invariante_predict_by_store() -> None:
    """Invariante end-to-end: predicciones de tienda 1 con IC coherente."""
    from src.models.predict import predict_by_store

    historical = pd.read_parquet("data/processed/train_processed.parquet")
    for horizon in (7, 30):
        df = predict_by_store(historical, horizon=horizon, store_nbr=1)
        assert (df["upper_bound"] >= df["lower_bound"]).all()
        assert (df["lower_bound"] >= 0).all()
        assert (df["upper_bound"] >= 0).all()