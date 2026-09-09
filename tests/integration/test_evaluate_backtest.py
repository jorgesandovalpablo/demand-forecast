"""Tests de integración: persistencia de predicciones backtest del test set.

Verifican parquets y artefactos que solo existen después de ejecutar
evaluate.py localmente. Corren solo con `pytest tests/integration/`.
"""
import pandas as pd

import joblib


def test_parquet_generado_columnas_correctas() -> None:
    """El parquet persistido por evaluate tiene las columnas esperadas."""
    df = pd.read_parquet(
        "data/predictions/backtest_predictions_h7.parquet"
    )
    expected_cols = [
        'date', 'store_nbr', 'family',
        'real_sales', 'y_pred_real', 'y_pred_log',
    ]
    assert list(df.columns) == expected_cols
    assert df['real_sales'].notna().all()
    assert df['y_pred_real'].notna().all()
    assert (df['real_sales'] >= 0).all()
    assert (df['y_pred_real'] >= 0).all()


def test_residual_std_persistido_estructura() -> None:
    """El archivo persistido por evaluate tiene la estructura esperada."""
    data = joblib.load("models/residual_std_h7.pkl")
    assert set(data.keys()) == {'global', 'df'}
    assert set(data['df'].columns) == {
        'store_nbr', 'family', 'resid_std'
    }
    assert data['global'] > 0.0
