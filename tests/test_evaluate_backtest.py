# tests/test_evaluate_backtest.py
"""Tests de la persistencia de predicciones backtest del test set.

`build_backtest_df` es una función pura, así que se testea sin artefactos.
Un test adicional verifica el parquet generado por evaluate (skip si no hay
data/predictions).
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.models.evaluate import build_backtest_df, compute_residual_std


def test_backtest_columnas_y_escalas() -> None:
    """Genera el backtest y valida columnas y escalas (real vs log)."""
    test_df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'store_nbr': [1, 1],
        'family': [10, 10],
        'sales': np.log1p([100.0, 250.0]),  # target en escala log
    })
    y_pred_log = np.array([np.log1p(110.0), np.log1p(230.0)])

    result = build_backtest_df(test_df, y_pred_log, target='sales')

    expected_cols = [
        'date', 'store_nbr', 'family',
        'real_sales', 'y_pred_real', 'y_pred_log',
    ]
    assert list(result.columns) == expected_cols

    np.testing.assert_allclose(result['real_sales'], [100.0, 250.0])
    np.testing.assert_allclose(result['y_pred_real'], [110.0, 230.0])
    np.testing.assert_allclose(result['y_pred_log'], y_pred_log)


def test_backtest_preserva_claves_agrupacion() -> None:
    """Las columnas date/store_nbr/family se conservan sin alterar."""
    test_df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01']),
        'store_nbr': [3],
        'family': [22],
        'sales': [np.log1p(50.0)],
    })
    result = build_backtest_df(test_df, np.array([np.log1p(60.0)]), 'sales')
    assert result['store_nbr'].iloc[0] == 3
    assert result['family'].iloc[0] == 22
    assert result['date'].iloc[0] == pd.Timestamp('2024-01-01')


@pytest.mark.skipif(
    not Path("data/predictions/backtest_predictions_h7.parquet").exists(),
    reason="Parquet de backtest no presente",
)
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


def test_compute_residual_std_valores() -> None:
    """compute_residual_std calcula el std residual por grupo y global."""
    rng = np.random.default_rng(42)
    n = 20
    bt = pd.DataFrame({
        'store_nbr': np.repeat([1, 2], n),
        'family': np.repeat([10, 20], n),
        'y_pred_log': np.concatenate([np.zeros(n), np.ones(n)]),
        'real_sales': np.concatenate([
            np.expm1(rng.normal(0.0, 0.5, n)),
            np.expm1(rng.normal(1.0, 0.2, n)),
        ]),
    })
    result = compute_residual_std(bt)
    assert set(result.keys()) == {'global', 'df'}
    assert set(result['df'].columns) == {
        'store_nbr', 'family', 'resid_std'
    }
    assert len(result['df']) == 2
    assert result['global'] > 0.0
    # El grupo con mayor dispersión de residuos debe tener mayor std.
    stds = result['df'].sort_values('store_nbr')['resid_std'].values
    assert stds[0] > stds[1]


def test_compute_residual_std_cero_residuo() -> None:
    """Residuo nulo → std 0 (intervalo degenerado, sin ruido)."""
    bt = pd.DataFrame({
        'store_nbr': [1, 1, 1],
        'family': [10, 10, 10],
        'y_pred_log': np.log1p([100.0, 101.0, 99.0]),
        'real_sales': [100.0, 101.0, 99.0],
    })
    result = compute_residual_std(bt)
    assert result['global'] == 0.0
    assert result['df']['resid_std'].iloc[0] == 0.0


@pytest.mark.skipif(
    not Path("models/residual_std_h7.pkl").exists(),
    reason="Artefacto residual_std_h7 no presente",
)
def test_residual_std_persistido_estructura() -> None:
    """El archivo persistido por evaluate tiene la estructura esperada."""
    import joblib

    data = joblib.load("models/residual_std_h7.pkl")
    assert set(data.keys()) == {'global', 'df'}
    assert set(data['df'].columns) == {
        'store_nbr', 'family', 'resid_std'
    }
    assert data['global'] > 0.0