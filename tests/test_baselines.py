"""
Tests unitarios del módulo de baselines.

Estrategia: datos sintéticos para validar la lógica de
Naive y Seasonal Naive sin depender del dataset completo.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.models.baselines import (
    prepare_test_set,
    naive_last_value,
    seasonal_naive,
    generate_report,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame sintético con 3 tiendas, 2 familias, 90 días."""
    dates = pd.date_range("2017-01-01", periods=90, freq="D")
    stores = [1, 2, 3]
    families = ["A", "B"]

    rows = []
    for d in dates:
        for s in stores:
            for f in families:
                rows.append({
                    "date": d,
                    "store_nbr": s,
                    "family": f,
                    "sales": np.log1p(
                        10 + s * 5 + (d.dayofweek % 7) * 2
                    ),
                })

    return pd.DataFrame(rows)


@pytest.fixture
def split_df(sample_df) -> tuple:
    """Train/test split usando la misma lógica que evaluate.py."""
    target = "sales"
    group_cols = ["store_nbr", "family"]
    test_weeks = 2

    cutoff = sample_df["date"].max() - pd.Timedelta(weeks=test_weeks)
    train_df = sample_df[sample_df["date"] <= cutoff].copy()
    test_df = sample_df[sample_df["date"] > cutoff].copy()

    return train_df, test_df, target, group_cols


class TestPrepareTestSet:

    @patch("src.models.baselines.config", {
        "training": {"test_size_weeks": 2},
        "data": {"target": "sales", "group_cols": ["store_nbr", "family"]},
    })
    def test_split_ratio(self, sample_df):
        train_df, test_df, target, group_cols = prepare_test_set(
            sample_df
        )
        assert target == "sales"
        assert group_cols == ["store_nbr", "family"]
        assert len(test_df) > 0
        assert len(train_df) > len(test_df)
        assert test_df["date"].min() > train_df["date"].max()


class TestNaiveLastValue:

    def test_returns_last_value_per_series(self, split_df):
        train_df, test_df, target, group_cols = split_df
        preds = naive_last_value(train_df, test_df, target, group_cols)

        assert len(preds) == len(test_df)
        assert not np.any(np.isnan(preds))

        # Cada predicción debe ser el último valor de train para esa serie
        for (_, row), pred in zip(test_df.iterrows(), preds):
            last_val = train_df.loc[
                (train_df["store_nbr"] == row["store_nbr"])
                & (train_df["family"] == row["family"]),
                target,
            ].iloc[-1]
            assert abs(pred - last_val) < 1e-6

    def test_matches_naive_logic(self, split_df):
        train_df, test_df, target, group_cols = split_df
        preds = naive_last_value(train_df, test_df, target, group_cols)

        # Para una serie específica, verificar contra cálculo manual
        store, fam = test_df.iloc[0][["store_nbr", "family"]]
        last = train_df.loc[
            (train_df["store_nbr"] == store)
            & (train_df["family"] == fam),
            target,
        ].iloc[-1]

        mask = (test_df["store_nbr"] == store) & (
            test_df["family"] == fam
        )
        series_preds = preds[mask.values]
        assert np.all(series_preds == last)


class TestSeasonalNaive:

    def test_h7_returns_array(self, split_df):
        train_df, test_df, target, group_cols = split_df
        preds = seasonal_naive(
            train_df, test_df, target, group_cols, horizon=7
        )

        assert len(preds) == len(test_df)
        assert not np.any(np.isnan(preds))

    def test_h30_returns_array(self, split_df):
        train_df, test_df, target, group_cols = split_df
        preds = seasonal_naive(
            train_df, test_df, target, group_cols, horizon=30
        )

        assert len(preds) == len(test_df)
        assert not np.any(np.isnan(preds))

    def test_h7_matches_7day_lag(self, split_df):
        train_df, test_df, target, group_cols = split_df
        preds = seasonal_naive(
            train_df, test_df, target, group_cols, horizon=7
        )

        # Para una fecha específica, verificar que usa el valor de hace 7d
        row0 = test_df.iloc[0]
        ref_date = row0["date"] - pd.Timedelta(days=7)
        match = train_df.loc[
            (train_df["store_nbr"] == row0["store_nbr"])
            & (train_df["family"] == row0["family"])
            & (train_df["date"] == ref_date),
            target,
        ]

        if len(match) > 0:
            assert abs(preds[0] - match.values[0]) < 1e-6

    def test_fallback_to_last_value(self):
        """Cuando no hay match de fecha, usa el último valor."""
        dates_train = pd.date_range("2017-01-01", periods=30, freq="D")
        dates_test = pd.date_range("2017-06-01", periods=7, freq="D")

        train_df = pd.DataFrame({
            "date": list(dates_train) * 2,
            "store_nbr": [1] * 30 + [2] * 30,
            "family": "A",
            "sales": np.arange(60, dtype=float),
        })
        test_df = pd.DataFrame({
            "date": list(dates_test) * 2,
            "store_nbr": [1] * 7 + [2] * 7,
            "family": "A",
            "sales": 0.0,
        })

        preds = seasonal_naive(
            train_df, test_df, "sales", ["store_nbr", "family"], 7
        )

        # Sin match para junio → fallback al último valor de cada serie
        last_store1 = train_df.loc[
            train_df["store_nbr"] == 1, "sales"
        ].iloc[-1]
        last_store2 = train_df.loc[
            train_df["store_nbr"] == 2, "sales"
        ].iloc[-1]

        assert np.all(preds[:7] == last_store1)
        assert np.all(preds[7:] == last_store2)


class TestGenerateReport:

    def test_produces_valid_markdown(self):
        results = [
            {
                "horizon": 7,
                "naive": {
                    "wape": 26.60, "mae": 125.93, "rmse": 462.00
                },
                "snaive": {
                    "wape": 25.39, "mae": 120.20, "rmse": 445.03
                },
            }
        ]

        report = generate_report(results)

        assert "# Baselines" in report
        assert "## Modelo Diario (h=7)" in report
        assert "26.60%" in report
        assert "10.51%" in report
        assert "| Naive" in report
        assert "| Seasonal Naive" in report
        assert "**LightGBM (Optuna)**" in report

    def test_both_horizons(self):
        results = [
            {
                "horizon": 7,
                "naive": {
                    "wape": 26.60, "mae": 125.93, "rmse": 462.00
                },
                "snaive": {
                    "wape": 25.39, "mae": 120.20, "rmse": 445.03
                },
            },
            {
                "horizon": 30,
                "naive": {
                    "wape": 26.60, "mae": 125.93, "rmse": 462.00
                },
                "snaive": {
                    "wape": 29.42, "mae": 139.26, "rmse": 517.06
                },
            },
        ]

        report = generate_report(results)

        assert "## Modelo Diario (h=7)" in report
        assert "## Modelo Mensual (h=30)" in report
        assert "### h7" in report
        assert "### h30" in report

    def test_improvement_calculation(self):
        results = [
            {
                "horizon": 7,
                "naive": {"wape": 26.60, "mae": 0, "rmse": 0},
                "snaive": {"wape": 25.39, "mae": 0, "rmse": 0},
            }
        ]

        report = generate_report(results)

        # 60.5% improvement vs naive
        assert "+60.5%" in report
        # 58.6% improvement vs snaive
        assert "+58.6%" in report
