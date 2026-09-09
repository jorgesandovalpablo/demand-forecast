# tests/test_baselines_coverage.py
"""Tests para elevar cobertura de baselines.py de 70% a ~80%.

Cubre: run_baselines (con mocks).
"""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest


class TestRunBaselines:
    def test_raises_when_no_file(self):
        from src.models.baselines import run_baselines
        with patch("src.models.baselines.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            with pytest.raises(FileNotFoundError):
                run_baselines(7)

    def test_runs_both_baselines(self):
        from src.models.baselines import run_baselines
        fake_df = pd.DataFrame({
            "date": pd.date_range("2016-01-01", periods=200),
            "store_nbr": [1] * 200,
            "family": [10] * 200,
            "sales": np.log1p(np.abs(np.random.default_rng(42).normal(100, 30, 200))),
        })
        with patch("src.models.baselines.Path") as mock_path, \
             patch("src.models.baselines.pd.read_parquet", return_value=fake_df), \
             patch("src.models.baselines.prepare_test_set") as mock_prepare, \
             patch("src.models.baselines.naive_last_value") as mock_naive, \
             patch("src.models.baselines.seasonal_naive") as mock_snaive, \
             patch("src.models.baselines.compute_metrics") as mock_metrics:
            mock_path.return_value.exists.return_value = True
            target = "sales"
            test_df = fake_df.iloc[-50:]
            mock_prepare.return_value = (
                fake_df.iloc[:-50], test_df, target, ["store_nbr", "family"]
            )
            mock_naive.return_value = np.ones(50)
            mock_snaive.return_value = np.ones(50)
            mock_metrics.return_value = {
                "rmse": 1.0, "mae": 0.5, "mape": 5.0,
                "rmsle": 0.1, "wape": 6.0,
            }
            result = run_baselines(7)
            assert "naive" in result
            assert "snaive" in result
            assert mock_metrics.call_count == 2
