# tests/test_validation_coverage.py
"""Tests para elevar cobertura de validation.py de 59% a ~75%.

Cubre: summarize_validation, plot_folds, walk_forward_splits,
FoldResult/ValidationResult dataclasses.
"""
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest


class TestSummarizeValidation:
    def test_returns_validation_result(self):
        from src.models.validation import summarize_validation, FoldResult
        results = [
            FoldResult(
                fold=i, train_start=pd.Timestamp("2016-01-01"),
                train_end=pd.Timestamp("2016-06-01"),
                val_start=pd.Timestamp("2016-06-02"),
                val_end=pd.Timestamp("2016-06-29"),
                rmse=200.0 + i, mae=50.0 + i, mape=10.0 + i,
                rmsle=0.4 + i * 0.1, wape=12.0 + i, n_train=100, n_val=50,
            )
            for i in range(1, 4)
        ]
        from src.models.validation import summarize_validation
        summary = summarize_validation(results)
        assert summary.rmse_mean > 0
        assert summary.rmse_std >= 0
        assert summary.mae_mean > 0
        assert len(summary.folds) == 3

    def test_std_zero_for_single_fold(self):
        from src.models.validation import summarize_validation, FoldResult
        results = [
            FoldResult(
                fold=1, train_start=pd.Timestamp("2016-01-01"),
                train_end=pd.Timestamp("2016-06-01"),
                val_start=pd.Timestamp("2016-06-02"),
                val_end=pd.Timestamp("2016-06-29"),
                rmse=100.0, mae=50.0, mape=10.0,
                rmsle=0.5, wape=15.0, n_train=100, n_val=50,
            )
        ]
        from src.models.validation import summarize_validation
        summary = summarize_validation(results)
        assert summary.rmse_std == 0.0
        assert summary.mae_std == 0.0


class TestPlotFolds:
    def test_closes_plt(self):
        from src.models.validation import plot_folds, FoldResult
        results = [
            FoldResult(
                fold=i, train_start=pd.Timestamp("2016-01-01"),
                train_end=pd.Timestamp("2016-06-01"),
                val_start=pd.Timestamp("2016-06-02"),
                val_end=pd.Timestamp("2016-06-29"),
                rmse=200.0, mae=50.0, mape=10.0,
                rmsle=0.4, wape=12.0, n_train=100, n_val=50,
            )
            for i in range(1, 4)
        ]
        with patch("matplotlib.pyplot.close") as mock_close:
            plot_folds(results)
            mock_close.assert_called_once()


class TestWalkForwardSplits:
    @patch("src.models.validation.config", {
        "training": {"n_folds": 2, "test_size_weeks": 8},
        "data": {"date_col": "date"},
    })
    def test_generates_folds(self):
        from src.models.validation import walk_forward_splits
        n_days = 200
        df = pd.DataFrame({
            "date": pd.date_range("2015-01-01", periods=n_days),
            "store_nbr": [1] * n_days,
        })
        folds = list(walk_forward_splits(df, n_folds=2))
        assert len(folds) >= 1
        for train_idx, val_idx, fold_info in folds:
            assert "fold" in fold_info
            assert fold_info["n_train"] > 0
            assert fold_info["n_val"] > 0

    @patch("src.models.validation.config", {
        "training": {"n_folds": 2, "test_size_weeks": 8},
        "data": {"date_col": "date"},
    })
    def test_skips_when_insufficient_data(self):
        from src.models.validation import walk_forward_splits
        df = pd.DataFrame({
            "date": pd.date_range("2015-01-01", periods=10),
            "store_nbr": [1] * 10,
        })
        folds = list(walk_forward_splits(df, n_folds=2, val_weeks=4))
        assert len(folds) == 0


class TestDataclasses:
    def test_fold_result_defaults(self):
        from src.models.validation import FoldResult
        fr = FoldResult(
            fold=1, train_start=pd.Timestamp("2016-01-01"),
            train_end=pd.Timestamp("2016-06-01"),
            val_start=pd.Timestamp("2016-06-02"),
            val_end=pd.Timestamp("2016-06-29"),
        )
        assert fr.rmse == 0.0
        assert fr.wape == 0.0

    def test_validation_result_defaults(self):
        from src.models.validation import ValidationResult
        vr = ValidationResult()
        assert vr.folds == []
        assert vr.rmse_mean == 0.0
