# tests/test_evaluate_coverage.py
"""Tests para elevar cobertura de evaluate.py de 16% a ~70%.

Cubre: prepare_test_set, evaluate_global, evaluate_by_family,
evaluate_by_store, evaluate_by_time, evaluate_feature_importance,
plot_predictions, plot_feature_importance, plot_errors_by_family.
"""
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest


# ── Helpers ────────────────────────────────
def _fake_df(n=200):
    dates = pd.date_range("2016-01-01", periods=n)
    return pd.DataFrame({
        "date": dates,
        "store_nbr": np.tile([1, 2], n // 2),
        "family": np.tile([10, 20], n // 2),
        "city": np.tile(["Quito", "Guayaquil"], n // 2),
        "sales": np.log1p(np.abs(np.random.default_rng(42).normal(100, 30, n))),
    })


def _fake_test_df(n=50):
    dates = pd.date_range("2016-06-01", periods=n)
    return pd.DataFrame({
        "date": dates,
        "store_nbr": np.tile([1, 2], n // 2),
        "family": np.tile([10, 20], n // 2),
        "city": np.tile(["Quito", "Guayaquil"], n // 2),
        "sales": np.log1p(np.abs(np.random.default_rng(99).normal(100, 30, n))),
    })


# ── prepare_test_set ───────────────────────
class TestPrepareTestSet:
    @patch("src.models.evaluate.config", {
        "training": {"test_size_weeks": 8},
        "data": {"target": "sales"},
    })
    def test_returns_xy_and_df(self):
        from src.models.evaluate import prepare_test_set
        df = _fake_df(200)
        feature_cols = ["store_nbr", "family"]
        X, y, test_df = prepare_test_set(df, feature_cols)
        assert len(X) > 0
        assert len(y) > 0
        assert "date" in test_df.columns
        assert list(X.columns) == feature_cols

    @patch("src.models.evaluate.config", {
        "training": {"test_size_weeks": 8},
        "data": {"target": "sales"},
    })
    def test_test_set_is_last_weeks(self):
        from src.models.evaluate import prepare_test_set
        df = _fake_df(200)
        X, y, test_df = prepare_test_set(df, ["store_nbr"])
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(weeks=8)
        assert (test_df["date"] > cutoff).all()


# ── evaluate_global ────────────────────────
class TestEvaluateGlobal:
    @patch("src.models.evaluate.mlflow")
    def test_returns_metrics_dict(self, mock_mlflow):
        from src.models.evaluate import evaluate_global
        y_true = np.log1p(np.array([100.0, 200.0, 300.0]))
        y_pred = np.log1p(np.array([110.0, 190.0, 310.0]))
        result = evaluate_global(y_true, y_pred, horizon=7)
        assert "rmse" in result
        assert "mae" in result
        assert "wape" in result
        mock_mlflow.log_metric.assert_called()

    @patch("src.models.evaluate.mlflow")
    def test_logs_with_test_prefix(self, mock_mlflow):
        from src.models.evaluate import evaluate_global
        y_true = np.log1p(np.array([100.0]))
        y_pred = np.log1p(np.array([100.0]))
        evaluate_global(y_true, y_pred, horizon=30)
        calls = [c[0][0] for c in mock_mlflow.log_metric.call_args_list]
        assert all(k.startswith("test_") for k in calls)


# ── evaluate_by_family ─────────────────────
class TestEvaluateByFamily:
    @patch("src.models.evaluate.config", {
        "data": {"target": "sales"},
        "model": {"top_families": [10, 20]},
    })
    def test_returns_dataframe_with_family(self):
        from src.models.evaluate import evaluate_by_family
        test_df = _fake_test_df(50)
        y_pred = test_df["sales"].values * 0.95
        result = evaluate_by_family(test_df, y_pred)
        assert "family" in result.columns
        assert "rmse" in result.columns
        assert "n_rows" in result.columns
        assert len(result) == test_df["family"].nunique()

    @patch("src.models.evaluate.config", {
        "data": {"target": "sales"},
        "model": {"top_families": [10]},
    })
    def test_sorted_by_rmse_desc(self):
        from src.models.evaluate import evaluate_by_family
        test_df = _fake_test_df(50)
        y_pred = test_df["sales"].values
        result = evaluate_by_family(test_df, y_pred)
        assert result["rmse"].is_monotonic_decreasing


# ── evaluate_by_store ──────────────────────
class TestEvaluateByStore:
    @patch("src.models.evaluate.config", {
        "data": {"target": "sales"},
    })
    def test_returns_store_metrics(self):
        from src.models.evaluate import evaluate_by_store
        test_df = _fake_test_df(50)
        y_pred = test_df["sales"].values * 0.9
        result = evaluate_by_store(test_df, y_pred)
        assert "store_nbr" in result.columns
        assert "city" in result.columns
        assert "rmse" in result.columns
        assert "n_rows" in result.columns


# ── evaluate_by_time ───────────────────────
class TestEvaluateByTime:
    @patch("src.models.evaluate.config", {
        "data": {"target": "sales"},
    })
    def test_returns_time_metrics(self):
        from src.models.evaluate import evaluate_by_time
        test_df = _fake_test_df(50)
        y_pred = test_df["sales"].values * 1.1
        result = evaluate_by_time(test_df, y_pred)
        assert "date" in result.columns
        assert "mae_diario" in result.columns
        assert "std_diario" in result.columns


# ── evaluate_feature_importance ────────────
class TestEvaluateFeatureImportance:
    def test_returns_importance_df(self):
        from src.models.evaluate import evaluate_feature_importance
        mock_model = MagicMock()
        mock_model.feature_importance.return_value = np.array([100, 50, 10])
        features = ["f1", "f2", "f3"]
        result = evaluate_feature_importance(mock_model, features)
        assert "feature" in result.columns
        assert "importance" in result.columns
        assert "importance_pct" in result.columns
        assert len(result) == 3
        mock_model.feature_importance.assert_called_once_with(
            importance_type="gain"
        )

    def test_importance_pct_sums_to_100(self):
        from src.models.evaluate import evaluate_feature_importance
        mock_model = MagicMock()
        mock_model.feature_importance.return_value = np.array([60, 30, 10])
        result = evaluate_feature_importance(mock_model, ["a", "b", "c"])
        assert abs(result["importance_pct"].sum() - 100.0) < 0.1


# ── plot_predictions ───────────────────────
class TestPlotPredictions:
    @patch("src.models.evaluate.plt")
    @patch("src.models.evaluate.config", {
        "data": {"target": "sales"},
    })
    def test_calls_savefig_when_path(self, mock_plt):
        from src.models.evaluate import plot_predictions
        test_df = _fake_test_df(10)
        mask = (test_df["store_nbr"] == 1) & (test_df["family"] == 10)
        y_pred = np.random.default_rng(42).normal(0, 1, mask.sum())
        fig_mock = MagicMock()
        axes_mock = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (fig_mock, axes_mock)
        plot_predictions(test_df[mask], y_pred, save_path="/tmp/test.png")
        mock_plt.savefig.assert_called_once()

    @patch("src.models.evaluate.plt")
    @patch("src.models.evaluate.config", {
        "data": {"target": "sales"},
    })
    def test_no_savefig_without_path(self, mock_plt):
        from src.models.evaluate import plot_predictions
        test_df = _fake_test_df(10)
        mask = (test_df["store_nbr"] == 1) & (test_df["family"] == 10)
        y_pred = np.random.default_rng(42).normal(0, 1, mask.sum())
        fig_mock = MagicMock()
        axes_mock = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (fig_mock, axes_mock)
        plot_predictions(test_df[mask], y_pred)
        mock_plt.savefig.assert_not_called()


# ── plot_feature_importance ────────────────
class TestPlotFeatureImportance:
    @patch("src.models.evaluate.plt")
    def test_calls_savefig_when_path(self, mock_plt):
        from src.models.evaluate import plot_feature_importance
        imp_df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "importance": [100, 50, 10],
            "importance_pct": [60.0, 30.0, 10.0],
        })
        fig_mock = MagicMock()
        ax_mock = MagicMock()
        mock_plt.subplots.return_value = (fig_mock, ax_mock)
        plot_feature_importance(imp_df, save_path="/tmp/test.png")
        mock_plt.savefig.assert_called_once()


# ── plot_errors_by_family ──────────────────
class TestPlotErrorsByFamily:
    @patch("src.models.evaluate.plt")
    def test_calls_savefig_when_path(self, mock_plt):
        from src.models.evaluate import plot_errors_by_family
        family_df = pd.DataFrame({
            "family": ["A", "B", "C"],
            "rmse": [300.0, 200.0, 100.0],
        })
        fig_mock = MagicMock()
        ax_mock = MagicMock()
        mock_plt.subplots.return_value = (fig_mock, ax_mock)
        plot_errors_by_family(family_df, save_path="/tmp/test.png")
        mock_plt.savefig.assert_called_once()
