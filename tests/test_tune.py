"""
Tests del módulo de tuning Optuna.

Valida la lógica de suggest_params, _objective, run_optuna_search,
_train_with_best_params y _save_tuning_results sin ejecutar
entrenamientos reales (todo mockeado).
"""
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.models.tune import (
    suggest_params,
    build_params_from_dict,
    _objective,
    run_optuna_search,
    _save_tuning_results,
)


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────
@pytest.fixture
def fake_fold_result():
    """FoldResult simulado con MAE conocido."""
    fr = MagicMock()
    fr.mae = 150.0
    fr.rmse = 200.0
    fr.mape = 10.0
    fr.rmsle = 0.4
    fr.wape = 12.0
    fr.fold = 1
    fr.train_start = pd.Timestamp("2015-01-01")
    fr.train_end = pd.Timestamp("2016-01-01")
    fr.val_start = pd.Timestamp("2016-01-02")
    fr.val_end = pd.Timestamp("2016-01-29")
    fr.n_train = 10000
    fr.n_val = 2000
    return fr


@pytest.fixture
def fake_df():
    """DataFrame simulado para tests."""
    dates = pd.date_range("2015-01-01", "2017-06-30", freq="D")
    rows = []
    for d in dates[:100]:
        for s in range(1, 4):
            for f in range(0, 3):
                rows.append({
                    "date": d,
                    "store_nbr": s,
                    "family": f,
                    "sales": np.random.lognormal(5, 1),
                    "dia_semana": d.weekday(),
                    "lag_7": np.random.lognormal(5, 1),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_optuna_trial():
    """Trial de Optuna simulado."""
    trial = MagicMock()
    trial.suggest_int.side_effect = lambda name, low, high, **kw: (low + high) // 2
    trial.suggest_float.side_effect = lambda name, low, high, **kw: (low + high) / 2
    trial.report = MagicMock()
    trial.should_prune.return_value = False
    return trial


# ─────────────────────────────────────────
# Tests de suggest_params
# ─────────────────────────────────────────
class TestSuggestParams:
    """suggest_params genera un dict válido de hiperparámetros."""

    def test_returns_dict_with_required_keys(self, mock_optuna_trial):
        params = suggest_params(mock_optuna_trial, horizon=7)
        required = [
            "objective", "metric", "random_state",
            "num_leaves", "learning_rate", "min_data_in_leaf",
            "lambda_l1", "lambda_l2", "feature_fraction",
            "bagging_fraction", "bagging_freq", "max_bin",
        ]
        for key in required:
            assert key in params, f"Falta key: {key}"

    def test_horizon_7_uses_regression_l1(self, mock_optuna_trial):
        params = suggest_params(mock_optuna_trial, horizon=7)
        assert params["objective"] == "regression_l1"

    def test_horizon_30_uses_huber(self, mock_optuna_trial):
        params = suggest_params(mock_optuna_trial, horizon=30)
        assert params["objective"] == "huber"

    def test_n_jobs_is_1_for_reproducibility(self, mock_optuna_trial):
        params = suggest_params(mock_optuna_trial, horizon=7)
        assert params["n_jobs"] == 1

    def test_num_leaves_in_valid_range(self, mock_optuna_trial):
        params = suggest_params(mock_optuna_trial, horizon=7)
        assert 20 <= params["num_leaves"] <= 127

    def test_learning_rate_in_valid_range(self, mock_optuna_trial):
        params = suggest_params(mock_optuna_trial, horizon=7)
        assert 0.01 <= params["learning_rate"] <= 0.15


class TestBuildParamsFromDict:
    """build_params_from_dict construye params completos desde JSON."""

    def test_adds_fixed_params(self):
        tuneable = {"num_leaves": 50, "learning_rate": 0.05}
        result = build_params_from_dict(tuneable, horizon=7)
        assert result["objective"] == "regression_l1"
        assert result["metric"] == "mae"
        assert result["random_state"] == 42
        assert result["num_leaves"] == 50

    def test_horizon_30_uses_huber(self):
        result = build_params_from_dict({}, horizon=30)
        assert result["objective"] == "huber"

    def test_n_jobs_minus_1_for_production(self):
        result = build_params_from_dict({}, horizon=7)
        assert result["n_jobs"] == -1


# ─────────────────────────────────────────
# Tests de _objective
# ─────────────────────────────────────────
class TestObjective:
    """_objective ejecuta CV y retorna MAE promedio."""

    @patch("src.models.tune.walk_forward_splits")
    @patch("src.models.tune._train_fold")
    def test_returns_mean_mae(
        self, mock_train_fold, mock_splits, fake_df, fake_fold_result
    ):
        mock_splits.return_value = [
            (pd.Index([0, 1]), pd.Index([2, 3]), {"fold": 1}),
            (pd.Index([0, 1, 2]), pd.Index([3, 4]), {"fold": 2}),
        ]
        mock_train_fold.return_value = (MagicMock(), fake_fold_result)

        trial = MagicMock()
        trial.report = MagicMock()
        trial.should_prune.return_value = False

        result = _objective(
            trial, fake_df, ["lag_7"], horizon=7,
            n_folds=2, max_boost_round=400,
            early_stopping_rounds=50,
        )
        assert result == 150.0
        assert mock_train_fold.call_count == 2

    @patch("src.models.tune.walk_forward_splits")
    @patch("src.models.tune._train_fold")
    def test_calls_train_fold_with_custom_params(
        self, mock_train_fold, mock_splits, fake_df, fake_fold_result
    ):
        mock_splits.return_value = [
            (pd.Index([0]), pd.Index([1]), {"fold": 1}),
        ]
        mock_train_fold.return_value = (MagicMock(), fake_fold_result)

        trial = MagicMock()
        trial.report = MagicMock()
        trial.should_prune.return_value = False

        _objective(
            trial, fake_df, ["lag_7"], horizon=7,
            n_folds=1, max_boost_round=200,
            early_stopping_rounds=30,
        )
        call_kwargs = mock_train_fold.call_args
        assert call_kwargs[1]["num_boost_round"] == 200
        assert call_kwargs[1]["early_stopping_rounds"] == 30


# ─────────────────────────────────────────
# Tests de _save_tuning_results
# ─────────────────────────────────────────
class TestSaveTuningResults:
    """_save_tuning_results persiste params, métricas y modelo."""

    def test_creates_json_files(self, tmp_path):
        study = MagicMock()
        study.best_trial.number = 5
        study.best_trial.value = 123.45
        study.best_trial.params = {"num_leaves": 50}
        study.best_params = {"num_leaves": 50}
        study.direction = "minimize"
        study.trials = [None] * 10

        model = MagicMock()
        metrics = {"mae": 123.45, "rmse": 200.0}

        with patch("src.models.tune.joblib.dump"):
            _save_tuning_results(
                horizon=7,
                study=study,
                model=model,
                metrics=metrics,
                output_dir=str(tmp_path),
            )

        params_file = tmp_path / "best_params_h7.json"
        metrics_file = tmp_path / "best_metrics_h7.json"
        stats_file = tmp_path / "study_stats_h7.json"
        assert params_file.exists()
        assert metrics_file.exists()
        assert stats_file.exists()

        with open(params_file) as f:
            data = json.load(f)
        assert data["num_leaves"] == 50

    def test_creates_model_file(self, tmp_path):
        study = MagicMock()
        study.best_trial.number = 0
        study.best_trial.value = 100.0
        study.best_trial.params = {}
        study.direction = "minimize"
        study.trials = []

        model = MagicMock()
        metrics = {"mae": 100.0}

        with patch("src.models.tune.joblib.dump") as mock_dump:
            _save_tuning_results(
                horizon=30,
                study=study,
                model=model,
                metrics=metrics,
                output_dir=str(tmp_path),
            )
            mock_dump.assert_called_once()


# ─────────────────────────────────────────
# Tests de run_optuna_search (integration mock)
# ─────────────────────────────────────────
class TestRunOptunaSearch:
    """run_optuna_search orquesta el flujo completo con mocks."""

    @patch("src.models.tune._save_tuning_results")
    @patch("src.models.tune._train_with_best_params")
    @patch("src.models.tune._objective")
    @patch("src.models.tune.walk_forward_splits")
    @patch("src.models.tune.DemandFeatureEngineer")
    @patch("src.models.tune.pd.read_parquet")
    @patch("src.models.tune.optuna.create_study")
    def test_runs_and_saves_results(
        self, mock_create_study, mock_read_parquet,
        mock_pipeline_cls, mock_splits, mock_obj,
        mock_train, mock_save, fake_df,
    ):
        mock_read_parquet.return_value = fake_df
        pipeline = MagicMock()
        pipeline.transform.return_value = fake_df
        mock_pipeline_cls.return_value = pipeline
        mock_train.return_value = (MagicMock(), {"mae": 100.0})

        mock_study = MagicMock()
        mock_study.best_trial.number = 0
        mock_study.best_trial.value = 123.0
        mock_study.best_trial.params = {"num_leaves": 50}
        mock_study.trials = [MagicMock()]
        mock_create_study.return_value = mock_study

        study, model, metrics = run_optuna_search(
            horizon=7, n_trials=2, output_dir="/tmp/test_optuna",
        )

        assert study is mock_study
        mock_save.assert_called_once()
        mock_train.assert_called_once()

    @patch("src.models.tune._save_tuning_results")
    @patch("src.models.tune._train_with_best_params")
    @patch("src.models.tune._objective")
    @patch("src.models.tune.walk_forward_splits")
    @patch("src.models.tune.DemandFeatureEngineer")
    @patch("src.models.tune.pd.read_parquet")
    @patch("src.models.tune.optuna.create_study")
    def test_passes_config_params(
        self, mock_create_study, mock_read_parquet,
        mock_pipeline_cls, mock_splits, mock_obj,
        mock_train, mock_save, fake_df,
    ):
        mock_read_parquet.return_value = fake_df
        pipeline = MagicMock()
        pipeline.transform.return_value = fake_df
        mock_pipeline_cls.return_value = pipeline
        mock_train.return_value = (MagicMock(), {"mae": 100.0})
        mock_obj.return_value = 100.0

        mock_study = MagicMock()
        mock_study.best_trial.number = 0
        mock_study.best_trial.value = 100.0
        mock_study.best_trial.params = {}
        mock_study.best_params = {}
        mock_study.trials = []
        mock_create_study.return_value = mock_study

        # Make optimize call the objective lambda once
        def capture_optimize(obj_fn, n_trials, timeout, **kw):
            trial = MagicMock()
            trial.suggest_int.side_effect = lambda n, lo, hi, **kw: (lo + hi) // 2
            trial.suggest_float.side_effect = lambda n, lo, hi, **kw: (lo + hi) / 2
            trial.report = MagicMock()
            trial.should_prune.return_value = False
            obj_fn(trial)
        mock_study.optimize.side_effect = capture_optimize

        run_optuna_search(
            horizon=30, n_trials=1, timeout=60,
            output_dir="/tmp/test_optuna2",
        )

        mock_obj.assert_called_once()
        call_args = mock_obj.call_args[0]
        assert isinstance(call_args[2], list)  # feature_cols
        assert call_args[3] == 30  # horizon
        assert call_args[4] == 3   # n_folds
        assert call_args[5] == 400  # max_boost_round
