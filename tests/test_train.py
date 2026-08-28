"""
Tests del módulo de entrenamiento.

Valida que run_training funciona correctamente con config.yaml.
El override de params_file vive en test_retrain.py.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd


def _fake_df():
    return pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=100),
        "store_nbr": [1] * 100,
        "family": [0] * 100,
        "sales": [10.0] * 100,
    })


def _mock_fold_result():
    fr = MagicMock()
    fr.rmse = 200.0
    fr.mae = 50.0
    fr.mape = 10.0
    fr.rmsle = 0.4
    fr.wape = 12.0
    fr.fold = 1
    fr.train_start = pd.Timestamp("2015-01-01")
    fr.train_end = pd.Timestamp("2016-01-01")
    fr.val_start = pd.Timestamp("2016-01-02")
    fr.val_end = pd.Timestamp("2016-01-29")
    fr.n_train = 50
    fr.n_val = 50
    return fr


def _mock_fold_model():
    m = MagicMock()
    m.best_iteration = 100
    return m


# ─────────────────────────────────────────
# Tests de run_training (config.yaml)
# ─────────────────────────────────────────
class TestRunTraining:
    """run_training usa params de config.yaml por defecto."""

    @patch("src.models.train._save_model")
    @patch("src.models.train._train_final_model")
    @patch("src.models.train._train_fold")
    @patch("src.models.train.walk_forward_splits")
    @patch("src.features.build_features.DemandFeatureEngineer")
    @patch("src.models.train.pd.read_parquet")
    @patch("src.models.train.setup_mlflow")
    @patch("src.models.train.mlflow")
    def test_run_training_uses_config_defaults(
        self, mock_mlflow, mock_setup, mock_read_parquet,
        mock_pipeline_cls, mock_splits, mock_train_fold,
        mock_final_model, mock_save,
    ):
        """Sin params_file, usa config.yaml (backward-compatible)."""
        from src.models.train import run_training

        fake_df = _fake_df()
        mock_read_parquet.return_value = fake_df
        pipeline = MagicMock()
        pipeline.transform.return_value = fake_df
        mock_pipeline_cls.return_value = pipeline

        mock_train_fold.return_value = (_mock_fold_model(), _mock_fold_result())
        mock_splits.return_value = [
            (pd.RangeIndex(50), pd.RangeIndex(50, 100), {"fold": 1}),
        ]

        mock_final_model.return_value = MagicMock()
        mock_save.return_value = (Path("models/lgbm_h7.pkl"), Path("models/feature_pipeline_h7.pkl"))

        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()

        result = run_training(horizon=7)

        assert result is not None
        assert "model" in result
        mock_final_model.assert_called_once()

        # Verificar que los params de config se usaron (num_leaves=63 para h7)
        call_kwargs = mock_final_model.call_args
        params_used = call_kwargs[1]["params"]
        assert params_used["num_leaves"] == 63  # default config h7
