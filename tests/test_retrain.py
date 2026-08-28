"""
Tests del módulo de reentrenamiento.

Valida la propagación de params_file y la lógica de promoción.
"""
from unittest.mock import patch, MagicMock

import pandas as pd


def _fake_df():
    return pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=100),
        "store_nbr": [1] * 100,
        "family": [0] * 100,
        "sales": [10.0] * 100,
    })


def _mock_metrics():
    return {
        "rmse": 200.0,
        "mae": 50.0,
        "mape": 10.0,
        "rmsle": 0.4,
        "wape": 12.0,
    }


class TestParamsFilePropagation:
    """run_retraining propaga params_file a run_training."""

    @patch("src.models.retrain.promote_local_artifacts")
    @patch("src.models.retrain._rotate_models")
    @patch("src.models.retrain._should_update_model")
    @patch("src.models.retrain._evaluate_new_model")
    @patch("src.models.retrain._load_current_metrics")
    @patch("src.models.retrain.run_preprocessing")
    @patch("src.models.retrain.load_raw_data")
    @patch("src.models.retrain.run_training")
    @patch("src.models.train.setup_mlflow")
    @patch("src.models.retrain.mlflow")
    def test_params_file_propagated_to_run_training(
        self, mock_mlflow, mock_setup, mock_run_training,
        mock_load_data, mock_preprocess, mock_load_metrics,
        mock_evaluate, mock_should_update, mock_rotate,
        mock_promote, tmp_path,
    ):
        """params_file se pasa correctamente a run_training."""
        from src.models.retrain import run_retraining

        mock_load_data.return_value = MagicMock()
        mock_preprocess.return_value = (_fake_df(), None)
        mock_load_metrics.return_value = _mock_metrics()

        fake_df = _fake_df()
        mock_run_training.return_value = {
            "model": MagicMock(),
            "summary": MagicMock(),
            "features": [],
            "horizon": 7,
            "df": fake_df,
        }
        mock_evaluate.return_value = _mock_metrics()
        mock_should_update.return_value = True

        params_file = tmp_path / "best_params_h7.json"
        params_file.write_text('{"num_leaves": 47}')

        run_retraining(horizon=7, params_file=str(params_file))

        mock_run_training.assert_called_once_with(
            horizon=7,
            output_suffix="_new",
            params_file=str(params_file),
        )

    @patch("src.models.retrain.promote_local_artifacts")
    @patch("src.models.retrain._rotate_models")
    @patch("src.models.retrain._should_update_model")
    @patch("src.models.retrain._evaluate_new_model")
    @patch("src.models.retrain._load_current_metrics")
    @patch("src.models.retrain.run_preprocessing")
    @patch("src.models.retrain.load_raw_data")
    @patch("src.models.retrain.run_training")
    @patch("src.models.train.setup_mlflow")
    @patch("src.models.retrain.mlflow")
    def test_no_params_file_passes_none(
        self, mock_mlflow, mock_setup, mock_run_training,
        mock_load_data, mock_preprocess, mock_load_metrics,
        mock_evaluate, mock_should_update, mock_rotate,
        mock_promote,
    ):
        """Sin params_file, run_training recibe params_file=None."""
        from src.models.retrain import run_retraining

        mock_load_data.return_value = MagicMock()
        mock_preprocess.return_value = (_fake_df(), None)
        mock_load_metrics.return_value = _mock_metrics()

        fake_df = _fake_df()
        mock_run_training.return_value = {
            "model": MagicMock(),
            "summary": MagicMock(),
            "features": [],
            "horizon": 7,
            "df": fake_df,
        }
        mock_evaluate.return_value = _mock_metrics()
        mock_should_update.return_value = True

        run_retraining(horizon=7)

        mock_run_training.assert_called_once_with(
            horizon=7,
            output_suffix="_new",
            params_file=None,
        )


class TestShouldUpdateModel:
    """_should_update_model compara MAE con threshold."""

    def test_accepts_when_new_is_better(self):
        from src.models.retrain import _should_update_model
        current = {"mae": 100.0}
        new = {"mae": 90.0}
        assert _should_update_model(current, new, threshold=0.01) is True

    def test_rejects_when_new_is_worse(self):
        from src.models.retrain import _should_update_model
        current = {"mae": 100.0}
        new = {"mae": 110.0}
        assert _should_update_model(current, new, threshold=0.01) is False

    def test_accepts_when_no_previous_model(self):
        from src.models.retrain import _should_update_model
        current = {"mae": float("inf")}
        new = {"mae": 50.0}
        assert _should_update_model(current, new) is True

    def test_rejects_when_improvement_too_small(self):
        from src.models.retrain import _should_update_model
        current = {"mae": 100.0}
        new = {"mae": 99.5}  # 0.5% improvement < 1% threshold
        assert _should_update_model(current, new, threshold=0.01) is False
