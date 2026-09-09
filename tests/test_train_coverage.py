# tests/test_train_coverage.py
"""Tests para elevar cobertura de train.py de 59% a ~70%.

Cubre: get_feature_cols, _save_model, setup_mlflow.
"""
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


class TestGetFeatureCols:
    def test_excludes_date_and_target(self):
        from src.models.train import get_feature_cols
        df = pd.DataFrame({
            "date": [1], "sales": [10], "id": [1],
            "holiday_description": ["x"], "transactions": [5],
            "store_nbr": [1], "family": [10],
        })
        result = get_feature_cols(df)
        assert "date" not in result
        assert "sales" not in result
        assert "id" not in result
        assert "holiday_description" not in result
        assert "transactions" not in result
        assert "store_nbr" in result
        assert "family" in result

    def test_keeps_all_other_columns(self):
        from src.models.train import get_feature_cols
        df = pd.DataFrame({
            "date": [1], "sales": [10], "f1": [1.0], "f2": [2.0],
        })
        result = get_feature_cols(df)
        assert result == ["f1", "f2"]


class TestSaveModel:
    def test_saves_all_artifacts(self, tmp_path):
        from src.models.train import _save_model
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        feature_cols = ["f1", "f2", "f3"]
        with patch("src.models.train.Path", wraps=Path) as mock_path_cls, \
             patch("src.models.train.joblib") as mock_joblib:
            mock_path_cls.return_value = tmp_path / "models"
            (tmp_path / "models").mkdir()
            _save_model(mock_model, 7, feature_cols, mock_pipeline, {}, "")
            assert mock_joblib.dump.call_count == 2

    def test_suffix_affects_paths(self, tmp_path):
        from src.models.train import _save_model
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        with patch("src.models.train.Path", wraps=Path) as mock_path_cls, \
             patch("src.models.train.joblib") as mock_joblib:
            mock_path_cls.return_value = tmp_path / "models"
            (tmp_path / "models").mkdir()
            model_path, pipeline_path = _save_model(
                mock_model, 7, ["f1"], mock_pipeline, {}, "_new"
            )
            assert "h7_new.pkl" in str(model_path)
            assert "h7_new.pkl" in str(pipeline_path)


class TestSetupMlflow:
    @patch("src.models.train.mlflow")
    @patch("src.models.train.config", {
        "mlflow": {
            "tracking_uri": "https://dagshub.com/test",
            "experiment_name": "test-exp",
        }
    })
    @patch("dotenv.load_dotenv")
    @patch.dict("os.environ", {}, clear=True)
    def test_configures_mlflow(self, mock_load, mock_mlflow):
        from src.models.train import setup_mlflow
        setup_mlflow()
        mock_mlflow.set_tracking_uri.assert_called_once_with(
            "https://dagshub.com/test"
        )
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")
