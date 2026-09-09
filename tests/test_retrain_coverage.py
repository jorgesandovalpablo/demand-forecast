# tests/test_retrain_coverage.py
"""Tests para elevar cobertura de retrain.py de 55% a ~70%.

Cubre: _load_current_metrics, _evaluate_new_model, _rotate_models,
_discard_staging, _artifact_paths, ARTIFACT_KINDS.
"""
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import shutil


class TestLoadCurrentMetrics:
    def test_returns_inf_when_no_file(self):
        from src.models.retrain import _load_current_metrics
        with patch("src.models.retrain.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = _load_current_metrics(7)
            assert result["rmse"] == float("inf")
            assert result["wape"] == float("inf")

    def test_loads_from_parquet(self):
        from src.models.retrain import _load_current_metrics
        metrics_df = pd.DataFrame({
            "rmse": [200.0, 210.0],
            "mae": [50.0, 55.0],
            "mape": [10.0, 12.0],
            "rmsle": [0.4, 0.5],
            "wape": [12.0, 14.0],
        })
        with patch("src.models.retrain.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = True
            with patch("src.models.retrain.pd.read_parquet", return_value=metrics_df):
                result = _load_current_metrics(7)
                assert isinstance(result, dict)
                assert result["mae"] == round((50.0 + 55.0) / 2, 4)


class TestEvaluateNewModel:
    def test_loads_model_and_evaluates(self):
        from src.models.retrain import _evaluate_new_model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0])
        fake_df = pd.DataFrame({
            "date": pd.date_range("2016-01-01", periods=100),
            "store_nbr": [1] * 100,
            "family": [0] * 100,
            "sales": np.ones(100),
        })
        with patch("src.models.retrain.joblib.load", return_value=mock_model), \
             patch("src.models.retrain.get_feature_cols", return_value=["store_nbr"]), \
             patch("src.models.retrain.config", {
                 "data": {"target": "sales"},
                 "training": {"test_size_weeks": 8},
             }), \
             patch("src.models.retrain.compute_metrics", return_value={"rmse": 1.0, "mae": 0.5, "mape": 5.0, "rmsle": 0.1, "wape": 6.0}):
            result = _evaluate_new_model(7, fake_df)
            assert "rmse" in result


class TestArtifactPaths:
    def test_returns_three_paths(self):
        from src.models.retrain import _artifact_paths
        paths = _artifact_paths(7)
        assert len(paths) == 3
        for p in paths:
            assert isinstance(p, Path)
            assert "h7" in str(p)
            assert ".pkl" in str(p)

    def test_suffix_appended(self):
        from src.models.retrain import _artifact_paths
        paths = _artifact_paths(30, suffix="_new")
        for p in paths:
            assert "h30_new" in str(p)


class TestRotateModels:
    def test_raises_when_new_missing(self):
        from src.models.retrain import _rotate_models
        with patch("src.models.retrain._artifact_paths") as mock_paths:
            mock_paths.return_value = [Path("/nonexistent/file.pkl")]
            with pytest.raises(FileNotFoundError):
                _rotate_models(7)


class TestDiscardStaging:
    def test_removes_new_files(self, tmp_path):
        from src.models.retrain import _discard_staging
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        for kind in ["lgbm_h7", "features_h7", "feature_pipeline_h7"]:
            f = models_dir / f"{kind}_new.pkl"
            f.write_text("staging")
        with patch("src.models.retrain._artifact_paths") as mock_paths:
            mock_paths.return_value = [
                models_dir / "lgbm_h7_new.pkl",
                models_dir / "features_h7_new.pkl",
                models_dir / "feature_pipeline_h7_new.pkl",
            ]
            _discard_staging(7)
            for kind in ["lgbm_h7", "features_h7", "feature_pipeline_h7"]:
                assert not (models_dir / f"{kind}_new.pkl").exists()


class TestArtifactKinds:
    def test_three_kinds(self):
        from src.models.retrain import ARTIFACT_KINDS
        assert len(ARTIFACT_KINDS) == 3
        assert "lgbm_h" in ARTIFACT_KINDS
        assert "features_h" in ARTIFACT_KINDS
        assert "feature_pipeline_h" in ARTIFACT_KINDS
