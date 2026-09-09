import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.models.predict import (
    ModelRegistry,
    _build_confidence_intervals,
    _load_raw_predict_cached,
    save_predictions,
    predict_by_store,
    _RAW_PREDICT_CACHE,
)


class TestBuildConfidenceIntervals:
    def test_basic(self) -> None:
        y_log = np.array([2.0])
        std = np.array([0.5])
        lower, upper = _build_confidence_intervals(y_log, std, z=1.96)
        assert lower[0] >= 0
        assert upper[0] >= lower[0]

    def test_z_zero(self) -> None:
        y_log = np.array([3.0])
        std = np.array([1.0])
        lower, upper = _build_confidence_intervals(y_log, std, z=0)
        np.testing.assert_allclose(lower, upper, atol=0.01)

    def test_negative_clipped(self) -> None:
        y_log = np.array([0.0])
        std = np.array([10.0])
        lower, upper = _build_confidence_intervals(y_log, std, z=1.96)
        assert lower[0] >= 0

    def test_upper_never_below_lower(self) -> None:
        y_log = np.array([1.0])
        std = np.array([0.1])
        lower, upper = _build_confidence_intervals(
            y_log, std, z=1.96, upper_factor=0.01
        )
        assert upper[0] >= lower[0]


class TestModelRegistry:
    def setup_method(self) -> None:
        ModelRegistry.clear_cache()

    def test_clear_cache(self) -> None:
        ModelRegistry._models[99] = "dummy"
        ModelRegistry.clear_cache()
        assert ModelRegistry._models == {}
        assert ModelRegistry._features == {}
        assert ModelRegistry._pipelines == {}
        assert ModelRegistry._std == {}

    def test_load_artifacts_missing_raises(self) -> None:
        fake = MagicMock(exists=lambda: False)
        with patch("src.models.predict.Path", return_value=fake):
            with patch(
                "src.models.registry.ensure_local_artifacts",
                return_value=False,
            ):
                with pytest.raises(FileNotFoundError, match="no disponible"):
                    ModelRegistry.load(7)

    def test_load_artifacts_missing_ensure_returns_false(self) -> None:
        fake = MagicMock(exists=lambda: False)
        with patch("src.models.predict.Path", return_value=fake):
            with patch(
                "src.models.registry.ensure_local_artifacts",
                return_value=False,
            ):
                with pytest.raises(FileNotFoundError, match="no disponible"):
                    ModelRegistry.load(7)

    def test_load_with_existing_artifacts(self, tmp_path: Path) -> None:
        model_path = tmp_path / "lgbm_h7.pkl"
        features_path = tmp_path / "features_h7.pkl"
        pipeline_path = tmp_path / "feature_pipeline_h7.pkl"
        model_path.touch()
        features_path.touch()
        pipeline_path.touch()

        with patch("src.models.predict.Path") as mock_path_cls:
            def _side_effect(p: str) -> Path:
                mapping = {
                    "models/lgbm_h7.pkl": model_path,
                    "models/features_h7.pkl": features_path,
                    "models/feature_pipeline_h7.pkl": pipeline_path,
                }
                return mapping.get(p, tmp_path / "dummy.pkl")

            mock_path_cls.side_effect = _side_effect

            with patch("joblib.load") as mock_load:
                mock_load.return_value = MagicMock()
                with patch.object(
                    ModelRegistry, "_load_residual_std"
                ):
                    ModelRegistry.load(7)

                assert 7 in ModelRegistry._models
                assert 7 in ModelRegistry._features

    def test_load_residual_std_fallback(self) -> None:
        pipeline = MagicMock()
        pipeline.store_stats = pd.DataFrame({
            "store_nbr": [1, 2],
            "family": ["A", "B"],
            "venta_std_historica": [0.1, 0.2],
        })
        pipeline.categories_mapping = {"family": ["A", "B"]}

        with patch("src.models.predict.Path") as mock_path_cls:
            mock_path_cls.return_value = MagicMock(
                exists=lambda: False
            )
            with patch("joblib.load", return_value=pipeline):
                ModelRegistry._load_residual_std(7, Path("dummy.pkl"))

        assert 7 in ModelRegistry._std
        result = ModelRegistry._std[7]
        assert "global" in result
        assert "df" in result
        assert "resid_std" in result["df"].columns
        assert abs(result["global"] - 0.15) < 1e-6

    def test_load_residual_std_with_file(self) -> None:
        std_data = {"global": 0.3, "df": pd.DataFrame({"x": [1]})}
        with patch("src.models.predict.Path") as mock_path_cls:
            mock_path_cls.return_value = MagicMock(
                exists=lambda: True
            )
            with patch("joblib.load", return_value=std_data):
                ModelRegistry._load_residual_std(7, Path("dummy.pkl"))

        assert ModelRegistry._std[7] is std_data

    def test_get_residual_std_triggers_load(self) -> None:
        dummy = {"global": 0.1, "df": pd.DataFrame()}
        ModelRegistry._std = {}

        def _pop(h):
            ModelRegistry._std[h] = dummy

        with patch.object(ModelRegistry, "load", side_effect=_pop):
            result = ModelRegistry.get_residual_std(7)
        assert result is dummy

    def test_get_features_triggers_load(self) -> None:
        ModelRegistry._features = {}

        def _pop(h):
            ModelRegistry._features[h] = ["f1", "f2"]

        with patch.object(ModelRegistry, "load", side_effect=_pop):
            result = ModelRegistry.get_features(7)
        assert result == ["f1", "f2"]

    def test_get_pipeline_triggers_load(self) -> None:
        ModelRegistry._pipelines = {}
        dummy = MagicMock()

        def _pop(h):
            ModelRegistry._pipelines[h] = dummy

        with patch.object(ModelRegistry, "load", side_effect=_pop):
            result = ModelRegistry.get_pipeline(7)
        assert result is dummy


class TestLoadRawPredictCached:
    def setup_method(self) -> None:
        _RAW_PREDICT_CACHE.clear()

    def teardown_method(self) -> None:
        _RAW_PREDICT_CACHE.clear()

    def test_cache_populates(self) -> None:
        fake_data = {"stores": pd.DataFrame({"a": [1]})}
        with patch(
            "src.models.predict.load_raw_data", return_value=fake_data
        ):
            result = _load_raw_predict_cached()
            assert "stores" in result

    def test_cache_hit_returns_copy(self) -> None:
        fake_data = {"stores": pd.DataFrame({"a": [1]})}
        _RAW_PREDICT_CACHE["_raw"] = fake_data
        with patch(
            "src.models.predict.load_raw_data"
        ) as mock_load:
            result = _load_raw_predict_cached()
            mock_load.assert_not_called()
            assert "stores" in result
            assert result is not fake_data


class TestSavePredictions:
    def test_creates_parquet(self, tmp_path: Path) -> None:
        preds = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01"]),
            "store_nbr": [1],
            "family": ["A"],
            "predicted_sales": [10.0],
            "lower_bound": [5.0],
            "upper_bound": [15.0],
        })
        with patch("src.models.predict.Path") as mock_path_cls:
            def _side_effect(p: str) -> Path:
                if p == "data/predictions":
                    return tmp_path
                return Path(p)
            mock_path_cls.side_effect = _side_effect
            filepath = save_predictions(preds, 7)
        assert filepath.exists()


class TestPredictByStore:
    def test_store_not_found(self) -> None:
        df = pd.DataFrame({
            "store_nbr": [1, 1],
            "family": ["A", "A"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "sales": [10.0, 20.0],
        })
        with pytest.raises(ValueError, match="no encontrada"):
            predict_by_store(df, 7, store_nbr=999)

    def test_store_found(self) -> None:
        df = pd.DataFrame({
            "store_nbr": [1],
            "family": ["A"],
            "date": pd.to_datetime(["2026-01-01"]),
            "sales": [10.0],
        })
        with patch("src.models.predict.predict") as mock_predict:
            mock_predict.return_value = pd.DataFrame({
                "date": pd.to_datetime(["2026-01-02"]),
                "store_nbr": [1],
                "family": ["A"],
                "predicted_sales": [11.0],
                "lower_bound": [5.0],
                "upper_bound": [17.0],
            })
            result = predict_by_store(df, 7, store_nbr=1)
            mock_predict.assert_called_once()
            assert len(result) == 1
