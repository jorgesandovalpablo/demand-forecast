"""
Tests unitarios del módulo de análisis SHAP.

Estrategia: mock del modelo LightGBM y datos sintéticos para
validar la lógica de SHAP values sin depender de modelos
entrenados ni del dataset completo (CI rápida y aislada).
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.models.shap_analysis import (
    classify_features,
    compute_shap_values,
)


@pytest.fixture
def sample_shap_df() -> pd.DataFrame:
    """DataFrame de ejemplo con SHAP values simulados."""
    rng = np.random.default_rng(42)
    n_features = 20
    return pd.DataFrame({
        "feature": [f"feat_{i}" for i in range(n_features)],
        "shap_mean": rng.exponential(0.1, n_features),
    }).sort_values("shap_mean", ascending=False).reset_index(
        drop=True
    )


@pytest.fixture
def sample_shap_df_with_ratio(sample_shap_df) -> pd.DataFrame:
    """DataFrame con shap_ratio ya calculado."""
    df = sample_shap_df.copy()
    df["shap_ratio"] = df["shap_mean"] / df["shap_mean"].max()
    return df


class TestClassifyFeatures:
    """Clasificación KEEP / DROP_CANDIDATE."""

    def test_all_keep_when_uniform(self):
        """Si todos los valores son iguales, todos son KEEP."""
        df = pd.DataFrame({
            "feature": [f"f{i}" for i in range(10)],
            "shap_mean": [1.0] * 10,
            "shap_ratio": [1.0] * 10,
        })
        result = classify_features(df)
        assert (result["decision"] == "KEEP").all()

    def test_low_features_dropped(self):
        """Features con shap_mean bajo y ratio bajo → DROP_CANDIDATE."""
        df = pd.DataFrame({
            "feature": ["high", "mid", "low1", "low2"],
            "shap_mean": [1.0, 0.5, 0.01, 0.005],
            "shap_ratio": [1.0, 0.5, 0.01, 0.005],
        })
        result = classify_features(
            df, low_percentile=25, min_ratio=0.05
        )
        assert result.iloc[0]["decision"] == "KEEP"
        assert result.iloc[-1]["decision"] == "DROP_CANDIDATE"

    def test_returns_dataframe_with_decision_column(
        self, sample_shap_df_with_ratio
    ):
        """El resultado tiene columna 'decision'."""
        result = classify_features(sample_shap_df_with_ratio)
        assert "decision" in result.columns
        assert set(result["decision"].unique()).issubset(
            {"KEEP", "DROP_CANDIDATE"}
        )


class TestComputeShapValues:
    """Cálculo de SHAP values con modelo mock."""

    def _make_mock_model(self, n_features: int = 10):
        """Crea un modelo LightGBM mock con predict."""
        mock_model = MagicMock()
        rng = np.random.default_rng(42)

        def mock_shap_values(X):
            return rng.exponential(0.1, (len(X), n_features))

        mock_model.predict = MagicMock(
            return_value=rng.normal(0, 1, 100)
        )
        return mock_model, MagicMock(side_effect=mock_shap_values)

    @patch("src.models.shap_analysis.shap.TreeExplainer")
    def test_compute_shap_returns_correct_shape(
        self, mock_tree_explainer
    ):
        """SHAP values tienen shape (n_samples, n_features)."""
        n_features = 8
        n_samples = 50

        mock_model, mock_shap_fn = self._make_mock_model(
            n_features
        )
        mock_explainer = MagicMock()
        mock_explainer.shap_values = mock_shap_fn
        mock_tree_explainer.return_value = mock_explainer

        X = pd.DataFrame(
            np.random.default_rng(42).normal(
                0, 1, (n_samples, n_features)
            ),
            columns=[f"f{i}" for i in range(n_features)]
        )

        result = compute_shap_values(
            mock_model, X, n_sample=n_samples
        )

        assert len(result) == n_features
        assert "feature" in result.columns
        assert "shap_mean" in result.columns
        assert "shap_ratio" in result.columns

    @patch("src.models.shap_analysis.shap.TreeExplainer")
    def test_shap_ratio_max_is_one(
        self, mock_tree_explainer
    ):
        """El shap_ratio máximo debe ser 1.0."""
        mock_model, mock_shap_fn = self._make_mock_model(5)
        mock_explainer = MagicMock()
        mock_explainer.shap_values = mock_shap_fn
        mock_tree_explainer.return_value = mock_explainer

        X = pd.DataFrame(
            np.random.default_rng(42).normal(0, 1, (30, 5)),
            columns=[f"f{i}" for i in range(5)]
        )

        result = compute_shap_values(
            mock_model, X, n_sample=30
        )
        assert result["shap_ratio"].max() == pytest.approx(1.0)

    @patch("src.models.shap_analysis.shap.TreeExplainer")
    def test_subsampleo(
        self, mock_tree_explainer
    ):
        """Si n_sample < len(X), se submuestrea."""
        mock_model, mock_shap_fn = self._make_mock_model(3)
        mock_explainer = MagicMock()
        mock_explainer.shap_values = mock_shap_fn
        mock_tree_explainer.return_value = mock_explainer

        X = pd.DataFrame(
            np.random.default_rng(42).normal(0, 1, (100, 3)),
            columns=[f"f{i}" for i in range(3)]
        )

        compute_shap_values(mock_model, X, n_sample=20)
        # shap_values se llama con 20 muestras, no 100
        call_args = mock_explainer.shap_values.call_args[0][0]
        assert len(call_args) == 20

    @patch("src.models.shap_analysis.shap.TreeExplainer")
    def test_max_no_subsample(
        self, mock_tree_explainer
    ):
        """Si n_sample >= len(X), no se submuestrea."""
        mock_model, mock_shap_fn = self._make_mock_model(3)
        mock_explainer = MagicMock()
        mock_explainer.shap_values = mock_shap_fn
        mock_tree_explainer.return_value = mock_explainer

        X = pd.DataFrame(
            np.random.default_rng(42).normal(0, 1, (10, 3)),
            columns=[f"f{i}" for i in range(3)]
        )

        compute_shap_values(mock_model, X, n_sample=100)
        call_args = mock_explainer.shap_values.call_args[0][0]
        assert len(call_args) == 10
