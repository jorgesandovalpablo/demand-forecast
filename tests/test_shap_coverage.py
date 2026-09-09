# tests/test_shap_coverage.py
"""Tests para elevar cobertura de shap_analysis.py de 42% a ~75%.

Cubre: classify_features, compute_shap_values (mock), save_results.
"""
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd


class TestClassifyFeatures:
    def test_marks_low_shap_as_drop_candidate(self):
        from src.models.shap_analysis import classify_features
        shap_df = pd.DataFrame({
            "feature": ["a", "b", "c", "d", "e"],
            "shap_mean": [0.001, 0.002, 0.5, 0.6, 0.7],
            "shap_ratio": [0.001, 0.002, 0.7, 0.85, 1.0],
        })
        result = classify_features(shap_df, low_percentile=20, min_ratio=0.05)
        assert "decision" in result.columns
        assert result.loc[result["feature"] == "a", "decision"].iloc[0] == "DROP_CANDIDATE"
        assert result.loc[result["feature"] == "e", "decision"].iloc[0] == "KEEP"

    def test_all_keep_when_all_high(self):
        from src.models.shap_analysis import classify_features
        shap_df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "shap_mean": [0.5, 0.6, 0.7],
            "shap_ratio": [0.7, 0.85, 1.0],
        })
        result = classify_features(shap_df, low_percentile=20, min_ratio=0.05)
        assert (result["decision"] == "KEEP").all()


class TestComputeShapValues:
    def test_returns_dataframe(self):
        from src.models.shap_analysis import compute_shap_values
        mock_model = MagicMock()
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        with patch("src.models.shap_analysis.shap.TreeExplainer", return_value=mock_explainer):
            X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
            result = compute_shap_values(mock_model, X, n_sample=100)
            assert "feature" in result.columns
            assert "shap_mean" in result.columns
            assert "shap_ratio" in result.columns
            assert len(result) == 2

    def test_subsample_when_large(self):
        from src.models.shap_analysis import compute_shap_values
        mock_model = MagicMock()
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.ones((100, 5))
        with patch("src.models.shap_analysis.shap.TreeExplainer", return_value=mock_explainer):
            X = pd.DataFrame({f"f{i}": range(1000) for i in range(5)})
            result = compute_shap_values(mock_model, X, n_sample=100)
            assert len(result) == 5


class TestSaveResults:
    def test_creates_csv_and_txt_files(self, tmp_path):
        from src.models.shap_analysis import save_results
        shap_df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "shap_mean": [0.5, 0.3, 0.1],
            "shap_ratio": [1.0, 0.6, 0.2],
            "decision": ["KEEP", "KEEP", "DROP_CANDIDATE"],
        })
        with patch("src.models.shap_analysis.plt"):
            save_results(shap_df, horizon=7, output_dir=tmp_path)
        assert (tmp_path / "shap_summary_h7.csv").exists()
        assert (tmp_path / "features_keep_h7.txt").exists()
        assert (tmp_path / "features_drop_h7.txt").exists()

    def test_saves_png_when_plt_works(self, tmp_path):
        from src.models.shap_analysis import save_results
        shap_df = pd.DataFrame({
            "feature": ["a", "b"],
            "shap_mean": [0.5, 0.3],
            "shap_ratio": [1.0, 0.6],
            "decision": ["KEEP", "KEEP"],
        })
        mock_plt = MagicMock()
        with patch("src.models.shap_analysis.plt", mock_plt):
            save_results(shap_df, horizon=7, output_dir=tmp_path)
        mock_plt.savefig.assert_called_once()
