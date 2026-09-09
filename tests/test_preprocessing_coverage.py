import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.data.preprocessing import (
    _process_holidays,
    _merge_datasets,
    _handle_nulls,
    _transform_target,
    _reduce_memory,
    run_preprocessing,
    _save_processed,
)
from src.utils.config import config


def _fake_holidays() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
        ]),
        "type": ["Holiday", "Transfer", "Holiday", "Event"],
        "locale": ["National", "Regional", "Local", "National"],
        "locale_name": ["Ecuador", "Pichincha", "Quito", "Ecuador"],
        "description": ["Año Nuevo", "Feriado", "Local", "Evento"],
        "transferred": [False, True, False, False],
    })


def _fake_stores() -> pd.DataFrame:
    return pd.DataFrame({
        "store_nbr": [1],
        "city": ["Quito"],
        "state": ["Pichincha"],
        "type": ["A"],
        "cluster": [1],
    })


def _fake_oil() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "dcoilwtico": [70.0, np.nan, 72.0],
    })


def _fake_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01"]),
        "store_nbr": [1],
        "transactions": [100],
    })


def _fake_train() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1],
        "date": pd.to_datetime(["2026-01-01"]),
        "store_nbr": [1],
        "family": ["GROCERY I"],
        "sales": [100.0],
        "onpromotion": [0],
    })


def _fake_test() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [2],
        "date": pd.to_datetime(["2026-01-01"]),
        "store_nbr": [1],
        "family": ["GROCERY I"],
        "onpromotion": [0],
    })


class TestProcessHolidays:
    def test_returns_three_groups(self) -> None:
        holidays = _fake_holidays()
        nat, reg, loc = _process_holidays(holidays)
        assert len(nat) >= 1
        assert len(reg) >= 1
        assert len(loc) >= 1

    def test_national_locale_column(self) -> None:
        holidays = _fake_holidays()
        nat, _, _ = _process_holidays(holidays)
        assert "holiday_type" in nat.columns
        assert "holiday_description" in nat.columns

    def test_regional_renames_to_state(self) -> None:
        holidays = _fake_holidays()
        _, reg, _ = _process_holidays(holidays)
        assert "state" in reg.columns

    def test_local_renames_to_city(self) -> None:
        holidays = _fake_holidays()
        _, _, loc = _process_holidays(holidays)
        assert "city" in loc.columns


class TestMergeDatasets:
    def test_merges_all_sources(self) -> None:
        train = _fake_train()
        result = _merge_datasets(
            train, _fake_stores(), _fake_oil(),
            _fake_transactions(), _fake_holidays(),
        )
        assert len(result) == 1
        assert "dcoilwtico" in result.columns
        assert "transactions" in result.columns

    def test_oil_interpolated(self) -> None:
        train = _fake_train()
        result = _merge_datasets(
            train, _fake_stores(), _fake_oil(),
            _fake_transactions(), _fake_holidays(),
        )
        assert result["dcoilwtico"].isnull().sum() == 0


class TestHandleNulls:
    def test_transactions_fills_with_zero(self) -> None:
        df = pd.DataFrame({
            "dcoilwtico": [70.0],
            "transactions": [np.nan],
            "holiday_type": [np.nan],
            "holiday_description": [np.nan],
            "transferred": [np.nan],
        })
        result = _handle_nulls(df)
        assert result["transactions"].iloc[0] == 0
        assert result["holiday_type"].iloc[0] == "No_Holiday"
        assert result["transferred"].iloc[0] == False

    def test_oil_ffill_fallback(self) -> None:
        df = pd.DataFrame({
            "dcoilwtico": [70.0, np.nan],
            "transactions": [0, 0],
            "holiday_type": ["x", "y"],
            "holiday_description": ["a", "b"],
            "transferred": [True, False],
        })
        result = _handle_nulls(df)
        assert result["dcoilwtico"].isnull().sum() == 0


class TestTransformTarget:
    def test_applies_log1p(self) -> None:
        df = pd.DataFrame({config["data"]["target"]: [10.0, 100.0]})
        result = _transform_target(df)
        raw_col = f"{config['data']['target']}_raw"
        assert raw_col in result.columns
        np.testing.assert_allclose(
            result[config["data"]["target"]],
            np.log1p([10.0, 100.0]),
        )


class TestReduceMemory:
    def test_reduces_types(self) -> None:
        df = pd.DataFrame({
            "float_col": [1.0, 2.0],
            "int_col": [1, 2],
            "obj_col": ["a", "b"],
        })
        result = _reduce_memory(df)
        assert result["float_col"].dtype == np.float32
        assert result["int_col"].dtype == np.int32
        assert result["obj_col"].dtype.name == "category"


class TestRunPreprocessing:
    def test_predict_false(self) -> None:
        data = {
            "train": _fake_train(),
            "test": _fake_test(),
            "stores": _fake_stores(),
            "oil": _fake_oil(),
            "holidays": _fake_holidays(),
            "transactions": _fake_transactions(),
        }
        train, test = run_preprocessing(data, save=False, predict=False)
        assert len(train) > 0
        assert "sales_log1p" in train.columns or config["data"]["target"] in train.columns

    def test_predict_true(self) -> None:
        data = {
            "train": _fake_train(),
            "test": _fake_test(),
            "stores": _fake_stores(),
            "oil": _fake_oil(),
            "holidays": _fake_holidays(),
            "transactions": _fake_transactions(),
        }
        train, test = run_preprocessing(data, save=False, predict=True)
        assert len(test) > 0
        assert len(train) == 0


class TestSaveProcessed:
    def test_creates_parquet_files(self, tmp_path: Path) -> None:
        train = pd.DataFrame({"a": [1]})
        test = pd.DataFrame({"b": [2]})
        with patch("src.data.preprocessing.Path") as mock_path:
            mock_path.return_value = tmp_path
            _save_processed(train, test)
        assert (tmp_path / "train_processed.parquet").exists()
        assert (tmp_path / "test_processed.parquet").exists()
