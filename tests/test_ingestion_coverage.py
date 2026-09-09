import pandas as pd
import pytest
from pathlib import Path
from src.data.ingestion import (
    _validate_columns,
    _validate_nulls,
    _validate_schema,
    _load_csv,
    load_raw_data,
    SCHEMAS,
)


def _make_csv(path: Path, name: str, rows: int = 3) -> None:
    """Genera un CSV mínimo que cumple el esquema esperado."""
    schema = SCHEMAS[name]
    data: dict = {}
    for col in schema["columns"]:
        if col in ("date",):
            data[col] = pd.date_range("2026-01-01", periods=rows)
        elif col in ("id", "store_nbr", "onpromotion", "transactions", "cluster"):
            data[col] = [1] * rows
        elif col == "sales":
            data[col] = [10.0] * rows
        elif col == "dcoilwtico":
            data[col] = [70.0] * rows
        elif col == "transferred":
            data[col] = [False] * rows
        else:
            data[col] = ["test"] * rows
    pd.DataFrame(data).to_csv(path / f"{name}.csv", index=False)


class TestValidateColumns:
    def test_happy_path(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        _validate_columns(df, "x", ["a", "b"])

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="faltantes"):
            _validate_columns(df, "x", ["a", "b"])

    def test_extra_column_warns(self) -> None:
        df = pd.DataFrame({"a": [1], "extra": [2]})
        _validate_columns(df, "x", ["a"])


class TestValidateNulls:
    def test_with_nulls(self) -> None:
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        _validate_nulls(df, "test")

    def test_without_nulls(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0]})
        _validate_nulls(df, "test")


class TestValidateSchema:
    def test_valid_schema(self) -> None:
        df = pd.DataFrame({"id": [1], "date": pd.to_datetime(["2026-01-01"])})
        _validate_schema(df, "test_simple", {"columns": ["id", "date"]})

    def test_invalid_schema_raises(self) -> None:
        df = pd.DataFrame({"id": [1]})
        with pytest.raises(ValueError, match="faltantes"):
            _validate_schema(df, "x", {"columns": ["id", "date"]})


class TestLoadCsv:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_csv("train", tmp_path)

    def test_loads_valid_csv(self, tmp_path: Path) -> None:
        _make_csv(tmp_path, "train")
        df = _load_csv("train", tmp_path)
        assert len(df) == 3
        assert "date" in df.columns


class TestLoadRawData:
    def test_predict_false(self, tmp_path: Path) -> None:
        for name in SCHEMAS:
            _make_csv(tmp_path, name)
        data = load_raw_data(data_path=str(tmp_path), predict=False)
        assert set(data.keys()) == {
            "train", "test", "stores", "oil",
            "holidays", "transactions",
        }
        assert len(data["train"]) == 3

    def test_predict_true(self, tmp_path: Path) -> None:
        for name in SCHEMAS:
            _make_csv(tmp_path, name)
        data = load_raw_data(data_path=str(tmp_path), predict=True)
        assert set(data.keys()) == {
            "stores", "oil", "holidays", "transactions",
        }
        assert "train" not in data
        assert "test" not in data
