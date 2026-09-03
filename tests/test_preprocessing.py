"""
Tests de imputación de petróleo en preprocessing.

Ejercitan el CÓDIGO REAL `_merge_datasets` (no una réplica), con data
sintética que replica el esquema del dataset real de Store Sales:

- `df`: cada día del rango, con 33 familias por (store, date).
- `oil`: SOLO días hábiles (sábados y domingos ausentes como en el CSV
  ecuatoriano) + algún NaN en día hábil.

Bug cubierto: la interpolación debe hacerse ANTES del merge y tras
reindexar al rango diario completo; si no, los fines de semana quedan
como NaN o con valores diferentes por familia.
"""
import numpy as np
import pandas as pd

from src.data.preprocessing import _merge_datasets


def _dates(start: str = "2016-01-04", periods: int = 11) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="D")


def _make_oil(weekdays_only: bool = True) -> pd.DataFrame:
    """
    Serie de petróleo realista: solo días hábiles (fin de semana ausente),
    con un NaN en un día hábil (festivo donde el mercado no cotiza).
    """
    dates = _dates()
    rows = []
    for i, date in enumerate(dates):
        if weekdays_only and date.dayofweek >= 5:
            continue  # sábado/domingo NO existen (como el CSV real)
        rows.append({"date": date, "dcoilwtico": 95.0 + i})
    oil = pd.DataFrame(rows)
    # Marcar el jueves 2016-01-07 (índice 3) como NaN (festivo)
    oil.loc[oil["date"] == pd.Timestamp("2016-01-07"), "dcoilwtico"] = np.nan
    return oil


def _make_stores(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "store_nbr": list(range(1, n + 1)),
        "city": ["Quito"] * n,
        "state": ["Pichincha"] * n,
        "type": ["A"] * n,
        "cluster": [1] * n,
    })


def _make_transactions(n_stores: int = 1) -> pd.DataFrame:
    dates = _dates()
    rows = []
    for date in dates:
        for store in range(1, n_stores + 1):
            rows.append({"date": date, "store_nbr": store,
                         "transactions": 100 + store})
    return pd.DataFrame(rows)


def _make_holidays_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date", "locale", "type", "locale_name", "description", "transferred"
    ])


def _make_train(n_families: int = 33, n_stores: int = 1) -> pd.DataFrame:
    dates = _dates()
    families = [f"F{i}" for i in range(n_families)]
    rows = []
    for date in dates:
        for store in range(1, n_stores + 1):
            for fam in families:
                rows.append({"date": date, "store_nbr": store, "family": fam})
    return pd.DataFrame(rows)


def _run_merge() -> pd.DataFrame:
    """Ejecuta el merge real con los 5 inputs del pipeline."""
    return _merge_datasets(
        df=_make_train(),
        stores=_make_stores(),
        oil=_make_oil(),
        transactions=_make_transactions(),
        holidays=_make_holidays_empty(),
    )


class TestOilInterpolationBeforeMerge:
    """Verifica que el merge real produce oil global e interpolado."""

    def test_all_families_same_oil_weekend(self) -> None:
        """Todas las familias deben tener el MISMO valor de petróleo en un
        mismo (store, date), incluidos los fines de semana."""
        merged = _run_merge()

        weekend = merged[merged["date"].dt.dayofweek >= 5]
        assert len(weekend) > 0, "debe haber filas de fin de semana tras el merge"

        nunique = weekend.groupby(["store_nbr", "date"])["dcoilwtico"].nunique()
        assert (nunique == 1).all(), (
            f"Fines de semana con >1 valor de oil: "
            f"{(nunique > 1).sum()} de {len(nunique)}"
        )

    def test_weekend_interpolation_midpoint(self) -> None:
        """Sábado y domingo deben ser el midpoint lineal entre viernes y lunes.

        Sáb 2016-01-09 y Dom 2016-01-10 entre Vie 2016-01-08 (99.0)
        y Lun 2016-01-11 (102.0): sábado ≈ 100.0, domingo ≈ 101.0.
        """
        merged = _run_merge()
        vals = (
            merged[merged["date"].isin(pd.to_datetime(
                ["2016-01-08", "2016-01-09", "2016-01-10", "2016-01-11"]
            ))]
            .groupby("date")["dcoilwtico"]
            .first()
        )
        assert abs(vals.loc[pd.Timestamp("2016-01-09")] - 100.0) < 0.01
        assert abs(vals.loc[pd.Timestamp("2016-01-10")] - 101.0) < 0.01

    def test_no_residual_nan_after_interpolation(self) -> None:
        """No debe quedar ningún NaN de oil tras el merge."""
        merged = _run_merge()
        assert merged["dcoilwtico"].isnull().sum() == 0, (
            f"Quedan {merged['dcoilwtico'].isnull().sum()} NaN de oil"
        )

    def test_holiday_nan_also_interpolated(self) -> None:
        """El NaN de festivo (jueves 2016-01-07) también se rellena."""
        merged = _run_merge()
        assert merged["dcoilwtico"].notna().all()
