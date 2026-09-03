"""
Tests de imputación de petróleo en preprocessing.

Verifica que la interpolación de dcoilwtico se realice
ANTES del merge (cuando oil tiene 1 fila por fecha) y no
después (cuando ya tiene 33 filas por fecha por familia).

Bug: interpolate() trata 33 filas NaN como 33 pasos temporales
→ diferentes valores por familia en la misma fecha.
"""
import numpy as np
import pandas as pd


def _make_oil_series() -> pd.DataFrame:
    """
    Crea una serie de petróleo con gaps de fin de semana.

    Lunes a viernes con valores, sábados y domingos ausentes
    (como en el CSV real del mercado ecuatoriano).
    """
    dates = pd.date_range("2016-01-04", periods=11, freq="D")
    values = [95.0, 96.0, 97.0, 98.0, 99.0,
              np.nan, np.nan,
              100.0, 101.0, 102.0, 103.0]
    return pd.DataFrame({"date": dates, "dcoilwtico": values})


def _make_main_df(n_families: int = 33) -> pd.DataFrame:
    """
    Crea un DataFrame principal con n_families familias
    por fecha (simula el esquema real de 54 tiendas × 33 familias).
    """
    dates = pd.date_range("2016-01-04", periods=11, freq="D")
    families = [f"F{i}" for i in range(n_families)]

    rows = []
    for date in dates:
        for fam in families:
            rows.append({"date": date, "family": fam, "store_nbr": 1})

    return pd.DataFrame(rows)


class TestOilInterpolationBeforeMerge:
    """Verifica que la interpolación ocurre antes del merge."""

    def test_all_families_same_oil_weekend(self) -> None:
        """Todas las familias deben tener el MISMO valor de petróleo
        en días de fin de semana (oil es global, no por familia)."""
        oil = _make_oil_series()
        df = _make_main_df(n_families=5)

        # Interpolar antes del merge (como hace _merge_datasets)
        oil = oil.sort_values("date").copy()
        oil["dcoilwtico"] = (
            oil["dcoilwtico"]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )

        merged = df.merge(oil, on="date", how="left")

        # En sábado 2016-01-09: todas las familias deben tener el mismo valor
        sat_rows = merged[merged["date"] == "2016-01-09"]
        oil_values_sat = sat_rows["dcoilwtico"].unique()
        assert len(oil_values_sat) == 1, (
            f"Sábado debe tener 1 valor único, got {len(oil_values_sat)}: "
            f"{oil_values_sat}"
        )

        # En domingo 2016-01-10: todas las familias deben tener el mismo valor
        sun_rows = merged[merged["date"] == "2016-01-10"]
        oil_values_sun = sun_rows["dcoilwtico"].unique()
        assert len(oil_values_sun) == 1, (
            f"Domingo debe tener 1 valor único, got {len(oil_values_sun)}: "
            f"{oil_values_sun}"
        )

    def test_weekend_interpolation_midpoint(self) -> None:
        """Los valores de fin de semana deben ser el midpoint correcto
        entre viernes y lunes (interpolación lineal temporal)."""
        oil = _make_oil_series()
        oil = oil.sort_values("date").copy()
        oil["dcoilwtico"] = (
            oil["dcoilwtico"]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )

        # Viernes 2016-01-08 = 99.0, Lunes 2016-01-11 = 100.0
        # Sábado debe ser 99.333, Domingo debe ser 99.667
        sat_val = oil.loc[oil["date"] == "2016-01-09", "dcoilwtico"].iloc[0]
        sun_val = oil.loc[oil["date"] == "2016-01-10", "dcoilwtico"].iloc[0]

        assert abs(sat_val - 99.333) < 0.01, (
            f"Sábado debe ser ~99.333, got {sat_val}"
        )
        assert abs(sun_val - 99.667) < 0.01, (
            f"Domingo debe ser ~99.667, got {sun_val}"
        )

    def test_no_residual_nan_after_interpolation(self) -> None:
        """No debe quedar ningún NaN después de la interpolación."""
        oil = _make_oil_series()
        oil = oil.sort_values("date").copy()
        oil["dcoilwtico"] = (
            oil["dcoilwtico"]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )

        assert oil["dcoilwtico"].isnull().sum() == 0, (
            f"Quedan {oil['dcoilwtico'].isnull().sum()} NaN "
            f"después de interpolación"
        )
