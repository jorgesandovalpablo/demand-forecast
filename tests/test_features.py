"""
Tests unitarios del pipeline de feature engineering.

Objetivo de MLOps: garantizar la PARIDAD TRAIN/SERVING.
`DemandFeatureEngineer` es el único componente autorizado a
transformar datos tanto en `train.py` como en serving (`api/`,
`predict.py`). Si estos tests fallan, cualquier modelo entrenado
y las inferencias servidas dejarían de ser comparables.

Estrategia: dataset sintético determinista (semilla fija) que
replica el esquema mínimo de `train_processed.parquet`, sin
depender de datos crudos externos (CI reproducible y rápida).
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import DemandFeatureEngineer


def make_synthetic_df(n_days: int = 90, n_stores: int = 2, n_families: int = 2) -> pd.DataFrame:
    """
    Genera un dataset sintético con el esquema del histórico procesado.

    Determinista (rng semilla 42) para que los asserts sobre
    medias/rankings sean estables entre ejecuciones.

    Args:
        n_days: días de historial por serie (>=28 para lags del horizonte 7).
        n_stores: número de tiendas simuladas.
        n_families: familias de producto simuladas (máx. 2).

    Returns:
        DataFrame con columnas: date, store_nbr, family, sales,
        city, state, type, cluster, holiday_type,
        holiday_description, transferred, dcoilwtico,
        onpromotion, transactions.
    """
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    stores = range(1, n_stores + 1)
    families = ["GROCERY I", "BEVERAGES"][:n_families]

    rows = []
    rng = np.random.default_rng(42)
    for store in stores:
        for fam_idx, family in enumerate(families):
            base = 10.0 * store + 5.0 * fam_idx
            sales = base + rng.normal(0, 1.0, n_days).clip(min=0)
            rows.append(pd.DataFrame({
                "date": dates,
                "store_nbr": store,
                "family": family,
                "sales": sales,
                "city": "Quito",
                "state": "Pichincha",
                "type": "A",
                "cluster": 1,
                "holiday_type": "No_Holiday",
                "holiday_description": "No_Holiday",
                "transferred": False,
                "dcoilwtico": 50.0,
                "onpromotion": rng.integers(0, 5, n_days),
                "transactions": rng.integers(100, 500, n_days),
            }))

    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def fitted_pipeline() -> tuple[DemandFeatureEngineer, pd.DataFrame]:
    """Pipeline ajustado una sola vez por test (fit + df de entrada)."""
    df = make_synthetic_df()
    fe = DemandFeatureEngineer(horizon=7)
    fe.fit(df)
    return fe, df


class TestFit:
    """Estado aprendido en fit(): debe ser global y determinista."""

    def test_fit_learns_category_mapping(self, fitted_pipeline):
        """El mapeo de categorías se congela ordenado alfabéticamente (cat.codes)."""
        fe, _ = fitted_pipeline
        assert fe.is_fitted
        assert list(fe.categories_mapping["family"]) == ["BEVERAGES", "GROCERY I"]

    def test_transform_without_fit_raises(self):
        """Garantía fail-fast: no se sirve nada con estado sin aprender."""
        fe = DemandFeatureEngineer(horizon=7)
        with pytest.raises(ValueError):
            fe.transform(make_synthetic_df())

    def test_store_stats_cover_all_series(self, fitted_pipeline):
        """Debe existir una fila de stats por cada serie tienda-familia."""
        fe, df = fitted_pipeline
        n_series = df[["store_nbr", "family"]].drop_duplicates().shape[0]
        assert len(fe.store_stats) == n_series

    def test_store_ranking_is_global(self, fitted_pipeline):
        """El ranking se calcula sobre TODO el histórico, no sobre subsets.

        Regresión cubierta: en serving el rank colapsaba a 1 al filtrar
        por tienda antes de transformar.
        """
        fe, df = fitted_pipeline
        expected_top_store = (
            df.groupby("store_nbr")["sales"].sum().idxmax()
        )
        assert fe.store_ranking[expected_top_store] == 1


class TestTransformParity:
    """Paridad train/serving: transform() debe ser puro respecto al fit()."""

    def test_encoding_reproducible(self, fitted_pipeline):
        """Dos llamadas a transform() producen encoding idéntico."""
        fe, df = fitted_pipeline
        out_a = fe.transform(df, is_train=False)
        out_b = fe.transform(df, is_train=False)
        assert out_a["family"].tolist() == out_b["family"].tolist()

    def test_family_codes_match_fit_categories(self, fitted_pipeline):
        """Los códigos post-transform son decodificables con las categorías del fit.

        Se compara por clave (store_nbr, date) porque merge() resetea el
        índice del DataFrame y la alineación posicional no es fiable.
        """
        fe, df = fitted_pipeline
        out = fe.transform(df, is_train=False)
        cats = list(fe.categories_mapping["family"])

        # Los códigos deben pertenecer al vocabulario aprendido en fit
        assert set(out["family"].unique()).issubset(range(len(cats)))

        original_sets = (
            df.assign(key=df["store_nbr"].astype(str) + "_" + df["date"].astype(str))
            .groupby("key")["family"]
            .agg(set)
        )
        decoded_sets = (
            out.assign(key=out["store_nbr"].astype(str) + "_" + out["date"].astype(str))
            .groupby("key")["family"]
            .agg(lambda s: set(s.map(lambda c: cats[c])))
        )
        # Decodificar codes → nombres debe recuperar las familias originales
        assert decoded_sets.sort_index().tolist() == original_sets.sort_index().tolist()

    def test_stateful_features_injected(self, fitted_pipeline):
        """Las stats históricas completas se inyectan aunque el subset sea parcial.

        Regresión cubierta: en serving se recalculaban stats sobre la
        ventana de inferencia en lugar del historial completo del fit().
        """
        fe, df = fitted_pipeline
        out = fe.transform(df, is_train=False)
        assert "venta_media_historica" in out.columns
        assert "venta_std_historica" in out.columns
        assert "ranking_tienda" in out.columns

        # La media inyectada corresponde a la serie decodificada correcta
        stats_lookup = fe.store_stats.set_index(["store_nbr", "family"])
        sample = out.iloc[0]
        expected_mean = stats_lookup.loc[
            (sample["store_nbr"], fe.categories_mapping["family"][sample["family"]]),
            "venta_media_historica",
        ]
        assert np.isclose(sample["venta_media_historica"], expected_mean)

    def test_transform_does_not_mutate_input(self, fitted_pipeline):
        """transform() es side-effect free: el DataFrame de entrada queda intacto."""
        fe, df = fitted_pipeline
        before = df.copy()
        fe.transform(df, is_train=False)
        assert df.equals(before)

    def test_dcoilwtico_dropped(self, fitted_pipeline):
        """El precio del petróleo crudo se elimina para evitar data leakage."""
        fe, df = fitted_pipeline
        out = fe.transform(df, is_train=False)
        assert "dcoilwtico" not in out.columns
