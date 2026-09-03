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
    # Transacciones son por TIENDA (una serie por store, compartida entre
    # familias) — refleja el esquema real donde todas las familias de un
    # (store, date) comparten el mismo valor de 'transactions'.
    store_transactions = {
        store: rng.integers(100, 500, n_days)
        for store in stores
    }
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
                "transactions": store_transactions[store],
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


def _make_future_rows(df: pd.DataFrame, n_days: int = 7) -> pd.DataFrame:
    """Filas futuras (frontera de datos): transactions=0 y sales=0.

    Simula la ventana de predicción donde ya no existen transacciones
    reales (el dataset termina en df['date'].max()).
    """
    stores = df["store_nbr"].unique()
    families = df["family"].unique()
    future_dates = pd.date_range(
        df["date"].max() + pd.Timedelta(days=1), periods=n_days, freq="D"
    )
    rows = []
    for store in stores:
        for family in families:
            rows.append(pd.DataFrame({
                "date": future_dates,
                "store_nbr": store,
                "family": family,
                "sales": 0.0,
                "city": "Quito",
                "state": "Pichincha",
                "type": "A",
                "cluster": 1,
                "holiday_type": "No_Holiday",
                "holiday_description": "No_Holiday",
                "transferred": False,
                "dcoilwtico": 50.0,
                "onpromotion": 0,
                "transactions": 0.0,
            }))
    return pd.concat(rows, ignore_index=True)


class TestTransactionFeaturesDateAligned:
    """Regresión: las features de transacciones deben estar alineadas por fecha.

    Antes se calculaban con shift() posicional sobre el df completo, donde hay
    33 filas por (store_nbr, date) → retrocedían ~1 día en lugar del horizonte
    real. Esto colapsaba a 0 en la frontera futura y rompía la predicción.
    """

    def test_transaction_lag_is_date_aligned(self):
        """trans_lag_h de (store, D) == transactions de (store, D-h días)."""
        df = make_synthetic_df()
        fe = DemandFeatureEngineer(horizon=7)
        fe.fit(df)
        out = fe.transform(df, is_train=False)

        store = df["store_nbr"].iloc[0]
        fam_code = 0
        d = df["date"].max() - pd.Timedelta(days=10)
        expected = df[
            (df["store_nbr"] == store) & (df["date"] == d - pd.Timedelta(days=7))
        ]["transactions"].iloc[0]
        got = out[
            (out["store_nbr"] == store)
            & (out["family"] == fam_code)
            & (out["date"] == d)
        ]["trans_lag_7"].iloc[0]
        assert np.isclose(got, expected)

    def test_transaction_lag_does_not_collapse_at_future_boundary(self):
        """En la frontera futura trans_lag_h NO debe colapsar a 0.

        Debe igualar las transacciones reales de hace `horizon` días.
        """
        df = make_synthetic_df()
        fe = DemandFeatureEngineer(horizon=7)
        fe.fit(df)
        combined = pd.concat(
            [df, _make_future_rows(df, 7)], ignore_index=True
        )
        out = fe.transform(combined, is_train=False)

        future_start = df["date"].max() + pd.Timedelta(days=1)
        future_rows = out[out["date"] >= future_start]
        assert len(future_rows) > 0
        assert future_rows["trans_lag_7"].notna().all()
        assert (future_rows["trans_lag_7"] > 0).all()

    def test_transaction_features_are_store_level(self):
        """Todas las familias de un mismo (store, date) comparten el mismo lag."""
        df = make_synthetic_df()
        fe = DemandFeatureEngineer(horizon=7)
        fe.fit(df)
        out = fe.transform(df, is_train=False)
        nonnull = out.dropna(subset=["trans_lag_7"])
        nunique = nonnull.groupby(["store_nbr", "date"])["trans_lag_7"].nunique()
        assert (nunique == 1).all()


class TestOilFeaturesDateAligned:
    """Regresión: las features de petróleo deben estar alineadas por fecha.

    Antes se calculaban con shift() posicional sobre el df completo (33 filas
    por store/date), retrocediendo ~0.2 días en vez del lag real. Ahora se
    deduplica a serie store-level antes de shift.
    """

    def test_oil_lag_is_date_aligned(self):
        """oil_lag_h de (store, D) == dcoilwtico de (store, D-h días)."""
        df = make_synthetic_df()
        fe = DemandFeatureEngineer(horizon=7)
        fe.fit(df)
        out = fe.transform(df, is_train=False)

        store = df["store_nbr"].iloc[0]
        d = df["date"].max() - pd.Timedelta(days=10)
        expected = df[
            (df["store_nbr"] == store) & (df["date"] == d - pd.Timedelta(days=7))
        ]["dcoilwtico"].iloc[0]
        got = out[
            (out["store_nbr"] == store)
            & (out["family"] == 0)
            & (out["date"] == d)
        ]["oil_lag_7"].iloc[0]
        assert np.isclose(got, expected)

    def test_oil_lag_does_not_collapse_at_future_boundary(self):
        """En la frontera futura oil_lag_h NO debe colapsar a 0."""
        df = make_synthetic_df()
        fe = DemandFeatureEngineer(horizon=7)
        fe.fit(df)
        combined = pd.concat(
            [df, _make_future_rows(df, 7)], ignore_index=True
        )
        out = fe.transform(combined, is_train=False)

        future_start = df["date"].max() + pd.Timedelta(days=1)
        future_rows = out[out["date"] >= future_start]
        assert len(future_rows) > 0
        assert future_rows["oil_lag_7"].notna().all()
        assert (future_rows["oil_lag_7"] > 0).all()

    def test_oil_features_are_store_level(self):
        """Todas las familias de un mismo (store, date) comparten el mismo oil_lag."""
        df = make_synthetic_df()
        fe = DemandFeatureEngineer(horizon=7)
        fe.fit(df)
        out = fe.transform(df, is_train=False)
        nonnull = out.dropna(subset=["oil_lag_7"])
        nunique = nonnull.groupby(["store_nbr", "date"])["oil_lag_7"].nunique()
        assert (nunique == 1).all()


class TestFeatureIntegrity:
    """Integridad del pipeline: cardinalidad y paridad train/serving."""

    def test_no_row_fanout_after_merge(self, fitted_pipeline):
        """Los merges (stateful + store-level) no deben duplicar ni perder filas.

        El merge de transacciones/oil se hace sobre (store_nbr, date) único en
        `uniq`; si la clave dejara de ser única, habría fan-out (más de
        n_families filas por (store, date)) o pérdida de series.
        """
        fe, df = fitted_pipeline
        out = fe.transform(df, is_train=False)

        n_families = df["family"].nunique()
        assert len(out) == len(df), "El transform altera el número de filas"
        counts = out.groupby(["store_nbr", "date"])["family"].nunique()
        assert (counts == n_families).all(), (
            f"Fan-out detectado: se esperan {n_families} familias por (store, date)"
        )
        assert not out.duplicated(
            subset=["store_nbr", "date", "family"]
        ).any(), "Existen filas duplicadas por (store, date, family)"

    def test_is_train_parity_feature_values(self, fitted_pipeline):
        """Los features son idénticos con is_train=True y False (misma entrada).

        La única diferencia permitida es que is_train=True elimina las filas
        con lags nulos (dropna). Sobre las filas compartidas, los valores de
        TODOS los features deben coincidir exactamente (garantía de paridad).
        """
        fe, df = fitted_pipeline
        out_train = fe.transform(df, is_train=True)
        out_serving = fe.transform(df, is_train=False)

        keys = ["store_nbr", "family", "date"]
        feat_cols = [c for c in out_train.columns if c not in keys + ["sales"]]

        merged = out_train[keys + feat_cols].merge(
            out_serving[keys + feat_cols], on=keys, suffixes=("_t", "_s")
        )
        assert len(merged) == len(out_train), (
            "Filas de train no encuentan su par en serving"
        )

        for c in feat_cols:
            a = merged[f"{c}_t"]
            b = merged[f"{c}_s"]
            if a.dtype == "category":
                # Las categoricals (p. ej. holiday_impact_type) se comparan
                # por código de categoría (mismo vocabulario en train y serving).
                assert (a.cat.codes.fillna(-1) == b.cat.codes.fillna(-1)).all(), (
                    f"Feature '{c}' difiere entre train y serving"
                )
            else:
                assert (a.fillna(-999) == b.fillna(-999)).all(), (
                    f"Feature '{c}' difiere entre train y serving"
                )
