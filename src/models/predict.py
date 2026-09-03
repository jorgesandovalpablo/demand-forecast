import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import config
from src.data.ingestion import load_raw_data
from src.data.preprocessing import run_preprocessing

logger = get_logger(__name__)


# ─────────────────────────────────────────
# 0. Cache de datos raw (predict=True)
# ─────────────────────────────────────────
_RAW_PREDICT_CACHE: dict = {}


def _load_raw_predict_cached() -> dict:
    """
    Carga los datos raw para el modo predict=True con cache.

    Los CSVs de apoyo (stores, oil, holidays, transactions) son
    estáticos y determinísticos, así que leerlos en cada llamada es
    trabajo repetido. Se cargan una única vez y se reutilizan.

    Se retorna un dict nuevo (shallow copy) referenciando los mismos
    DataFrames cacheados; los DataFrames subyacentes nunca se mutan
    durante el merge, por lo que es seguro compartirlos entre llamadas.
    """
    if '_raw' not in _RAW_PREDICT_CACHE:
        logger.info("Cargando datos raw (predict=True) por primera vez...")
        _RAW_PREDICT_CACHE['_raw'] = load_raw_data(predict=True)
    return dict(_RAW_PREDICT_CACHE['_raw'])


# ─────────────────────────────────────────
# 1. Carga del modelo
# ─────────────────────────────────────────
class ModelRegistry:
    """
    Gestiona la carga y caché de modelos.
    Evita recargar el modelo en cada predicción
    — crítico para performance en la API.
    """
    _models: dict = {}
    _features: dict = {}
    _pipelines: dict = {}
    _std: dict = {}

    @classmethod
    def load(cls, horizon: int) -> object:
        """
        Carga el modelo si no está en caché.
        Si ya está cargado, lo retorna directo.

        Estrategia local-primero: usa los artefactos de models/;
        si falta alguno intenta recuperarlos desde la versión
        'production' del MLflow Model Registry antes de fallar.
        """
        if horizon not in cls._models:
            model_path = Path(
                f"models/lgbm_h{horizon}.pkl"
            )
            features_path = Path(
                f"models/features_h{horizon}.pkl"
            )
            pipeline_path = Path(
                f"models/feature_pipeline_h{horizon}.pkl"
            )

            artifacts_missing = not all([
                model_path.exists(),
                features_path.exists(),
                pipeline_path.exists()
            ])

            if artifacts_missing:
                logger.warning(
                    f"Artefactos locales incompletos para "
                    f"horizon={horizon}; consultando el registry..."
                )
                from src.models.registry import ensure_local_artifacts
                if not ensure_local_artifacts(horizon):
                    raise FileNotFoundError(
                        f"Modelo no disponible para horizonte {horizon}: "
                        f"sin artefactos locales y sin versión "
                        f"'production' en el registry. "
                        f"Ejecuta train.py --horizon {horizon}"
                    )

            logger.info(
                f"Cargando modelo horizon={horizon} "
                f"desde {model_path}..."
            )
            cls._models[horizon] = joblib.load(model_path)
            cls._features[horizon] = joblib.load(features_path)
            cls._pipelines[horizon] = joblib.load(pipeline_path)
            cls._load_residual_std(horizon, pipeline_path)
            logger.info(
                f"Modelo horizon={horizon} cargado |"
                f"Features: {len(cls._features[horizon])}"
            )

        return cls._models[horizon]

    @classmethod
    def _load_residual_std(
        cls, horizon: int, pipeline_path: Path
    ) -> None:
        """
        Carga el σ del error residual (escala log) persistido por evaluate.

        Si el archivo no existe, usa como fallback el σ de la volatilidad
        histórica del pipeline (store_stats.venta_std_historica), con el
        mismo formato {'global', 'df'}.
        """
        std_path = Path(f"models/residual_std_h{horizon}.pkl")
        if std_path.exists():
            cls._std[horizon] = joblib.load(std_path)
            logger.info(
                f"σ residual cargado: global="
                f"{cls._std[horizon].get('global', 0.0):.4f}"
            )
            return

        logger.warning(
            f"models/residual_std_h{horizon}.pkl no encontrado; "
            f"usando σ histórico del pipeline como fallback."
        )
        pipeline = joblib.load(pipeline_path)
        std_by_group = pipeline.store_stats[
            ['store_nbr', 'family', 'venta_std_historica']
        ].rename(columns={'venta_std_historica': 'resid_std'})
        if 'family' in pipeline.categories_mapping:
            cats = pipeline.categories_mapping['family']
            std_by_group['family'] = (
                pd.Categorical(std_by_group['family'], categories=cats)
                .codes.astype('int16')
            )
        cls._std[horizon] = {
            'global': float(std_by_group['resid_std'].mean()),
            'df': std_by_group.reset_index(drop=True),
        }

    @classmethod
    def get_residual_std(cls, horizon: int) -> dict:
        """Retorna el σ residual (o fallback) del horizonte."""
        if horizon not in cls._std:
            cls.load(horizon)
        return cls._std[horizon]

    @classmethod
    def get_features(cls, horizon: int) -> list:
        """Retorna las features del modelo cargado."""
        if horizon not in cls._features:
            cls.load(horizon)
        return cls._features[horizon]

    @classmethod
    def get_pipeline(cls, horizon: int) -> object:
        """Retorna el pipeline de features del modelo cargado."""
        if horizon not in cls._pipelines:
            cls.load(horizon)
        return cls._pipelines[horizon]


    @classmethod
    def clear_cache(cls) -> None:
        """Limpia el caché — útil después de retraining."""
        cls._models = {}
        cls._features = {}
        cls._pipelines = {}
        cls._std = {}
        logger.info("Caché de modelos limpiado")


# ─────────────────────────────────────────
# 2. Preparar datos para predicción
# ─────────────────────────────────────────
def prepare_prediction_data(
    historical_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    horizon: int,
    pipeline: object
) -> pd.DataFrame:
    """
    Prepara el dataset para predecir fechas futuras.

    El modelo necesita el historial para calcular
    los lag features del período a predecir.

    Parámetros:
        historical_df: DataFrame con historial procesado
        future_dates:  fechas para las que predecir
        horizon:       7 (diario) o 30 (mensual)
        pipeline:      Instancia ajustada de DemandFeatureEngineer

    Retorna:
        DataFrame con features para predicción
    """
    logger.info(
        f"Preparando datos para predicción | "
        f"fechas: {future_dates[0]} -> {future_dates[-1]}"
    )

    # 1 Crear filas vacías para las fechas futuras
    # por cada combinación tienda-familia
    stores_families = (
        historical_df[['store_nbr', 'family']]
        .drop_duplicates()
    )

    future_df = stores_families.assign(key=1).merge(
        pd.DataFrame({'date': future_dates, 'key': 1}), on='key'
    ).drop('key', axis=1)
    # 2 El target es desconocido en el futuro
    # se inicializa en 0
    future_df[config['data']['target']] = 0.0

    data = _load_raw_predict_cached()
    data['test'] = future_df
    _, test = run_preprocessing(data, save=False, predict=True)

    # 4. CONCATENACIÓN CRÍTICA: Solo lo necesario para los lags (365 días)
    combined = pd.concat([historical_df, test], ignore_index=True)
    combined = combined.sort_values(['store_nbr', 'family', 'date'])

    # 5. Build Features usando el pipeline (estado congelado)
    combined = pipeline.transform(combined, is_train=False)

    # 6. Extraer solo futuro y liberar memoria
    start_date = future_dates.min()
    prediction_df = combined[combined['date'] >= start_date].copy()

    logger.info(
        f"Datos preparados: {prediction_df.shape}"
    )

    return prediction_df

# ─────────────────────────────────────────
# 3. Predicción principal
# ─────────────────────────────────────────
def _build_confidence_intervals(
    y_pred_log: np.ndarray,
    std_sales: np.ndarray,
    z: float = 1.96,
    upper_factor: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye los intervalos de confianza en escala real.

    Los cuantiles se calculan en escala log y se revierten con
    expm1. Ambos extremos se recortan a >= 0 y se garantiza que
    upper_bound nunca quede por debajo de lower_bound (las familias
    esporádicas con predicción ~0 producían upper_bound negativo).

    El ancho usa el σ del error residual del modelo (persistido por
    evaluate). `upper_factor` por defecto es 1.0 → intervalo simétrico
    en escala log, lo que evita que el punto quede pegado al límite
    inferior en escala real.

    Parámetros:
        y_pred_log:  predicción del modelo en escala log
        std_sales:   desviación del error residual (escala log)
        z:           multiplicador del cuantil (1.96 ≈ 95%)
        upper_factor: factor que amplía el extremo superior (1.0 = simétrico)

    Retorna:
        tuple: (lower_bound, upper_bound) redondeados a 2 decimales
    """
    lower = np.clip(np.expm1(y_pred_log - z * std_sales), 0, None)
    upper = np.clip(
        np.expm1(y_pred_log + z * std_sales * upper_factor),
        0, None
    )
    upper = np.maximum(upper, lower)
    return lower.round(2), upper.round(2)


def predict(
    historical_df: pd.DataFrame,
    horizon: int,
    n_periods: int = None
) -> pd.DataFrame:
    """
    Genera predicciones para el horizonte
    especificado.

    Parámetros:
        historical_df: DataFrame con historial procesado
        horizon:       7  → próximos 7 días
                       30 → próximos 30 días
        n_periods:     número de períodos a predecir
                       default: igual al horizonte

    Retorna:
        DataFrame con columnas:
        [date, store_nbr, family,
         predicted_sales, lower_bound, upper_bound]
    """
    n_periods = n_periods or horizon
    model     = ModelRegistry.load(horizon)

    max_history_needed = config['lags']['max_lag']

    # Generar fechas futuras
    last_date = historical_df['date'].max()
    cutoff_date = last_date - pd.Timedelta(days=max_history_needed)
    
    reduced_history = historical_df[historical_df['date'] >= cutoff_date].copy()
    logger.info(f"Historial reducido para calculo de lags de:"
                f" {len(historical_df)} a : {len(reduced_history)}")


    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_periods,
        freq='D'
    )

    logger.info(
        f"Generando predicciones | "
        f"horizon={horizon} | "
        f"períodos={n_periods} | "
        f"fechas: {future_dates[0].date()} → "
        f"{future_dates[-1].date()}"
    )

    pipeline = ModelRegistry.get_pipeline(horizon)

    # Preparar features
    prediction_df = prepare_prediction_data(
        historical_df=reduced_history,
        future_dates=future_dates,
        horizon=horizon,
        pipeline=pipeline
    )

    # Usa exactamente las features del entrenamiento
    feature_cols = ModelRegistry.get_features(horizon)

    # Verificar que todas las features existen
    missing = set(feature_cols) - set(prediction_df.columns)
    if missing:
        logger.error(f"Features faltantes: {missing}")
        raise ValueError(
            f"Faltan features en predicción: {missing}"
        )

    X_pred = prediction_df[feature_cols]

    # Predicción en escala log
    y_pred_log  = model.predict(X_pred)

    # Revertir log1p → escala real
    y_pred_real = np.expm1(y_pred_log)
    y_pred_real = np.clip(y_pred_real, 0, None)

    # Intervalo de confianza basado en el σ del error residual del modelo
    # (persistido por evaluate). Si falta, get_residual_std usa el σ
    # histórico del pipeline como fallback. Cualquier (store, family) sin
    # σ residual usa el global.
    residual_std = ModelRegistry.get_residual_std(horizon)
    std_by_group = residual_std['df'].copy()
    std_by_group = std_by_group.rename(
        columns={'resid_std': 'std_sales'}
    )
    global_std = float(residual_std.get('global', 0.0))

    results = prediction_df[
        ['date', 'store_nbr', 'family']
    ].copy()
    results['predicted_sales'] = np.round(
        y_pred_real, 2
    )

    results = pd.merge(
        results, std_by_group,
        on=['store_nbr', 'family'],
        how='left'
    )
    results['std_sales'] = results['std_sales'].fillna(global_std)

    # Cuantiles en escala log y luego revertir log1p
    lower_bound, upper_bound = _build_confidence_intervals(
        y_pred_log,
        results['std_sales'].values,
    )
    results['lower_bound'] = lower_bound
    results['upper_bound'] = upper_bound

    results = results.drop(columns=['std_sales'])

    logger.info(
        f" Predicciones generadas: "
        f"{len(results):,} filas"
    )
    return results


# ─────────────────────────────────────────
# 4. Predicción agregada por tienda
# ─────────────────────────────────────────
def predict_by_store(
    historical_df: pd.DataFrame,
    horizon: int,
    store_nbr: int
) -> pd.DataFrame:
    """
    Genera predicciones para una tienda específica.
    Útil para la API cuando se consulta por tienda.
    """
    store_df = historical_df[
        historical_df['store_nbr'] == store_nbr
    ].copy()

    if store_df.empty:
        raise ValueError(
            f"Tienda {store_nbr} no encontrada"
        )

    predictions = predict(store_df, horizon)
    return predictions


# ─────────────────────────────────────────
# 5. Guardar predicciones
# ─────────────────────────────────────────
def save_predictions(
    predictions: pd.DataFrame,
    horizon: int
) -> Path:
    """
    Guarda las predicciones en data/predictions/
    """
    output_path = Path("data/predictions")
    output_path.mkdir(parents=True, exist_ok=True)

    filename = (
        f"predictions_"
        f"{'daily' if horizon == 7 else 'monthly'}_"
        f"{pd.Timestamp.now().strftime('%Y%m%d')}"
        f".parquet"
    )
    filepath = output_path / filename

    predictions.to_parquet(filepath, index=False)
    logger.info(f"Predicciones guardadas: {filepath}")

    return filepath


# ─────────────────────────────────────────
# Ejecutar directamente desde terminal
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--horizon',
        type=int,
        choices=[7, 30],
        required=True,
        help='Horizonte: 7 (diario) o 30 (mensual)'
    )
    args = parser.parse_args()

    data_path = Path("data/processed/train_processed.parquet")


    if not data_path.exists():
        raise FileNotFoundError(
            "No se encontró train_processed.parquet\n"
            "Ejecuta primero: python src/data/preprocessing.py"
        )

    logger.info(f"Cargando historial desde: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Historial cargado: {df.shape}")

    # Generar predicciones
    predictions = predict(df, horizon=args.horizon)
    filepath    = save_predictions(
        predictions, horizon=args.horizon
    )

    print(f"\n Predicciones guardadas: {filepath}")
    print(predictions.head(10).to_string(index=False))