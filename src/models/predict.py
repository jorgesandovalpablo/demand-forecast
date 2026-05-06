import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import config
from src.features.build_features import build_features, _encode_categoricals
from src.data.ingestion import load_raw_data
from src.data.preprocessing import run_preprocessing

logger = get_logger(__name__)


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
    _store_stats: dict = {}

    @classmethod
    def load(cls, horizon: int) -> object:
        """
        Carga el modelo si no está en caché.
        Si ya está cargado, lo retorna directo.
        """
        if horizon not in cls._models:
            model_path = Path(
                f"models/lgbm_h{horizon}.pkl"
            )
            features_path = Path(
                f"models/features_h{horizon}.pkl"
            )
            stats_path = Path(
                f"models/store_stats_h{horizon}.pkl"
            )
            

            if not model_path.exists():
                logger.error(
                    f"Modelo no encontrado: {model_path}"
                )
                raise FileNotFoundError(
                    f"Modelo no entrenado para "
                    f"horizonte {horizon}. "
                    f"Ejecuta primero train.py"
                )

            if not features_path.exists():
                raise FileNotFoundError(
                    f"Features no encontradas: {features_path}\n"
                )

            if not stats_path.exists():
                raise FileNotFoundError(
                    f"Store stats no encontradas: {stats_path}\n"
                )

            logger.info(
                f"Cargando modelo horizon={horizon} "
                f"desde {model_path}..."
            )
            cls._models[horizon] = joblib.load(model_path)
            cls._features[horizon] = joblib.load(features_path)
            cls._store_stats[horizon] = joblib.load(stats_path)
            logger.info(
                f"Modelo horizon={horizon} cargado |"
                f"Features: {len(cls._features[horizon])}"
            )

        return cls._models[horizon]

    @classmethod
    def get_features(cls, horizon: int) -> list:
        """Retorna las features del modelo cargado."""
        if horizon not in cls._features:
            cls.load(horizon)
        return cls._features[horizon]

    @classmethod
    def get_store_stats(cls, horizon: int) -> list:
        """Retorna las store stats del modelo cargado."""
        if horizon not in cls._store_stats:
            cls.load(horizon)
        return cls._store_stats[horizon]


    @classmethod
    def clear_cache(cls) -> None:
        """Limpia el caché — útil después de retraining."""
        cls._models = {}
        cls._features = {}
        cls._store_stats = {}
        logger.info("Caché de modelos limpiado")


# ─────────────────────────────────────────
# 2. Preparar datos para predicción
# ─────────────────────────────────────────
def prepare_prediction_data(
    historical_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    horizon: int,
    store_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepara el dataset para predecir fechas futuras.

    El modelo necesita el historial para calcular
    los lag features del período a predecir.

    Parámetros:
        historical_df: DataFrame con historial procesado
        future_dates:  fechas para las que predecir
        horizon:       7 (diario) o 30 (mensual)

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

    data = load_raw_data(predict=True)
    data['test'] = future_df
    _, test = run_preprocessing(data, save=False, predict=True)

    # 4. CONCATENACIÓN CRÍTICA: Solo lo necesario para los lags (365 días)
    combined = pd.concat([historical_df, test], ignore_index=True)
    combined = combined.sort_values(['store_nbr', 'family', 'date'])

    # 5. Build Features (Asegúrate que devuelva float32)
    combined = build_features(combined, horizon=horizon, save=False)

    print("combined columns:", combined.columns.tolist())
    print("store_stats columns:", store_stats.columns.tolist())

    combined.to_parquet("data/predictions/predict_df.parquet")

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

    store_stats = ModelRegistry.get_store_stats(horizon)

    # Preparar features
    prediction_df = prepare_prediction_data(
        historical_df=reduced_history,
        future_dates=future_dates,
        horizon=horizon,
        store_stats=store_stats
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

    # Intervalo de confianza simple
    # basado en la desviación histórica
    # por tienda-familia
    std_by_group = (
        historical_df.groupby(
            ['store_nbr', 'family']
        )[config['data']['target']]
        .std()
        .apply(np.expm1)
        .reset_index()
        .rename(columns={
            config['data']['target']: 'std_sales'
        })
    )
    print('STD group:')
    print(std_by_group.head())
    print(f'Tipos:{std_by_group.info()}')

    print('Results:')
    results = prediction_df[
        ['date', 'store_nbr', 'family']
    ].copy()
    results['predicted_sales'] = np.round(
        y_pred_real, 2
    )

    print(results.head())

    std_by_group = _encode_categoricals(std_by_group)

    results = pd.merge(
        results, std_by_group, 
        on=['store_nbr', 'family'],
        how='left'
    )

    print(results['std_sales'])

    results['lower_bound'] = np.clip(
        results['predicted_sales'] -
        1.96 * results['std_sales'],
        0, None
    ).round(2)
    results['upper_bound'] = (
        results['predicted_sales'] +
        1.96 * results['std_sales'] * 1.5
    ).round(2)

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