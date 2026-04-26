import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import config
from src.features.build_features import build_features
from src.models.train import get_feature_cols

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
            if not model_path.exists():
                logger.error(
                    f"Modelo no encontrado: {model_path}"
                )
                raise FileNotFoundError(
                    f"Modelo no entrenado para "
                    f"horizonte {horizon}. "
                    f"Ejecuta primero train.py"
                )

            logger.info(
                f"Cargando modelo horizon={horizon} "
                f"desde {model_path}..."
            )
            cls._models[horizon] = joblib.load(model_path)
            logger.info(
                f"Modelo horizon={horizon} cargado"
            )

        return cls._models[horizon]

    @classmethod
    def clear_cache(cls) -> None:
        """Limpia el caché — útil después de retraining."""
        cls._models = {}
        logger.info("Caché de modelos limpiado")


# ─────────────────────────────────────────
# 2. Preparar datos para predicción
# ─────────────────────────────────────────
def prepare_prediction_data(
    historical_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    horizon: int
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

    # Crear filas vacías para las fechas futuras
    # por cada combinación tienda-familia
    stores_families = (
        historical_df[['store_nbr', 'family']]
        .drop_duplicates()
    )

    future_rows = []
    for date in future_dates:
        temp = stores_families.copy()
        temp['date'] = date
        future_rows.append(temp)

    future_df = pd.concat(future_rows, ignore_index=True)

    # Copiar columnas estáticas de tienda
    store_cols = ['store_nbr', 'city', 'state',
                  'type', 'cluster']
    store_info = (
        historical_df[store_cols]
        .drop_duplicates('store_nbr')
    )
    future_df = future_df.merge(
        store_info, on='store_nbr', how='left'
    )

    # El target es desconocido en el futuro
    # lo inicializamos en 0
    future_df[config['data']['target']] = 0.0

    # Concatenar historial + futuro para calcular lags
    combined = pd.concat(
        [historical_df, future_df],
        ignore_index=True
    ).sort_values(['store_nbr', 'family', 'date'])

    # Construir features sobre el combined
    combined = build_features(
        combined,
        horizon=horizon,
        save=False
    )

    # Retornar solo las filas futuras
    prediction_df = combined[
        combined['date'].isin(future_dates)
    ].copy()

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

    # Generar fechas futuras
    last_date    = historical_df['date'].max()
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

    # Preparar features
    prediction_df = prepare_prediction_data(
        historical_df=historical_df,
        future_dates=future_dates,
        horizon=horizon
    )

    # Obtener columnas de features
    feature_cols = get_feature_cols(prediction_df)
    X_pred       = prediction_df[feature_cols]

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

    results = prediction_df[
        ['date', 'store_nbr', 'family']
    ].copy()
    results['predicted_sales'] = np.round(
        y_pred_real, 2
    )

    results = results.merge(
        std_by_group,
        on=['store_nbr', 'family'],
        how='left'
    )

    results['lower_bound'] = np.clip(
        results['predicted_sales'] -
        1.96 * results['std_sales'],
        0, None
    ).round(2)
    results['upper_bound'] = (
        results['predicted_sales'] +
        1.96 * results['std_sales']
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

    # Cargar datos procesados
    df = pd.read_parquet(
        f"data/processed/"
        f"train_features_d{args.horizon}.parquet"
    )

    # Generar predicciones
    predictions = predict(df, horizon=args.horizon)
    filepath    = save_predictions(
        predictions, horizon=args.horizon
    )

    print(f"\n Predicciones guardadas: {filepath}")
    print(predictions.head(10).to_string(index=False))