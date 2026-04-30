import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


# -------------------------------
# Definición de features finales
# -------------------------------
CATEGORICAL_FEATURES = [
    'family', 'city', 'state',
    'type', 'cluster', 'holiday_type',
    'holiday_impact_type'
]

NUMERICAL_FEATURES = [
    # Temporales
    'dia_semana', 'es_fin_de_semana',
    'semana_del_año', 'semana_del_mes',
    'trimestre', 'dias_para_fin_de_mes',
    'es_quincena', 'es_inicio_mes',
    # Festivos
    'es_festivo', 'es_festivo_local',
    'dias_para_siguiente_festivo',
    'dias_desde_ultimo_festivo',
    # Promoción
    'onpromotion', 'tiene_promo',
    'promo_lag_7', 'rolling_promo_mean_14',
    # Oil
    'dcoilwtico', 'oil_lag_7',
    'oil_lag_14', 'oil_lag_30',
    'oil_rolling_mean_7', 'oil_rolling_mean_30',
    # Transacciones
    'trans_lag_7',
    'trans_rolling_mean_7',
    # Tienda
    'venta_media_historica',
    'venta_std_historica',
    'ranking_tienda',
]

def _select_windows_lag(horizon:int) -> list :
    """
    Funcion para devolver la lista de lags
    de acuerdo al horizonte del modelo
    """
    lags = {
        7:  [7, 14, 21, 28, 364],
        30: [30, 60, 90, 364]
    }[horizon]

    return lags 

def _select_windows_rolling(horizon:int) -> list :
    """
    Funcion para devolver la lista de rolling
    de acuerdo al horizonte del modelo
    """
    lags = {
        7:  [7, 14, 28],
        30: [30, 60, 90]
    }[horizon]

    return lags 


# -------------------------------
# 1. Features temporales
# -------------------------------
def _build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features basadas en la fecha.
    """
    logger.info("  Construyendo features temporales...")

    df['dia_semana'] = df['date'].dt.dayofweek
    df['es_fin_de_semana'] = (
        df['dia_semana'].isin([5, 6])
    ).astype('int8')
    df['semana_del_año'] = (
        df['date'].dt.isocalendar().week.astype('int32')
    )
    df['semana_del_mes'] = (
        (df['date'].dt.day - 1) // 7 + 1
    ).astype('int8')
    df['trimestre'] = df['date'].dt.quarter.astype('int8')
    df['dias_para_fin_de_mes'] = (
        df['date'].dt.days_in_month - df['date'].dt.day
    ).astype('int8')
    df['es_quincena'] = (
        df['date'].dt.day.isin([14, 15, 28, 29, 30, 31])
    ).astype('int8')
    df['es_inicio_mes'] = (
        df['date'].dt.day <= 3
    ).astype('int8')
    df['day_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    
    df['pico_quincena_findex'] = (df['es_fin_de_semana'] & df['es_quincena']).astype('int8')
    df['es_viernes'] = (df['dia_semana'] == 4).astype('int8')

    logger.info(" Features temporales completadas")
    return df


# -------------------------------
# 2. Features de festivos
# -------------------------------
def _build_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features de festivos usando el
    holiday_impact_type definido en el EDA.
    """
    logger.info("  Construyendo features de festivos...")

    # Flags básicos
    df['es_festivo'] = (
        df['holiday_type'] != 'No_Holiday'
    ).astype('int8')


    # Holiday impact type desde config
    impact = config['features']['holiday_impact']

    def classify_holiday(description: str) -> str:
        if description == 'No_Holiday':
            return 'none'
        for keyword in impact['atypical']:
            if keyword.lower() in description.lower():
                return 'atypical'
        for keyword in impact['positive']:
            if keyword.lower() in description.lower():
                return 'positive'
        for keyword in impact['negative']:
            if keyword.lower() in description.lower():
                return 'negative'
        return 'neutral'

    df['holiday_impact_type'] = (
        df['holiday_description']
        .apply(classify_holiday)
        .astype('category')
    )

    # Distancia al festivo más cercano
    logger.info("  Calculando distancia a festivos...")
    festivos_dates = pd.to_datetime(
        df[df['es_festivo'] == 1]['date'].unique()
    )

    daily_dates = df['date'].drop_duplicates().sort_values()

    dias_para  = {}
    dias_desde = {}

    for date in daily_dates:
        future = [(f - date).days
                  for f in festivos_dates if f >= date]
        past   = [(date - f).days
                  for f in festivos_dates if f <= date]
        dias_para[date]  = min(future) if future else 999
        dias_desde[date] = min(past)   if past   else 999

    df['dias_para_siguiente_festivo'] = (
        df['date'].map(dias_para).astype('int16')
    )
    df['dias_desde_ultimo_festivo'] = (
        df['date'].map(dias_desde).astype('int16')
    )

    logger.info(" Features de festivos completadas")
    return df


# -------------------------------
# 3. Lag features
# -------------------------------
def _build_lag_features(df: pd.DataFrame,
                        horizon: int) -> pd.DataFrame:
    """
    Construye lag features respetando el horizonte
    de predicción para evitar data leakage.

    Parámetros:
        horizon: días de anticipación de la predicción
                 7  → modelo diario
                 30 → modelo mensual
    """
    logger.info(f"  Construyendo lag features (horizon={horizon})...")
    group = ['store_nbr', 'family']
    target = config['data']['target']

    lags = _select_windows_lag(horizon)

    for lag in lags:
        col_name = f'lag_{lag}'
        df[col_name] = (
            df.sort_values('date')
            .groupby(group,observed=True)[target]
            .shift(lag)
            .astype('float32')
        )
        null_pct = df[col_name].isnull().mean() * 100
        logger.info(f"    lag_{lag}: {null_pct:.1f}% nulos")

    logger.info("   Lag features completadas")
    return df


# -------------------------------
# 4. Rolling features
# -------------------------------
def _build_rolling_features(df: pd.DataFrame,
                             horizon: int) -> pd.DataFrame:
    """
    Construye rolling statistics respetando
    el horizonte para evitar data leakage.
    """
    logger.info("  Construyendo rolling features...")
    group  = ['store_nbr', 'family']
    target = config['data']['target']

    windows = _select_windows_rolling(horizon)

    df = df.sort_values(['store_nbr', 'family', 'date'])

    for w in windows:
        shifted = (
            df.groupby(group,observed=True)[target]
            .shift(horizon)
        )

        df[f'rolling_mean_{w}d'] = (
            shifted.groupby(
                [df['store_nbr'], df['family']],
                observed=True              # ← fix FutureWarning
            ).transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            ).astype('float32')
        )

        df[f'rolling_std_{w}d'] = (
            shifted.groupby(
                [df['store_nbr'], df['family']],
                observed=True              # ← fix FutureWarning
            ).transform(
                lambda x: x.rolling(w, min_periods=1).std()
            ).astype('float32')
        )

        df[f'rolling_max_{w}d'] = (
            shifted.groupby(
                [df['store_nbr'], df['family']],
                observed=True              # ← fix FutureWarning
            ).transform(
                lambda x: x.rolling(w, min_periods=1).max()
            ).astype('float32')
        )

    # Coeficiente de variación
    # Usar la ventana más grande disponible
    # según el horizonte para evitar KeyError
    std_col  = f'rolling_std_{windows[-1]}d'
    mean_col = f'rolling_mean_{windows[-1]}d'

    df['cv_ventas'] = (
        df[std_col] /
        (df[mean_col] + 1)
    ).astype('float32')

    logger.info("  Rolling features completadas")
    return df


# -------------------------------
# 5. Features de Oil
# -------------------------------
def _build_oil_features(df: pd.DataFrame, horizon) -> pd.DataFrame:
    """
    Construye features del precio del petróleo.
    """
    logger.info("  Construyendo features de oil...")

    lags = _select_windows_lag(horizon)
    rolls = _select_windows_rolling(horizon)

    df = df.sort_values('date')

    for lag in lags:
        df[f'oil_lag_{lag}'] = (
            df.groupby('store_nbr',observed=True)['dcoilwtico']
            .shift(lag)
            .astype('float32')
        )

    for w in rolls:
        
        df[f'oil_rolling_mean_{w}'] = (
            df.groupby('store_nbr',observed=True)['dcoilwtico']
            .transform(
                lambda x: x.shift(horizon).rolling(w, min_periods=1).mean()
            ).astype('float32')
        )


    logger.info("   Features de oil completadas")
    return df


# -------------------------------
# 6. Features de promoción
# -------------------------------
def _build_promo_features(df: pd.DataFrame, horizon:int) -> pd.DataFrame:
    group = ['store_nbr', 'family']
    df = df.sort_values(['store_nbr', 'family', 'date'])

    df['tiene_promo'] = (
        df['onpromotion'] > 0
    ).astype('int8')


    df[f'promo_lag_{horizon}'] = (
        df.groupby(group, observed=True)['onpromotion']
        .shift(horizon)
        .astype('float32')
    )

    df[f'rolling_promo_mean_{horizon*2}'] = (
        df.groupby(group, observed=True)['onpromotion']  # ← fix
        .transform(
            lambda x: x.shift(horizon).rolling(horizon*2, min_periods=1).mean()
        ).astype('float32')
    )

    logger.info("   Features de promoción completadas")
    return df


# -------------------------------
# 7. Features de transacciones
# -------------------------------
def _build_transaction_features(df: pd.DataFrame, horizon:int) -> pd.DataFrame:
    df = df.sort_values(['store_nbr', 'date'])

    df[f'trans_lag_{horizon}'] = (
        df.groupby('store_nbr',observed=True)['transactions']
        .shift(horizon)
        .astype('float32')
    )

    df[f'trans_rolling_mean_{horizon}'] = (
        df.groupby('store_nbr', observed=True)['transactions']  # ← fix
        .transform(
            lambda x: x.shift(horizon).rolling(horizon, min_periods=1).mean()
        ).astype('float32')
    )

    logger.info("   Features de transacciones completadas")
    return df


# -------------------------------
# 8. Features de tienda
# -------------------------------
def _build_store_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Construye features históricas de tienda.
    """
    logger.info("  Construyendo features de tienda...")

    # Si ya fueron inyectadas desde el modelo guardado
    # no recalcular — usar las del historial completo
    if 'venta_media_historica' not in df.columns:
        store_stats = (
            df.groupby(
                ['store_nbr', 'family'],
                observed=True
            )[config['data']['target']]
            .agg(['mean', 'std'])
            .rename(columns={
                'mean': 'venta_media_historica',
                'std':  'venta_std_historica'
            })
            .reset_index()
        )
        df = df.merge(
            store_stats,
            on=['store_nbr', 'family'],
            how='left'
        )
    else:
        logger.info(
            "  Store stats ya disponibles — "
            "usando historial completo"
        )

    ranking = (
        df.groupby('store_nbr', observed=True)
        [config['data']['target']]
        .sum()
        .rank(ascending=False)
        .astype('int32')
    )
    df['ranking_tienda'] = (
        df['store_nbr'].map(ranking).astype('int32')
    )

    logger.info("   Features de tienda completadas")
    return df


# ─────────────────────────────────────────
# 9. Encoding de categóricas
# ─────────────────────────────────────────
def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica Label Encoding a las variables categóricas.
    LightGBM las maneja nativamente pero necesita
    que sean numéricas o tipo category.
    """
    logger.info("  Encodificando variables categóricas...")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if df[col].dtype.name != 'category':
                df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes.astype('int16')    
        # Garantizar tipos correctos para LightGBM
        # antes de pasar al modelo
        if 'transferred' in df.columns:
            df['transferred'] = df['transferred'].astype(bool)

    logger.info("  Encoding completado")
    return df


# ─────────────────────────────────────────
# Función principal — punto de entrada
# ─────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   horizon: int,
                   save: bool = True) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de feature engineering.

    Parámetros:
        df:      DataFrame procesado de preprocessing.py
        horizon: horizonte de predicción
                 7  → modelo diario
                 30 → modelo mensual
        save:    si True, guarda el resultado en .parquet

    Retorna:
        DataFrame con todas las features construidas
    """
    logger.info("=" * 50)
    logger.info(
        f"Iniciando feature engineering "
        f"(horizon={horizon} días)"
    )
    logger.info("=" * 50)
    df = df.copy()

    df = _build_temporal_features(df)
    df = _build_holiday_features(df)
    df = _build_lag_features(df, horizon)
    df = _build_rolling_features(df, horizon)
    df = _build_oil_features(df, horizon)
    df = _build_promo_features(df, horizon)
    df = _build_transaction_features(df, horizon)
    df = _build_store_features(df, horizon)
    df = _encode_categoricals(df)

    print("="*75)
    print(f"Features finales:{df.columns.to_list()}")
    print("="*75)

    # Eliminar filas con nulos en lags
    # (primeras semanas sin historial suficiente)
    initial_rows = len(df)
    lag_cols = [c for c in df.columns if 'lag_' in c]
    df = df.dropna(subset=lag_cols)
    dropped = initial_rows - len(df)
    logger.info(
        f"Filas eliminadas por nulos en lags: {dropped:,}"
    )

    if save:
        output_path = Path("data/processed")
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"train_features_d{horizon}.parquet"
        filepath = output_path / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Guardado: {filepath}")

    logger.info("=" * 50)
    logger.info(" Feature engineering completado")
    logger.info(f"  Shape final: {df.shape}")
    logger.info(f"  Features:    {len(df.columns)} columnas")
    logger.info("=" * 50)

    return df