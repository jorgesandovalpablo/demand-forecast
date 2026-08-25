import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

# -------------------------------
# Definición de features
# -------------------------------
CATEGORICAL_FEATURES = [
    'family', 'city', 'state',
    'type', 'cluster', 'holiday_type',
    'holiday_impact_type'
]

class DemandFeatureEngineer:
    """
    Pipeline unificado de feature engineering.
    Garantiza paridad exacta entre entrenamiento e inferencia.
    """
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.categories_mapping = {}
        self.store_stats = None
        self.store_ranking = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Aprende el estado global necesario para el serving.
        Solo se ejecuta en el pipeline de entrenamiento.
        """
        logger.info(f"Ajustando DemandFeatureEngineer (horizon={self.horizon})...")
        
        # 1. Aprender categorías exactas
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                if df[col].dtype.name != 'category':
                    df[col] = df[col].astype('category')
                self.categories_mapping[col] = df[col].cat.categories

        # 2. Aprender estadísticas históricas base
        target = config['data']['target']
        self.store_stats = (
            df.groupby(['store_nbr', 'family'], observed=True)[target]
            .agg(['mean', 'std'])
            .rename(columns={
                'mean': 'venta_media_historica',
                'std':  'venta_std_historica'
            })
            .reset_index()
        )

        # 3. Aprender ranking de tiendas global
        ranking = (
            df.groupby('store_nbr', observed=True)[target]
            .sum()
            .rank(ascending=False)
            .astype('int32')
        )
        self.store_ranking = ranking.to_dict()

        self.is_fitted = True
        logger.info("DemandFeatureEngineer ajustado correctamente.")
        return self

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Aplica las transformaciones asegurando paridad con el fit.
        """
        if not self.is_fitted:
            raise ValueError("El pipeline no ha sido ajustado (llamar a fit() primero).")

        logger.info("=" * 50)
        logger.info(f"Transformando datos (horizon={self.horizon}, is_train={is_train})")
        logger.info("=" * 50)
        
        df = df.copy()

        # Generación de features stateless
        df = self._build_temporal_features(df)
        df = self._build_holiday_features(df)
        df = self._build_lag_features(df)
        df = self._build_rolling_features(df)
        df = self._build_oil_features(df)
        df = self._build_promo_features(df)
        df = self._build_transaction_features(df)

        # Inyectar features stateful (aprendidas en fit)
        logger.info("  Inyectando store stats y ranking aprendidos...")
        df = df.merge(self.store_stats, on=['store_nbr', 'family'], how='left')
        df['ranking_tienda'] = df['store_nbr'].map(self.store_ranking).astype('int32')

        # Encoding categórico estricto
        logger.info("  Aplicando encoding categórico estricto...")
        for col, categories in self.categories_mapping.items():
            if col in df.columns:
                # Se fuerza a usar exactamente las categorías aprendidas
                df[col] = pd.Categorical(df[col], categories=categories).codes.astype('int16')
        
        if 'transferred' in df.columns:
            df['transferred'] = df['transferred'].astype(bool)

        # Eliminar data leakage
        if 'dcoilwtico' in df.columns:
            df = df.drop(columns=['dcoilwtico'])
            logger.info("  Eliminado 'dcoilwtico' para evitar data leakage.")

        # Limpieza de nulos iniciales solo en entrenamiento
        if is_train:
            initial_rows = len(df)
            lag_cols = [c for c in df.columns if 'lag_' in c]
            df = df.dropna(subset=lag_cols)
            dropped = initial_rows - len(df)
            logger.info(f"  Filas eliminadas por nulos en lags: {dropped:,}")

        logger.info("=" * 50)
        logger.info(" Transformación completada")
        logger.info(f"  Shape final: {df.shape}")
        logger.info("=" * 50)
        
        return df

    # -------------------------------
    # Métodos privados de construcción
    # -------------------------------
    def _select_windows_lag(self) -> list:
        return {
            7:  [7, 14, 21, 28, 364],
            30: [30, 60, 90, 364]
        }[self.horizon]

    def _select_windows_rolling(self) -> list:
        return {
            7:  [7, 14, 28],
            30: [30, 60, 90]
        }[self.horizon]

    def _build_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Construyendo features temporales...")
        df['dia_semana'] = df['date'].dt.dayofweek
        df['es_fin_de_semana'] = df['dia_semana'].isin([5, 6]).astype('int8')
        df['semana_del_año'] = df['date'].dt.isocalendar().week.astype('int32')
        df['semana_del_mes'] = ((df['date'].dt.day - 1) // 7 + 1).astype('int8')
        df['trimestre'] = df['date'].dt.quarter.astype('int8')
        df['dias_para_fin_de_mes'] = (df['date'].dt.days_in_month - df['date'].dt.day).astype('int8')
        df['es_quincena'] = df['date'].dt.day.isin([14, 15, 28, 29, 30, 31]).astype('int8')
        df['es_inicio_mes'] = (df['date'].dt.day <= 3).astype('int8')
        df['day_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        df['pico_quincena_findex'] = (df['es_fin_de_semana'] & df['es_quincena']).astype('int8')
        df['es_viernes'] = (df['dia_semana'] == 4).astype('int8')
        return df

    def _build_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Construyendo features de festivos...")
        df['es_festivo'] = (df['holiday_type'] != 'No_Holiday').astype('int8')
        impact = config['features']['holiday_impact']

        def classify_holiday(description: str) -> str:
            if description == 'No_Holiday': return 'none'
            for keyword in impact['atypical']:
                if keyword.lower() in description.lower(): return 'atypical'
            for keyword in impact['positive']:
                if keyword.lower() in description.lower(): return 'positive'
            for keyword in impact['negative']:
                if keyword.lower() in description.lower(): return 'negative'
            return 'neutral'

        df['holiday_impact_type'] = df['holiday_description'].apply(classify_holiday).astype('category')

        festivos_dates = pd.to_datetime(df[df['es_festivo'] == 1]['date'].unique())
        daily_dates = df['date'].drop_duplicates().sort_values()

        dias_para, dias_desde = {}, {}
        for date in daily_dates:
            future = [(f - date).days for f in festivos_dates if f >= date]
            past   = [(date - f).days for f in festivos_dates if f <= date]
            dias_para[date]  = min(future) if future else 999
            dias_desde[date] = min(past)   if past   else 999

        df['dias_para_siguiente_festivo'] = df['date'].map(dias_para).astype('int16')
        df['dias_desde_ultimo_festivo'] = df['date'].map(dias_desde).astype('int16')
        return df

    def _build_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Construyendo lag features...")
        group = ['store_nbr', 'family']
        target = config['data']['target']
        lags = self._select_windows_lag()

        for lag in lags:
            col_name = f'lag_{lag}'
            df[col_name] = (
                df.sort_values('date')
                .groupby(group, observed=True)[target]
                .shift(lag)
                .astype('float32')
            )
        return df

    def _build_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Construyendo rolling features...")
        group  = ['store_nbr', 'family']
        target = config['data']['target']
        windows = self._select_windows_rolling()

        df = df.sort_values(['store_nbr', 'family', 'date'])
        shifted_target = df.groupby(group, observed=True)[target].shift(self.horizon)

        for w in windows:
            rolling_obj = (shifted_target.groupby([df['store_nbr'], df['family']], observed=True)
                           .rolling(window=w, min_periods=1))
            
            df[f'rolling_mean_{w}d'] = rolling_obj.mean().reset_index(level=[0,1], drop=True).astype('float32')
            df[f'rolling_std_{w}d']  = rolling_obj.std().reset_index(level=[0,1], drop=True).astype('float32')
            df[f'rolling_max_{w}d']  = rolling_obj.max().reset_index(level=[0,1], drop=True).astype('float32')

        std_col  = f'rolling_std_{windows[-1]}d'
        mean_col = f'rolling_mean_{windows[-1]}d'
        df['cv_ventas'] = (df[std_col] / (df[mean_col] + 1e-6)).astype('float32')
        return df

    def _build_oil_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Construyendo features de oil...")
        lags = self._select_windows_lag()
        rolls = self._select_windows_rolling()
        df = df.sort_values('date')

        for lag in lags:
            df[f'oil_lag_{lag}'] = (
                df.groupby('store_nbr', observed=True)['dcoilwtico']
                .shift(lag)
                .astype('float32')
            )

        for w in rolls:
            df[f'oil_rolling_mean_{w}'] = (
                df.groupby('store_nbr', observed=True)['dcoilwtico']
                .transform(lambda x: x.shift(self.horizon).rolling(w, min_periods=1).mean())
                .astype('float32')
            )
        return df

    def _build_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Construyendo features de promoción...")
        group = ['store_nbr', 'family']
        df = df.sort_values(['store_nbr', 'family', 'date'])
        
        df['tiene_promo'] = (df['onpromotion'] > 0).astype('int8')
        df[f'promo_lag_{self.horizon}'] = (
            df.groupby(group, observed=True)['onpromotion']
            .shift(self.horizon)
            .astype('float32')
        )
        df[f'rolling_promo_mean_{self.horizon*2}'] = (
            df.groupby(group, observed=True)['onpromotion']
            .transform(lambda x: x.shift(self.horizon).rolling(self.horizon*2, min_periods=1).mean())
            .astype('float32')
        )
        return df

    def _build_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Construyendo features de transacciones...")
        df = df.sort_values(['store_nbr', 'date'])
        
        df[f'trans_lag_{self.horizon}'] = (
            df.groupby('store_nbr', observed=True)['transactions']
            .shift(self.horizon)
            .astype('float32')
        )
        df[f'trans_rolling_mean_{self.horizon}'] = (
            df.groupby('store_nbr', observed=True)['transactions']
            .transform(lambda x: x.shift(self.horizon).rolling(self.horizon, min_periods=1).mean())
            .astype('float32')
        )
        return df

    def save(self, filepath: Path):
        joblib.dump(self, filepath)
        logger.info(f"DemandFeatureEngineer guardado en {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'DemandFeatureEngineer':
        return joblib.load(filepath)

# Función puente por compatibilidad con scripts antiguos que todavía la llamen
def build_features(df: pd.DataFrame, horizon: int, save: bool = True) -> pd.DataFrame:
    logger.warning("Usando función bridge build_features. Prefiere usar DemandFeatureEngineer directamente.")
    fe = DemandFeatureEngineer(horizon=horizon)
    fe.fit(df)
    df_transformed = fe.transform(df, is_train=True)
    
    if save:
        output_path = Path("data/processed")
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / f"train_features_d{horizon}.parquet"
        df_transformed.to_parquet(filepath, index=False)
        logger.info(f"Guardado: {filepath}")
        
    return df_transformed