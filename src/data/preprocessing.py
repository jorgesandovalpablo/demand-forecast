import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

# --------------------------------
# Merge Holidays
# --------------------------------

def _process_holidays(holidays:pd.DataFrame) -> tuple:
    """
    Separa holidays por locale para el merge
    evitando duplicados.

    Retorna:
        tuple: (National_df, regional_df, local_df)
    """
    logger.info("Processando holidays...")

    national = (
        holidays[holidays['locale']== 'National']
        .sort_values('transferred')
        .drop_duplicates(subset=['date'])
        [['date','type','locale',
          'description','transferred']]
        .rename(columns={
            'type':         'holiday_type',
            'description':  'holiday_descrition'
        })
    )

    regional = (
        holidays[holidays['locale']== ' Regional']
        .sort_values('transferred')
        .drop_duplicates(subset=['date','locale_name'])
        [['date','locale_name','type',
          'description','transferred']]
        .rename(columns={
            'type':         'holiday_type',
            'locale_name':   'state',
            'description':  'holiday_descrition'
        })
    )

    local = (
        holidays[holidays['locale'] == 'Local']
        .sort_values('transferred')
        .drop_duplicates(subset=['date', 'locale_name'])
        [['date', 'locale_name', 'type',
          'description', 'transferred']]
        .rename(columns={
            'type':        'holiday_type',
            'locale_name': 'city',
            'description': 'holiday_description'
        })
    )

    logger.info(
        f"Ncional: {len(national)} registros |"
        f"Regional {len(regional)} |"
        f"Local: {len(local)}"
    )

    return national, regional, local 

# -----------------------
# Merge principal
# -----------------------

def _merge_datasets(df: pd.DataFrame,
                    stores: pd.DataFrame,
                    oil:pd.DataFrame,
                    transactions:pd.DataFrame,
                    holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el merge completo de los dataframes
    """

    logger.info("Ejecutandomerge de datasets")
    initial_shape = df.shape

    # 1. Stores
    df = df.merge(stores, on='store_nbr', how='left')
    logger.info(f"  Post merge stores: {df.shape}")

    # 2. Oil
    df = df.merge(oil, on='date', how='left')
    logger.info(f"  Post merge oil: {df.shape}")

    # 3. Transactions
    df = df.merge(
        transactions,
        on=['date', 'store_nbr'],
        how='left'
    )
    logger.info(f"  Post merge transactions: {df.shape}")

    # 4. Holidays — merge diferenciado por locale
    national, regional, local = _process_holidays(holidays)

    df = df.merge(national, on='date', how='left')
    df = df.merge(
        regional,
        on=['date','state'],
        how='left',
        suffixes=('','_regional')
    )

    df = df.merge(
        local,
        on=['date', 'city'],
        how='left',
        suffixes=('', '_local')
    )

    for col in ['dolifay_type','holiday_dexcription',
                      'transfered']:
        local_col = f"{col}_local" 
        regional_col = f"{col}_regional"

        if local_col in df.columns:
            df[col] = (
                df[local_col]
                .fillna(df.get(regional_col, np.nan))
            )
    
    cols_to_drop = [
        c for c in df.columns
        if '_regional' in c or '_local' in c 
    ]
    df = df.drop(columns=cols_to_drop)

    dupes = df.duplicated(
        subset=['date', 'store_nbr', 'family']
    ).sum()

    if dupes > 0:
        logger.error(f"Duplicados detectados: {dupes:,}")
        raise ValueError(
            f"El merge generó {dupes} duplicados inesperados"
        )

    logger.info(
        f"  Merge completo: {initial_shape} : {df.shape}"
    )

    return df

# --------------------
# Tratamiento de nulos
# --------------------

def _handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata los nulos de cada columna
    según la estrategia definida en el EDA.
    """
    logger.info("Tratando nulos...")

    # A Oil se aplica interpolación lineal + forward fill
    # Los mercados no cotizan fines de semana
    # interpolamos para tener un valor continuo
    null_oil_before = df['dcoilwtico'].isnull().sum()
    df['dcoilwtico'] = (
        df.sort_values('date')
        .groupby('store_nbr')['dcoilwtico']
        .transform(lambda x:
            x.interpolate(method='linear')
             .ffill()
             .bfill()
        )
    )
    null_oil_after = df['dcoilwtico'].isnull().sum()
    logger.info(
        f"  Oil: {null_oil_before:,} nulos : "
        f"{null_oil_after:,} nulos"
    )


    # Transactions asegurar ceros
    # Si no hay registro, no hubo transacciones
    null_trans = df['transactions'].isnull().sum()
    df['transactions'] = df['transactions'].fillna(0)
    logger.info(
        f"  Transactions: {null_trans:,} nulos → imputados con 0"
    )

    # Holidays: imputar con 'No_Holiday'
    # Días sin festivo son la mayoría — es esperado
    for col in ['holiday_type', 'holiday_description']:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            df[col] = df[col].fillna('No_Holiday')
            logger.info(
                f"  {col}: {null_count:,} nulos → "
                f"imputados con 'No_Holiday'"
            )

    # Transferred: imputar con False
    if 'transferred' in df.columns:
        df['transferred'] = df['transferred'].fillna(False)

    logger.info("Tratamiento de nulos completado")
    return df

# ------------------------
# Tranformacion del target
# ------------------------

def _transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica log1p al target para reducir el sesgo
    de la distribución.

    Guarda el valor original en 'sales_raw'
    para referencia y auditoría.
    """
    target = config['data']['target']
    logger.info(f"Aplicando log1p al target '{target}'...")

    df[f'{target}_raw'] = df[target]
    df[target] = np.log1p(df[target])
    logger.info(
        f"{target}_raw range: "
        f"[{df[f'{target}_raw'].min():.2f}, "
        f"{df[f'{target}_raw'].max():.2f}]"
    )
    logger.info(
        f"{target} (log1p) range: "
        f"[{df[target].min():.2f}, "
        f"{df[target].max():.2f}]"
    )
    return df

# -----------------------
# Optimizacion de memoria
# -----------------------

def _reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce el uso de memoria optimizando
    los tipos de datos de cada columna.

    Se espera reducion 60-70% del uso de RAM.
    """
    logger.info("Optimizando uso de memoria...")
    mem_before = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')

    for col in df.select_dtypes('int64').columns:
        df[col] = df[col].astype('int32')

    for col in df.select_dtypes('object').columns:
        df[col] = df[col].astype('category')

    mem_after = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (1 - mem_after / mem_before) * 100

    logger.info(
        f"  Memoria: {mem_before:.1f} MB → "
        f"{mem_after:.1f} MB "
        f"({reduction:.1f}% reducción)"
    )
    return df

# ------------------
# Funcion Principal
# -----------------

def run_preprocessing(data: dict,
                      save: bool = True) -> tuple:
    """
    Ejecuta el pipeline completo de preprocessing.

    Parámetros:
        data: dict de DataFrames de ingestion.py
        save: si True, guarda los resultados en .parquet

    Retorna:
        tuple: (train_processed, test_processed)
    """
    logger.info("=" * 50)
    logger.info("Iniciando preprocessing pipeline")
    logger.info("=" * 50)

    logger.info("Procesando TRAIN...")
    train = _merge_datasets(
        df=data['train'],
        stores=data['stores'],
        oil=data['oil'],
        transactions=data['transactions'],
        holidays=data['holidays']
    )
    train = _handle_nulls(train)
    train = _transform_target(train)
    train = _reduce_memory(train)

    logger.info("Procesando TEST...")
    test = _merge_datasets(
        df=data['test'],
        stores=data['stores'],
        oil=data['oil'],
        transactions=data['transactions'],
        holidays=data['holidays']
    )
    test = _handle_nulls(test)
    test = _reduce_memory(test)

    if save:
        _save_processed(train, test)

    logger.info("=" * 50)
    logger.info(" Preprocessing completado")
    logger.info(f"  train: {train.shape}")
    logger.info(f"  test:  {test.shape}")
    logger.info("=" * 50)

    return train, test

def _save_processed(train: pd.DataFrame,
                    test: pd.DataFrame) -> None:
    """Guarda los datasets procesados en formato parquet."""
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train_processed.parquet"
    test_path  = output_path / "test_processed.parquet"

    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    logger.info(f"  Guardado: {train_path}")
    logger.info(f"  Guardado: {test_path}")