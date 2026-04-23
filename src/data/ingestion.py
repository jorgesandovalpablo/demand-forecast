from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

#--------------------------------------------------------------------------
#Se define el esquema de cad archivo, si lago cambia solo se actualiza aqui
#--------------------------------------------------------------------------

SCHEMAS =  {
    "train":{
        "columns":[
            "id","date","store_nbr",
            "family","sales","onpromotion"
        ],
        "dtypes":{
            "id":           "int64",
            "store_nbr":    "int64",
            "family":        "object",
            "sales":        "float64",
            "onpromotion":  "int64"
        },
        "parse_dates": ["date"]
    },
    "test":{
        "columns":[
            "id","date","store_nbr",
            "family","onpromotion"
        ],
        "dtypes":{
            "id":           "int64",
            "store_nbr":    "int64",
            "family":        "object",
            "onpromotion":  "int64"
        },
        "parse_dates": ["date"]
    },
    "stores":{
        "columns":[
            "store_nbr","city","state",
            "type","cluster"
        ],
        "dtypes":{
            "store_nbr":        "int64",
            "city":             "object",
            "state":            "object",
            "type":             "object",
            "cluester":         "int64"          
        },
        "parse_dates":[]
    },
    "oil":{
        "columns": ["date","dcoilwtico"],
        "dtypes":{
            "dcoilwtico":"float64"
        },
        "parse_dates":["date"]
    },
    "holidays_events": {
        "columns": [
            "date", "type", "locale",
            "locale_name", "description", "transferred"
        ],
        "dtypes": {
            "type":        "object",
            "locale":      "object",
            "locale_name": "object",
            "description": "object",
            "transferred": "bool"
        },
        "parse_dates": ["date"]
    },
    "transactions": {
        "columns": ["date", "store_nbr", "transactions"],
        "dtypes": {
            "store_nbr":    "int64",
            "transactions": "int64"
        },
        "parse_dates": ["date"]
    }   
}


# ------------------------
# Funciones de validacion
# ------------------------

def _validate_columns(df:pd.DataFrame, name:str, expected_cols: list) -> None:
    """
    Valida que el dataframe cumpla respecto al esquema
    """
    missing = set(expected_cols) -  set(df.columns)
    extra =  set(df.columns) - set (expected_cols)

    if missing:
        logger.error(f"{name}: columnas faltantes : {missing}")
        raise ValueError(
            f"Columnas faltantes en {name}: {missing}"
        )
    if extra:
        logger.warning(f"{name}: columnas extras detectadas : {extra}")


def _validate_nulls(df:pd.DataFrame, name:str) -> None:
    """Reporta nulos por columna"""
    nulls =  df.isnull().sum()
    nulls = nulls[nulls > 0]

    if not nulls.empty:
        for col, count in nulls.items():
            pct = count / len(df) * 100
            logger.warning(
                f"{name}: nulos ene '{col}' -> "
                f"{count:,}({pct:.1f}%)"
            )
    else:
        logger.info(f"{name}: sin nulos")

def _validate_schema(df:pd.DataFrame, name: str, schema: dict) -> None:
    """
    Ejecuta validaciones sobre el DF
    """    
    logger.info(f"Validando esuqema de {name}...")
    _validate_columns(df,name,schema["columns"])
    _validate_nulls(df,name)
    logger.info(
        f"{name}: esuqema valido |"
        f"shape: {df.shape}"
    )


# -------------------------------
# Funciones de carga individuales
# -------------------------------

def _load_csv(name:str, data_path: Path) -> pd.DataFrame:
    """
    Carga Un CSV, valida el esquema y retorna un DataFrame

    Params:
        name:       nombre del archivo sin extension
        data_path:  ruta base del archivo csv
    
    Retorna:
        Dataframe valido
    """

    path= data_path / f"{name}.csv"
    schema = SCHEMAS[name]

    if not path.exists():
        logger.error(f"Archivo no encotrado en:{path}")
        raise FileNotFoundError(f"No se encontro: {path}")
    
    logger.info(f"Cargando {name}.csv")
    df = pd.read_csv(
        path,
        parse_dates=schema["parse_dates"]
    )
    _validate_schema(df, name, schema)
    return df

def load_raw_data(data_path:str = None) -> dict:
    """
    Carga los acrhivos csv deseados.

    params: 
        data_path: ruta a la carpeta de csv's
                si es none, se usan los valores de config.yaml

    retrona:
        dict con DataFrames:
        {
            'train':        pd.Dataframe,
            'test':         pd.DataFrame,
            'stores':       pd.Dataframe,
            'oil':          pd.DataFrame,
            'holidays':     pd.DataFrame,
            'transactions': pd.DataFrame
        }
    """

    path = Path(data_path or config['paths']['data_raw'])
    logger.info(f"IUniciando carga de datos desde: {path}")

    data = {
        'train':        _load_csv('train',            path),
        'test':         _load_csv('test',             path),
        'stores':       _load_csv('stores',           path),
        'oil':          _load_csv('oil',              path),
        'holidays':     _load_csv('holidays_events',  path),
        'transactions': _load_csv('transactions',     path)
    }

    logger.info("Archivos cargados correctamente")
    logger.info(f"  train:        {data['train'].shape}")
    logger.info(f"  test:         {data['test'].shape}")
    logger.info(f"  stores:       {data['stores'].shape}")
    logger.info(f"  oil:          {data['oil'].shape}")
    logger.info(f"  holidays:     {data['holidays'].shape}")
    logger.info(f"  transactions: {data['transactions'].shape}")

    return data

