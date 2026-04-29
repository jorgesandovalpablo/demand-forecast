# src/api/main.py
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.utils.logger import get_logger
from src.utils.config import config
from src.models.predict import (
    predict,
    predict_by_store,
    ModelRegistry,
    save_predictions
)
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionItem,
    HealthResponse,
    MetricsResponse,
    RetrainingRequest,
    RetrainingResponse
)

logger = get_logger(__name__)


# ─────────────────────────────────────────
# Estado global de la aplicación
# ─────────────────────────────────────────
app_state = {
    'historical_df': None,
    'models_loaded': []
}


# ─────────────────────────────────────────
# Startup y shutdown
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ejecuta acciones al iniciar y apagar la API.
    Al iniciar: carga datos y modelos en memoria.
    Al apagar:  limpia recursos.
    """
    logger.info("Iniciando API...")

    # Cargar datos históricos
    data_path = Path(
        "data/processed/train_processed.parquet"
    )
    if data_path.exists():
        app_state['historical_df'] = pd.read_parquet(
            data_path
        )
        logger.info(
            f"Datos históricos cargados: "
            f"{app_state['historical_df'].shape}"
        )
    else:
        logger.warning(
            "Datos históricos no encontrados. "
            "Ejecuta primero el pipeline de preprocessing."
        )

    # Pre-cargar modelos en memoria
    for horizon in [7, 30]:
        model_path = Path(f"models/lgbm_h{horizon}.pkl")
        if model_path.exists():
            ModelRegistry.load(horizon)
            app_state['models_loaded'].append(horizon)
            logger.info(f"Modelo horizon={horizon} pre-cargado")
        else:
            logger.warning(
                f"Modelo horizon={horizon} no encontrado"
            )

    logger.info("✅ API lista")
    yield

    # Shutdown
    logger.info("Apagando API...")
    ModelRegistry.clear_cache()
    logger.info("API apagada")


# ─────────────────────────────────────────
# Inicialización de FastAPI
# ─────────────────────────────────────────
app = FastAPI(
    title="Demand Forecast API",
    description=(
        "API para predicción de demanda en "
        "minimercados de Ecuador. "
        "Modelos: LightGBM con horizontes "
        "de 7 y 30 días."
    ),
    version=config['project']['version'],
    lifespan=lifespan
)

# CORS — permite que frontends consuman la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"]
)
async def health_check():
    """
    Verifica que la API está funcionando
    y qué modelos están cargados.
    """
    return HealthResponse(
        status="healthy",
        models_loaded=app_state['models_loaded'],
        version=config['project']['version']
    )


@app.get(
    "/metrics/{horizon}",
    response_model=MetricsResponse,
    tags=["Monitoring"]
)
async def get_metrics(horizon: int):
    """
    Retorna las métricas del modelo
    para un horizonte específico.
    """
    if horizon not in [7, 30]:
        raise HTTPException(
            status_code=400,
            detail="horizon debe ser 7 o 30"
        )

    metrics_path = Path(
        f"data/predictions/"
        f"family_metrics_h{horizon}.parquet"
    )

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Métricas no encontradas para "
                f"horizon={horizon}. "
                f"Ejecuta primero evaluate.py"
            )
        )

    metrics_df = pd.read_parquet(metrics_path)
    avg_metrics = metrics_df[
        ['rmse', 'mae', 'mape', 'rmsle']
    ].mean()

    return MetricsResponse(
        horizon=horizon,
        rmse=round(float(avg_metrics['rmse']), 4),
        mae=round(float(avg_metrics['mae']),   4),
        mape=round(float(avg_metrics['mape']), 4),
        rmsle=round(float(avg_metrics['rmsle']), 4)
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"]
)
async def get_predictions(request: PredictionRequest):
    """
    Genera predicciones de demanda para
    una tienda y horizonte específicos.

    - **store_nbr**: número de tienda (1-54)
    - **horizon**: 7 días o 30 días
    - **family**: familia de producto (opcional)
    """
    if app_state['historical_df'] is None:
        raise HTTPException(
            status_code=503,
            detail="Datos históricos no disponibles"
        )

    if request.horizon not in app_state['models_loaded']:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Modelo para horizon={request.horizon} "
                f"no está cargado"
            )
        )

    try:
        predictions = predict_by_store(
            historical_df=app_state['historical_df'],
            horizon=request.horizon,
            store_nbr=request.store_nbr
        )

        # Filtrar por familia si se especificó
        if request.family:
            predictions = predictions[
                predictions['family'] == request.family
            ]
            if predictions.empty:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Familia '{request.family}' "
                        f"no encontrada para tienda "
                        f"{request.store_nbr}"
                    )
                )

        # Convertir a lista de PredictionItem
        items = [
            PredictionItem(
                date=row['date'],
                store_nbr=row['store_nbr'],
                family=row['family'],
                predicted_sales=row['predicted_sales'],
                lower_bound=row['lower_bound'],
                upper_bound=row['upper_bound']
            )
            for _, row in predictions.iterrows()
        ]

        return PredictionResponse(
            store_nbr=request.store_nbr,
            horizon=request.horizon,
            n_predictions=len(items),
            predictions=items
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor"
        )


@app.post(
    "/retrain",
    response_model=RetrainingResponse,
    tags=["Training"]
)
async def trigger_retraining(
    request: RetrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Dispara el reentrenamiento del modelo
    en background sin bloquear la API.
    """
    from src.models.retrain import run_retraining

    background_tasks.add_task(
        run_retraining,
        horizon=request.horizon,
        force=request.force
    )

    return RetrainingResponse(
        horizon=request.horizon,
        status="retraining_started",
        metrics_before={},
        metrics_after={},
        model_updated=False
    )