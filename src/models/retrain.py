import numpy as np
import pandas as pd
import mlflow
import joblib
import shutil
from pathlib import Path
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.config import config
from src.utils.seed import set_global_seed
from src.data.ingestion import load_raw_data
from src.data.preprocessing import run_preprocessing
from src.features.build_features import build_features
from src.models.train import run_training, get_feature_cols
from src.models.validation import compute_metrics
from src.models.predict import ModelRegistry

logger = get_logger(__name__)


# ─────────────────────────────────────────
# 1. Cargar métricas del modelo actual
# ─────────────────────────────────────────
def _load_current_metrics(horizon: int) -> dict:
    """
    Carga las métricas del modelo actualmente
    en producción para comparar después del
    reentrenamiento.

    Si no hay métricas previas retorna
    valores muy altos para que cualquier
    modelo nuevo sea mejor.
    """
    metrics_path = Path(
        f"data/predictions/"
        f"family_metrics_h{horizon}.parquet"
    )

    if not metrics_path.exists():
        logger.warning(
            "No hay métricas previas — "
            "cualquier modelo nuevo será aceptado"
        )
        return {
            'rmse': float('inf'),
            'mae':  float('inf'),
            'mape': float('inf'),
            'rmsle': float('inf'),
            'wape': float('inf')
        }

    metrics_df = pd.read_parquet(metrics_path)
    avg = metrics_df[
        ['rmse', 'mae', 'mape', 'rmsle','wape']
    ].mean()

    current = {
        'rmse':  round(float(avg['rmse']),  4),
        'mae':   round(float(avg['mae']),   4),
        'mape':  round(float(avg['mape']),  4),
        'rmsle': round(float(avg['rmsle']), 4),
        'wape': round(float(avg['wape']), 4)
    }

    logger.info(f"Métricas modelo actual: {current}")
    return current


# ─────────────────────────────────────────
# 2. Evaluar nuevo modelo sobre test set
# ─────────────────────────────────────────
def _evaluate_new_model(
    horizon: int,
    df: pd.DataFrame
) -> dict:
    """
    Evalúa el modelo recién entrenado
    sobre el test set para comparar
    con el modelo anterior.
    """
    logger.info("Evaluando nuevo modelo...")

    model_path = Path(f"models/lgbm_h{horizon}_new.pkl")
    model      = joblib.load(model_path)

    feature_cols = get_feature_cols(df)
    target       = config['data']['target']

    # Test set → últimas 8 semanas
    test_weeks = config['training']['test_size_weeks']
    cutoff     = (
        df['date'].max() -
        pd.Timedelta(weeks=test_weeks)
    )
    test_df  = df[df['date'] > cutoff]
    X_test   = test_df[feature_cols]
    y_test   = test_df[target].values
    y_pred   = model.predict(X_test)

    new_metrics = compute_metrics(
        y_test, y_pred,
        in_log_scale=True
    )

    logger.info(f"Métricas nuevo modelo: {new_metrics}")
    return new_metrics


# ─────────────────────────────────────────
# 3. Comparar modelos y decidir
# ─────────────────────────────────────────
def _should_update_model(
    current_metrics: dict,
    new_metrics: dict,
    threshold: float = 0.01
) -> bool:
    """
    Decide si el nuevo modelo reemplaza
    al modelo en producción.

    Criterio: el nuevo modelo debe mejorar
    el mae al menos en 'threshold' (1% default)
    para ser aceptado.

    Parámetros:
        threshold: mejora mínima requerida (0.01 = 1%)
    """
    current_mae = current_metrics['mae']
    new_mae     = new_metrics['mae']

    # Si no hay modelo previo → siempre actualizar
    if current_mae == float('inf'):
        logger.info(
            "No hay modelo previo → "
            "nuevo modelo aceptado automáticamente"
        )
        return True

    improvement = (current_mae - new_mae) / current_mae

    logger.info(
        f"Comparación de modelos:\n"
        f"  MAE actual:  {current_mae:.4f}\n"
        f"  MAE nuevo:   {new_mae:.4f}\n"
        f"  Mejora:       {improvement*100:.2f}%\n"
        f"  Threshold:    {threshold*100:.1f}%"
    )

    if improvement >= threshold:
        logger.info(
            " Nuevo modelo es mejor -> "
            "se actualizará en producción"
        )
        return True
    else:
        logger.warning(
            " Nuevo modelo no mejora suficiente → "
            "se mantiene modelo actual"
        )
        return False


# ─────────────────────────────────────────
# 4. Rotar modelos
# ─────────────────────────────────────────
def _rotate_models(horizon: int) -> None:
    """
    Reemplaza el modelo en producción
    con el nuevo modelo entrenado.

    Guarda el modelo anterior como backup
    por si necesitas hacer rollback.
    """
    current_path = Path(f"models/lgbm_h{horizon}.pkl")
    new_path     = Path(f"models/lgbm_h{horizon}_new.pkl")
    backup_path  = Path(
        f"models/lgbm_h{horizon}_backup_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f".pkl"
    )

    # Backup del modelo actual
    if current_path.exists():
        shutil.copy(current_path, backup_path)
        logger.info(f"Backup guardado: {backup_path}")

    # Reemplazar modelo en producción
    shutil.move(str(new_path), str(current_path))
    logger.info(
        f"Modelo actualizado: {current_path}"
    )

    # Limpiar backups antiguos
    # Mantener solo los últimos 3
    backups = sorted(
        Path("models").glob(
            f"lgbm_h{horizon}_backup_*.pkl"
        )
    )
    if len(backups) > 3:
        for old_backup in backups[:-3]:
            old_backup.unlink()
            logger.info(
                f"Backup antiguo eliminado: {old_backup}"
            )


# ─────────────────────────────────────────
# Función principal — punto de entrada
# ─────────────────────────────────────────
def run_retraining(
    horizon: int,
    force: bool = False
) -> dict:
    """
    Ejecuta el pipeline completo de
    reentrenamiento.

    Parámetros:
        horizon: 7 (diario) o 30 (mensual)
        force:   si True, actualiza el modelo
                 sin importar las métricas

    Retorna:
        dict con resultados del reentrenamiento
    """
    set_global_seed(config['project']['seed'])

    logger.info("=" * 50)
    logger.info(
        f"Iniciando reentrenamiento "
        f"horizon={horizon} | "
        f"force={force}"
    )
    logger.info(
        f"Timestamp: {datetime.now().isoformat()}"
    )
    logger.info("=" * 50)

    with mlflow.start_run(
        run_name=f"retrain_h{horizon}_"
                 f"{datetime.now().strftime('%Y%m%d')}"
    ):
        mlflow.log_param("horizon",   horizon)
        mlflow.log_param("force",     force)
        mlflow.log_param("timestamp",
                         datetime.now().isoformat())

        # ── Paso 1: Cargar métricas actuales ──
        current_metrics = _load_current_metrics(horizon)
        for k, v in current_metrics.items():
            if v != float('inf'):
                mlflow.log_metric(f"before_{k}", v)

        # ── Paso 2: Ejecutar pipeline de datos ──
        logger.info("Ejecutando pipeline de datos...")
        data  = load_raw_data()
        train, _ = run_preprocessing(data, save=True)

        # ── Paso 3: Entrenar nuevo modelo ──
        logger.info("Entrenando nuevo modelo...")
        result = run_training(horizon=horizon)
        df = result['df']

        # Renombrar modelo entrenado y pipeline a '_new'
        current_path = Path(f"models/lgbm_h{horizon}.pkl")
        new_path = Path(f"models/lgbm_h{horizon}_new.pkl")
        if current_path.exists():
            shutil.copy(str(current_path), str(new_path))
            
        current_pipeline_path = Path(f"models/feature_pipeline_h{horizon}.pkl")
        new_pipeline_path = Path(f"models/feature_pipeline_h{horizon}_new.pkl")
        if current_pipeline_path.exists():
            shutil.copy(str(current_pipeline_path), str(new_pipeline_path))

        # ── Paso 4: Evaluar nuevo modelo ──
        new_metrics = _evaluate_new_model(horizon, df)
        for k, v in new_metrics.items():
            mlflow.log_metric(f"after_{k}", v)

        # ── Paso 5: Decidir si actualizar ──
        model_updated = False
        if force or _should_update_model(
            current_metrics, new_metrics
        ):
            _rotate_models(horizon)
            model_updated = True

            # Limpiar caché para cargar
            # el nuevo modelo en la próxima predicción
            ModelRegistry.clear_cache()
            logger.info(
                " Modelo en producción actualizado"
            )
        else:
            # Eliminar nuevo modelo rechazado
            if new_path.exists():
                new_path.unlink()
            logger.warning(
                " Modelo anterior mantenido "
                "en producción"
            )

        mlflow.log_param(
            "model_updated", model_updated
        )

        logger.info("=" * 50)
        logger.info(" Reentrenamiento completado")
        logger.info(
            f"  Modelo actualizado: {model_updated}"
        )
        logger.info("=" * 50)

        return {
            'horizon':        horizon,
            'model_updated':  model_updated,
            'metrics_before': current_metrics,
            'metrics_after':  new_metrics
        }


# ─────────────────────────────────────────
# Ejecutar desde terminal
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--horizon',
        type=int,
        choices=[7, 30],
        required=True
    )
    parser.add_argument(
        '--force',
        action='store_true',
        default=False,
        help='Forzar actualización sin comparar métricas'
    )
    args = parser.parse_args()

    run_retraining(
        horizon=args.horizon,
        force=args.force
    )