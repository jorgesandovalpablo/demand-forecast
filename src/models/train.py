import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import joblib
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import config
from src.utils.seed import set_global_seed
from src.models.validation import (
    walk_forward_splits,
    compute_metrics,
    FoldResult,
    summarize_validation,
    plot_folds
)

logger = get_logger(__name__)


# ─────────────────────────────────────────
# Definición de features
# ─────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Retorna las columnas que entran al modelo.
    Excluye columnas de identificación y target.
    """
    exclude = [
        'date', 'sales', 'sales_raw',
        'id', 'holiday_description'
    ]
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────
# 1. Configuración de MLflow
# ─────────────────────────────────────────
def setup_mlflow() -> None:
    """
    Configura MLflow para trackear experimentos
    en DagsHub.
    """
    from dotenv import load_dotenv
    load_dotenv()

    tracking_uri = config['mlflow']['tracking_uri']
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    logger.info(f"MLflow configurado: {tracking_uri}")


# ─────────────────────────────────────────
# 2. Entrenamiento de un fold
# ─────────────────────────────────────────
def _train_fold(
    df: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    feature_cols: list,
    params: dict,
    fold_info: dict
) -> tuple:
    """
    Entrena el modelo en un fold y retorna
    el modelo y las métricas.

    Retorna:
        tuple: (model, FoldResult)
    """
    target = config['data']['target']

    X_train = df.loc[train_idx, feature_cols]
    y_train = df.loc[train_idx, target]
    X_val   = df.loc[val_idx, feature_cols]
    y_val   = df.loc[val_idx, target]

    # Dataset de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val,
                             reference=train_data)

    # Callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=50,
                           verbose=False),
        lgb.log_evaluation(period=100)
    ]

    # Entrenamiento
    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=params.get('n_estimators', 1000),
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )

    # Predicción y métricas
    y_pred   = model.predict(X_val)
    metrics  = compute_metrics(
        y_val.values, y_pred,
        in_log_scale=True
    )

    fold_result = FoldResult(
        fold        = fold_info['fold'],
        train_start = fold_info['train_start'],
        train_end   = fold_info['train_end'],
        val_start   = fold_info['val_start'],
        val_end     = fold_info['val_end'],
        n_train     = fold_info['n_train'],
        n_val       = fold_info['n_val'],
        rmse        = metrics['rmse'],
        mae         = metrics['mae'],
        mape        = metrics['mape'],
        rmsle       = metrics['rmsle']
    )

    logger.info(
        f"  Fold {fold_info['fold']} → "
        f"RMSE: {metrics['rmse']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"MAPE: {metrics['mape']:.2f}%"
    )

    return model, fold_result


# ─────────────────────────────────────────
# 3. Entrenamiento final
# Usa todos los datos sin val para producción
# ─────────────────────────────────────────
def _train_final_model(
    df: pd.DataFrame,
    feature_cols: list,
    params: dict,
    best_n_estimators: int
) -> lgb.Booster:
    """
    Entrena el modelo final con todos los datos.
    Se usa el mejor número de estimadores
    encontrado en la validación.
    """
    logger.info("Entrenando modelo final con todos los datos...")
    target = config['data']['target']

    # Excluir últimas 8 semanas (test set real)
    test_weeks = config['training']['test_size_weeks']
    cutoff     = df['date'].max() - pd.Timedelta(weeks=test_weeks)
    train_df   = df[df['date'] <= cutoff]

    X = train_df[feature_cols]
    y = train_df[target]

    train_data = lgb.Dataset(X, label=y)

    # Sin early stopping en el modelo final
    # usamos el n_estimators óptimo de la CV
    final_params = {**params, 'n_estimators': best_n_estimators}

    model = lgb.train(
        params=final_params,
        train_set=train_data,
        num_boost_round=best_n_estimators
    )

    logger.info(
        f"  Modelo final entrenado con "
        f"{len(train_df):,} filas"
    )
    return model


# ─────────────────────────────────────────
# 4. Guardar modelo
# ─────────────────────────────────────────
def _save_model(
    model: lgb.Booster,
    horizon: int,
    metrics: dict
) -> Path:
    """
    Guarda el modelo entrenado en disco
    y lo registra en MLflow.
    """
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)

    model_name = f"lgbm_h{horizon}.pkl"
    model_path = models_path / model_name

    joblib.dump(model, model_path)
    logger.info(f"Modelo guardado: {model_path}")

    return model_path


# ─────────────────────────────────────────
# Función principal — punto de entrada
# ─────────────────────────────────────────
def run_training(horizon: int) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento
    para un horizonte específico.

    Parámetros:
        horizon: 7 → modelo diario
                 30 → modelo mensual

    Retorna:
        dict con modelo entrenado y métricas
    """
    set_global_seed(config['project']['seed'])
    setup_mlflow()

    # Seleccionar parámetros según horizonte
    params_key = (
        'params_lgbm_diario'
        if horizon == 7
        else 'params_lgbm_mensual'
    )
    params = config['model'][params_key]

    # Cargar features
    features_path = (
        f"data/processed/train_features_d{horizon}.parquet"
    )
    logger.info(f"Cargando features desde: {features_path}")
    df = pd.read_parquet(features_path)
    feature_cols = get_feature_cols(df)

    logger.info("=" * 50)
    logger.info(
        f"Iniciando entrenamiento "
        f"{'DIARIO' if horizon == 7 else 'MENSUAL'}"
    )
    logger.info(f"  Features:  {len(feature_cols)}")
    logger.info(f"  Filas:     {len(df):,}")
    logger.info(f"  Horizonte: {horizon} días")
    logger.info("=" * 50)

    # ── MLflow run ──
    model_name = (
        f"lgbm_{'daily' if horizon == 7 else 'monthly'}"
    )

    with mlflow.start_run(run_name=model_name):

        # Loggear parámetros
        mlflow.log_params(params)
        mlflow.log_param("horizon",       horizon)
        mlflow.log_param("n_features",    len(feature_cols))
        mlflow.log_param("n_rows",        len(df))
        mlflow.log_param("feature_names", feature_cols)

        # ── Walk-forward cross validation ──
        fold_results = []
        best_n_estimators_list = []

        for train_idx, val_idx, fold_info in walk_forward_splits(
            df,
            n_folds=config['training']['n_folds']
        ):
            model, fold_result = _train_fold(
                df=df,
                train_idx=train_idx,
                val_idx=val_idx,
                feature_cols=feature_cols,
                params=params,
                fold_info=fold_info
            )
            fold_results.append(fold_result)
            best_n_estimators_list.append(
                model.best_iteration
            )

        # Resumen de validación
        summary = summarize_validation(fold_results)

        # Loggear métricas de CV en MLflow
        mlflow.log_metric("cv_rmse_mean",  summary.rmse_mean)
        mlflow.log_metric("cv_rmse_std",   summary.rmse_std)
        mlflow.log_metric("cv_mae_mean",   summary.mae_mean)
        mlflow.log_metric("cv_mape_mean",  summary.mape_mean)
        mlflow.log_metric("cv_rmsle_mean", summary.rmsle_mean)

        # Gráfica de folds
        plot_path = (
            f"notebooks/figures/"
            f"cv_folds_h{horizon}.png"
        )
        plot_folds(fold_results, save_path=plot_path)
        mlflow.log_artifact(plot_path)

        # ── Modelo final ──
        best_n_estimators = int(
            np.mean(best_n_estimators_list)
        )
        logger.info(
            f"Mejor n_estimators promedio CV: "
            f"{best_n_estimators}"
        )

        final_model = _train_final_model(
            df=df,
            feature_cols=feature_cols,
            params=params,
            best_n_estimators=best_n_estimators
        )

        # Guardar y loggear modelo
        model_path = _save_model(
            final_model, horizon,
            vars(summary)
        )
        mlflow.lightgbm.log_model(
            final_model,
            artifact_path=model_name
        )
        mlflow.log_artifact(str(model_path))

        logger.info("=" * 50)
        logger.info("✅ Entrenamiento completado")
        logger.info(
            f"  RMSE:  {summary.rmse_mean:.4f} "
            f"(±{summary.rmse_std:.4f})"
        )
        logger.info(
            f"  MAPE:  {summary.mape_mean:.2f}%"
        )
        logger.info("=" * 50)

        return {
            'model':    final_model,
            'summary':  summary,
            'features': feature_cols,
            'horizon':  horizon
        }


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
        help='Horizonte de predicción: 7 (diario) o 30 (mensual)'
    )
    args = parser.parse_args()

    run_training(horizon=args.horizon)