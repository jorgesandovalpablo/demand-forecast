import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import joblib
from pathlib import Path
from typing import Optional
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
        'id', 'holiday_description','transactions'
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
    import os
    load_dotenv()

    os.environ['MLFLOW_TRACKING_USERNAME'] = (
        os.getenv('MLFLOW_TRACKING_USERNAME','')
    )
    os.environ['MLFLOW_TRACKING_PASSWORD'] = (
        os.getenv('MLFLOW_TRACKING_PASSWORD','')
    )
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
    fold_info: dict,
    early_stopping_rounds: Optional[int] = None,
    num_boost_round: Optional[int] = None,
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

    top_families = config['model']['top_families']
    w_train = np.ones(len(y_train))
    w_train[df.loc[train_idx, 'family'].isin(top_families)] = config['training']['weight_value']

    # Resolver num_boost_round y early_stopping
    if num_boost_round is None:
        num_boost_round = params.get('n_estimators', 1000)
    if early_stopping_rounds is None:
        early_stopping_rounds = 150

    # Limpiar params para lgb.train (n_estimators no es param LightGBM)
    train_params = {k: v for k, v in params.items() if k != 'n_estimators'}

    # Dataset de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    val_data   = lgb.Dataset(X_val,   label=y_val,
                             reference=train_data)

    # Callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds,
                           verbose=False),
        lgb.log_evaluation(period=100)
    ]

    # Entrenamiento
    model = lgb.train(
        params=train_params,
        train_set=train_data,
        num_boost_round=num_boost_round,
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
        rmsle       = metrics['rmsle'],
        wape        = metrics['wape']
    )

    logger.info(
        f"  Fold {fold_info['fold']} → "
        f"RMSE: {metrics['rmse']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"MAPE: {metrics['mape']:.2f}% |"
        f"WAPE: {metrics['wape']:.2f}"
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

    top_families = config['model']['top_families']
    final_weights = np.ones(len(y))
    final_weights[train_df['family'].isin(top_families)] = config['training']['weight_value']

    train_data = lgb.Dataset(X, label=y, weight=final_weights)  

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
    feature_cols: list,
    pipeline,
    metrics: dict,
    output_suffix: str = ""
) -> tuple[Path, Path]:
    """
    Guarda el modelo entrenado, el pipeline de features en disco
    y lo registra en MLflow.

    El parámetro output_suffix permite escribir a un artefacto de
    staging (ej. '_new') sin tocar los artefactos de producción.
    """
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)

    model_path = models_path / f"lgbm_h{horizon}{output_suffix}.pkl"
    joblib.dump(model, model_path)

    features_path = models_path / f"features_h{horizon}{output_suffix}.pkl"
    joblib.dump(feature_cols, features_path)

    pipeline_path = models_path / f"feature_pipeline_h{horizon}{output_suffix}.pkl"
    pipeline.save(pipeline_path)

    logger.info(f"Modelo guardado:   {model_path}")
    logger.info(f"Features guardadas: {features_path}")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"Pipeline guardado: {pipeline_path}")

    return model_path, pipeline_path

# ─────────────────────────────────────────
# Función principal — punto de entrada
# ─────────────────────────────────────────
def run_training(
    horizon: int,
    output_suffix: str = "",
    params_file: Optional[str] = None,
) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento
    para un horizonte específico.

    Parámetros:
        horizon: 7 → modelo diario
                 30 → modelo mensual
        output_suffix: sufijo para los artefactos de salida.
                       Vacío escribe en producción; '_new'
                       escribe a staging (usado por retrain.py).

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
    params = dict(config['model'][params_key])

    if params_file:
        import json
        with open(params_file) as f:
            override = json.load(f)
        params.update(override)
        logger.info(f"Params override desde {params_file}")

    # Cargar features (Data Procesada, no Feature-Engineered)
    features_path = "data/processed/train_processed.parquet"
    logger.info(f"Cargando historial procesado desde: {features_path}")
    df_processed = pd.read_parquet(features_path)
    
    # Construcción Stateful de Features
    from src.features.build_features import DemandFeatureEngineer
    pipeline = DemandFeatureEngineer(horizon)
    pipeline.fit(df_processed)
    df = pipeline.transform(df_processed, is_train=True)
    
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

    with mlflow.start_run(run_name=model_name, nested=True):

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
        mlflow.log_metric("cv_wape_mean", summary.wape_mean)

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

        # Guardar y loggear modelo y pipeline
        model_path, pipeline_path = _save_model(
            final_model, horizon, feature_cols,
            pipeline,
            vars(summary),
            output_suffix=output_suffix
        )
        mlflow.lightgbm.log_model(
            final_model,
            artifact_path=model_name
        )
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(pipeline_path))

        logger.info("=" * 50)
        logger.info(" Entrenamiento completado")
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
            'horizon':  horizon,
            'df':       df
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