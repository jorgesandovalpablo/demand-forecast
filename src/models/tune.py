"""
Optuna hyperparameter tuning para LightGBM.

Estrategia agresiva (laptop sin GPU):
  - Subsampleo 30% de filas (stratificado por store+family)
  - 4 folds walk-forward (vs 5 en entrenamiento normal)
  - 600 boosting rounds con early stopping 80
  - Feature engineering caching (una sola vez por study)
  - Ranges diferenciados por horizon (h7: L1, h30: Huber)
  - Storage SQLite para resume automático (load_if_exists)

Coste estimado: ~10-15 min/trial → 150 trials en ~25-35 horas.
"""
import json
import numpy as np
import pandas as pd
import mlflow
import optuna
import optuna.storages
import joblib
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import config
from src.utils.seed import set_global_seed
from src.features.build_features import DemandFeatureEngineer
from src.models.train import (
    get_feature_cols,
    _train_fold,
    _train_final_model,
    setup_mlflow,
)
from src.models.validation import walk_forward_splits

logger = get_logger(__name__)


# ─────────────────────────────────────────
# 1. Sugerencia de hiperparámetros
# ─────────────────────────────────────────
def suggest_params(trial: optuna.Trial, horizon: int) -> dict:
    """
    Space de búsqueda para Optuna — ranges diferenciados por horizon.

    Fijo: objective, metric, random_state, n_jobs, verbosity.
    Tunneables (horizon-specific): num_leaves, learning_rate,
        min_data_in_leaf, lambda_l1, lambda_l2.
    Tunneables (shared): feature_fraction, bagging_fraction,
        bagging_freq, max_bin.
    """
    fixed = {
        "objective": "regression_l1" if horizon == 7 else "huber",
        "metric": "mae",
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": 1,
    }

    if horizon == 7:
        tunable = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 256),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.2, log=True
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 5, 200, log=True
            ),
            "lambda_l1": trial.suggest_float(
                "lambda_l1", 1e-8, 5.0, log=True
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 1e-8, 5.0, log=True
            ),
        }
    else:
        tunable = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 256),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.15, log=True
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 5, 300, log=True
            ),
            "lambda_l1": trial.suggest_float(
                "lambda_l1", 1e-8, 10.0, log=True
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 1e-8, 10.0, log=True
            ),
        }

    shared = {
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.4, 1.0
        ),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.4, 1.0
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "max_bin": trial.suggest_int("max_bin", 100, 300),
    }

    return {**fixed, **tunable, **shared}


def build_params_from_dict(
    params_dict: dict, horizon: int
) -> dict:
    """
    Construye el dict completo de params a partir de un dict
    de tuneables (ej. study.best_params).

    Añade los params fijo (objective, metric, etc.) a los
    tuneables guardados en el JSON de Optuna.
    """
    fixed = {
        "objective": "regression_l1" if horizon == 7 else "huber",
        "metric": "mae",
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }
    return {**fixed, **params_dict}


# ─────────────────────────────────────────
# 2. Función objetivo de Optuna
# ─────────────────────────────────────────
def _objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    feature_cols: list,
    horizon: int,
    n_folds: int,
    max_boost_round: int,
    early_stopping_rounds: int,
) -> float:
    """
    Evalúa un trial: sugiere params → 4-fold CV → retorna mean MAE.

    Cada trial usa subsampleo (ya aplicado al df de entrada) y
    configuración reducida para ser viable en CPU (~8-12 min).
    """
    params = suggest_params(trial, horizon)

    fold_maes = []
    for train_idx, val_idx, fold_info in walk_forward_splits(
        df, n_folds=n_folds
    ):
        _, fold_result = _train_fold(
            df=df,
            train_idx=train_idx,
            val_idx=val_idx,
            feature_cols=feature_cols,
            params=params,
            fold_info=fold_info,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=max_boost_round,
        )
        fold_maes.append(fold_result.mae)

        # Reportar para pruning
        trial.report(np.mean(fold_maes), step=fold_info['fold'])
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_maes))


# ─────────────────────────────────────────
# 3. Búsqueda Optuna completa
# ─────────────────────────────────────────
def run_optuna_search(
    horizon: int,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    output_dir: str = "reports/optuna",
) -> tuple:
    """
    Ejecuta la búsqueda de hiperparámetros con Optuna.

    Flujo:
        1. Feature engineering una sola vez (~5-10 min)
        2. Subsampleo estratificado 30%
        3. Optuna study (n_trials o timeout)
        4. Modelo final con best params en datos completos
        5. Guardado de resultados

    Retorna:
        tuple: (study, best_model, best_metrics)
    """
    optuna_cfg = config.get('optuna', {})
    n_trials = n_trials or optuna_cfg.get('n_trials', 150)
    timeout = timeout or optuna_cfg.get('timeout_seconds', 28800)
    n_folds = optuna_cfg.get('n_folds', 4)
    subsample_ratio = optuna_cfg.get('subsample_ratio', 0.30)
    max_boost_round = optuna_cfg.get('max_boost_round', 600)
    early_stopping_rounds = optuna_cfg.get('early_stopping_rounds', 80)
    storage_url = optuna_cfg.get('storage')

    set_global_seed(config['project']['seed'])
    setup_mlflow()

    # ── 1. Feature engineering una vez ──
    logger.info("=" * 50)
    logger.info(f"Optuna Search — horizon={horizon}")
    logger.info(f"  Trials: {n_trials} | Timeout: {timeout}s")
    logger.info(f"  Folds: {n_folds} | Subsample: {subsample_ratio}")
    logger.info(f"  Boost rounds: {max_boost_round}")
    logger.info(f"  Early stopping: {early_stopping_rounds}")
    logger.info("=" * 50)

    features_path = "data/processed/train_processed.parquet"
    logger.info(f"Cargando datos desde: {features_path}")
    df_processed = pd.read_parquet(features_path)

    pipeline = DemandFeatureEngineer(horizon)
    pipeline.fit(df_processed)
    df = pipeline.transform(df_processed, is_train=True)
    feature_cols = get_feature_cols(df)

    logger.info(f"  Features: {len(feature_cols)} | Filas: {len(df):,}")

    # ── 2. Subsampleo estratificado ──
    if subsample_ratio < 1.0:
        df_sub = df.sample(
            frac=subsample_ratio,
            random_state=config['project']['seed'],
            weights=df[['store_nbr', 'family']].apply(
                lambda r: hash(tuple(r)) % 10000 / 10000, axis=1
            ),
        )
        logger.info(f"  Subsample: {len(df_sub):,} filas ({subsample_ratio:.0%})")
    else:
        df_sub = df

    # ── 3. Optuna study + MLflow ──
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name=f"optuna_h{horizon}"):
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("timeout", timeout)
        mlflow.log_param("subsample_ratio", subsample_ratio)
        mlflow.log_param("n_folds_tune", n_folds)
        mlflow.log_param("max_boost_round", max_boost_round)
        mlflow.log_param("early_stopping", early_stopping_rounds)

        pruner = optuna.pruners.HyperbandPruner(
            min_resource=20,
            max_resource=max_boost_round,
            reduction_factor=4,
        )

        # ── Storage SQLite para resume ──
        storage = None
        if storage_url:
            storage = optuna.storages.RDBStorage(url=storage_url)

        study = optuna.create_study(
            study_name=f"lgbm_h{horizon}",
            storage=storage,
            direction="minimize",
            pruner=pruner,
            load_if_exists=True,
        )

        # ── Calcular trials restantes (resume support) ──
        target_trials = n_trials
        if storage:
            from optuna.trial import TrialState
            n_completed = len(study.get_trials(
                states=(TrialState.COMPLETE,)
            ))
            n_pruned = len(study.get_trials(
                states=(TrialState.PRUNED,)
            ))
            remaining = max(0, target_trials - n_completed)
            logger.info(
                f"Study existente: {n_completed} completados, "
                f"{n_pruned} pruned → "
                f"{remaining} restantes de {target_trials}"
            )
        else:
            remaining = target_trials

        mlflow.log_param("storage", storage_url or "in-memory")
        mlflow.log_param("target_trials", target_trials)
        if storage:
            mlflow.log_param("trials_completed", n_completed)
            mlflow.log_param("trials_remaining", remaining)

        # ── Optimize con MaxTrialsCallback safety net ──
        callbacks = []
        if storage:
            from optuna.study import MaxTrialsCallback
            from optuna.trial import TrialState
            callbacks.append(
                MaxTrialsCallback(
                    target_trials,
                    states=(TrialState.COMPLETE,),
                )
            )

        logger.info("Iniciando Optuna study...")
        if remaining > 0:
            study.optimize(
                lambda trial: _objective(
                    trial, df_sub, feature_cols, horizon,
                    n_folds, max_boost_round,
                    early_stopping_rounds,
                ),
                n_trials=remaining,
                timeout=timeout,
                callbacks=callbacks,
                show_progress_bar=True,
            )
        else:
            logger.info(
                "Study ya completado. "
                "Entrenando modelo final con best params."
            )

        best = study.best_trial
        logger.info(f"Best trial #{best.number}: MAE={best.value:.4f}")
        logger.info(f"Best params: {best.params}")

        mlflow.log_metric("best_mae", best.value)
        mlflow.log_param("best_trial", best.number)
        mlflow.log_params(best.params)

        # ── 4. Modelo final con best params en datos completos ──
        best_params = build_params_from_dict(best.params, horizon)
        best_model, best_metrics = _train_with_best_params(
            horizon, best_params, df, feature_cols, pipeline
        )

        # ── 5. Guardar resultados ──
        params_path = _save_tuning_results(
            horizon, study, best_model, best_metrics, output_dir
        )
        mlflow.log_artifact(str(params_path))
        mlflow.log_metric("cv_mae_final", best_metrics.get("mae_mean", 0))
        mlflow.log_metric("cv_wape_final", best_metrics.get("wape_mean", 0))

    return study, best_model, best_metrics


# ─────────────────────────────────────────
# 4. Entrenamiento con best params
# ─────────────────────────────────────────
def _train_with_best_params(
    horizon: int,
    params: dict,
    df: pd.DataFrame,
    feature_cols: list,
    pipeline: DemandFeatureEngineer,
) -> tuple:
    """
    Entrena el modelo final con los mejores params de Optuna.
    Usa walk-forward CV (5 folds, full data) para obtener métricas
    justas y luego entrena el modelo de producción.
    """
    from src.models.validation import (
        walk_forward_splits,
        summarize_validation,
    )

    logger.info("Entrenando modelo final con best params...")
    set_global_seed(config['project']['seed'])

    # CV con 5 folds para métricas justas
    fold_results = []
    best_n_estimators_list = []

    for train_idx, val_idx, fold_info in walk_forward_splits(
        df, n_folds=config['training']['n_folds']
    ):
        model, fold_result = _train_fold(
            df=df,
            train_idx=train_idx,
            val_idx=val_idx,
            feature_cols=feature_cols,
            params=params,
            fold_info=fold_info,
        )
        fold_results.append(fold_result)
        best_n_estimators_list.append(model.best_iteration)

    summary = summarize_validation(fold_results)
    best_n_estimators = int(np.mean(best_n_estimators_list))

    logger.info(
        f"CV results: MAE={summary.mae_mean:.4f} "
        f"(±{summary.mae_std:.4f}) | "
        f"best_n_estimators={best_n_estimators}"
    )

    # Modelo final
    final_model = _train_final_model(
        df=df,
        feature_cols=feature_cols,
        params=params,
        best_n_estimators=best_n_estimators,
    )

    return final_model, vars(summary)


# ─────────────────────────────────────────
# 5. Guardar resultados de tuning
# ─────────────────────────────────────────
def _save_tuning_results(
    horizon: int,
    study: optuna.Study,
    model,
    metrics: dict,
    output_dir: str,
) -> Path:
    """
    Guarda best params, métricas y modelo tunneado.
    Retorna la ruta del archivo de params (para MLflow artifact logging).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Best params JSON
    params_path = out_path / f"best_params_h{horizon}.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=2, default=str)
    logger.info(f"Best params guardados: {params_path}")

    # Métricas JSON
    metrics_path = out_path / f"best_metrics_h{horizon}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Best metrics guardadas: {metrics_path}")

    # Study stats
    stats_path = out_path / f"study_stats_h{horizon}.json"
    stats = {
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_value": study.best_trial.value,
        "direction": str(study.direction),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Study stats guardadas: {stats_path}")

    # Modelo tunneado
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)
    model_path = models_path / f"lgbm_tuned_h{horizon}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Modelo tunneado guardado: {model_path}")

    # Guidance para el usuario
    logger.info("=" * 50)
    logger.info("Tuning completado. Para entrenar el modelo final:")
    logger.info(
        f"  python src/models/train.py --horizon {horizon}"
        f" --params-file {params_path}"
    )
    logger.info("=" * 50)

    return params_path


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning para LightGBM"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[7, 30],
        required=True,
        help="Horizonte de prediccion: 7 (diario) o 30 (mensual)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Numero de trials (default: config optuna.n_trials)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout en segundos (default: config optuna.timeout_seconds)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/optuna",
        help="Directorio de salida para resultados",
    )
    args = parser.parse_args()

    run_optuna_search(
        horizon=args.horizon,
        n_trials=args.trials,
        timeout=args.timeout,
        output_dir=args.output_dir,
    )
