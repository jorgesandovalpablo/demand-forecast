"""
Optuna hyperparameter tuning para LightGBM.

Estrategia conservadora (laptop sin GPU):
  - Subsampleo 15% de filas (stratificado por store+family)
  - 3 folds walk-forward (vs 5 en entrenamiento normal)
  - 400 boosting rounds con early stopping 50 (vs 1500/150)
  - Feature engineering caching (una sola vez por study)

Coste estimado: ~8-12 min/trial → 50 trials en ~8-10 horas.
"""
import json
import numpy as np
import pandas as pd
import optuna
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
)
from src.models.validation import walk_forward_splits

logger = get_logger(__name__)


# ─────────────────────────────────────────
# 1. Sugerencia de hiperparámetros
# ─────────────────────────────────────────
def suggest_params(trial: optuna.Trial, horizon: int) -> dict:
    """
    Space de búsqueda para Optuna.

    Fijo: objective, metric, random_state, n_jobs, verbosity.
    Tunneables: num_leaves, learning_rate, min_data_in_leaf,
                lambda_l1, lambda_l2, feature_fraction,
                bagging_fraction, bagging_freq, max_bin.
    """
    return {
        "objective": "regression_l1" if horizon == 7 else "huber",
        "metric": "mae",
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": 1,

        "num_leaves": trial.suggest_int("num_leaves", 20, 127),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf", 10, 200, log=True
        ),
        "lambda_l1": trial.suggest_float(
            "lambda_l1", 1e-8, 10.0, log=True
        ),
        "lambda_l2": trial.suggest_float(
            "lambda_l2", 1e-8, 10.0, log=True
        ),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.5, 1.0
        ),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.5, 1.0
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "max_bin": trial.suggest_int("max_bin", 100, 300),
    }


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
    Evalúa un trial: sugiere params → 3-fold CV → retorna mean MAE.

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
        2. Subsampleo estratificado 15%
        3. Optuna study (n_trials o timeout)
        4. Modelo final con best params en datos completos
        5. Guardado de resultados

    Retorna:
        tuple: (study, best_model, best_metrics)
    """
    optuna_cfg = config.get('optuna', {})
    n_trials = n_trials or optuna_cfg.get('n_trials', 50)
    timeout = timeout or optuna_cfg.get('timeout_seconds', 36000)
    n_folds = optuna_cfg.get('n_folds', 3)
    subsample_ratio = optuna_cfg.get('subsample_ratio', 0.15)
    max_boost_round = optuna_cfg.get('max_boost_round', 400)
    early_stopping_rounds = optuna_cfg.get('early_stopping_rounds', 50)

    set_global_seed(config['project']['seed'])

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

    # ── 3. Optuna study ──
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=max_boost_round,
        reduction_factor=3,
    )
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name=f"lgbm_h{horizon}",
    )

    logger.info("Iniciando Optuna study...")
    study.optimize(
        lambda trial: _objective(
            trial, df_sub, feature_cols, horizon,
            n_folds, max_boost_round, early_stopping_rounds,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info(f"Best trial #{best.number}: MAE={best.value:.4f}")
    logger.info(f"Best params: {best.params}")

    # ── 4. Modelo final con best params en datos completos ──
    best_params = build_params_from_dict(best.params, horizon)
    best_model, best_metrics = _train_with_best_params(
        horizon, best_params, df, feature_cols, pipeline
    )

    # ── 5. Guardar resultados ──
    _save_tuning_results(
        horizon, study, best_model, best_metrics, output_dir
    )

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
) -> None:
    """Guarda best params, métricas y modelo tunneado."""
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
