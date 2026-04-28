import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

# ─────────────────────────────────────────
# Estructura de resultados por fold
# ─────────────────────────────────────────
@dataclass
class FoldResult:
    """
    Almacena los resultados de un fold
    de validación.
    """
    fold:        int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp
    rmse:        float = 0.0
    mae:         float = 0.0
    mape:        float = 0.0
    rmsle:       float = 0.0
    n_train:     int   = 0
    n_val:       int   = 0
    wape:        float = 0.0


@dataclass
class ValidationResult:
    """
    Almacena los resultados completos
    de la validación walk-forward.
    """
    folds:        list = field(default_factory=list)
    rmse_mean:    float = 0.0
    rmse_std:     float = 0.0
    mae_mean:     float = 0.0
    mae_std:      float = 0.0
    mape_mean:    float = 0.0
    mape_std:     float = 0.0
    rmsle_mean:   float = 0.0
    rmsle_std:    float = 0.0
    wape_mean:    float = 0.0
    wape_std:     float = 0.0


# ─────────────────────────────────────────
# 1. Generador de folds temporales
# ─────────────────────────────────────────
def walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = None,
    val_weeks: int = 4,
    date_col: str = 'date'
) -> Generator:
    """
    Genera splits temporales para walk-forward
    cross validation.

    Cada fold expande el train y mueve el val
    hacia adelante en el tiempo:

    Fold 1: Train [ene-dic 2015] → Val [ene 2016]
    Fold 2: Train [ene-dic 2015 + ene 2016] → Val [feb 2016]
    Fold 3: Train [ene-dic 2015 + ene-feb 2016] → Val [mar 2016]

    Parámetros:
        df:        DataFrame con columna de fecha
        n_folds:   número de folds (default: config.yaml)
        val_weeks: semanas de validación por fold
        date_col:  nombre de la columna de fecha

    Yields:
        tuple: (train_idx, val_idx, fold_info)
    """
    n_folds   = n_folds   or config['training']['n_folds']
    date_col  = date_col  or config['data']['date_col']

    dates     = df[date_col].sort_values().unique()
    n_dates   = len(dates)
    val_days  = val_weeks * 7

    # El test set (últimas 8 semanas) no se toca
    test_days   = config['training']['test_size_weeks'] * 7
    usable_end  = n_dates - test_days

    logger.info("Walk-forward splits configuración:")
    logger.info(f"  Folds:          {n_folds}")
    logger.info(f"  Val por fold:   {val_weeks} semanas")
    logger.info(f"  Test reservado: {config['training']['test_size_weeks']} semanas")

    # Calcular el inicio de cada fold de val
    # desde el final hacia atrás
    fold_ends = []
    for i in range(n_folds):
        end_idx = usable_end - (i * val_days)
        fold_ends.append(end_idx)
    fold_ends = sorted(fold_ends)

    for fold_num, val_end_idx in enumerate(fold_ends):
        val_start_idx = val_end_idx - val_days

        if val_start_idx <= 0:
            logger.warning(
                f"Fold {fold_num + 1}: no hay suficientes "
                f"datos de train, omitiendo..."
            )
            continue

        train_dates = dates[:val_start_idx]
        val_dates   = dates[val_start_idx:val_end_idx]

        train_mask  = df[date_col].isin(train_dates)
        val_mask    = df[date_col].isin(val_dates)

        fold_info = {
            'fold':        fold_num + 1,
            'train_start': pd.Timestamp(train_dates[0]),
            'train_end':   pd.Timestamp(train_dates[-1]),
            'val_start':   pd.Timestamp(val_dates[0]),
            'val_end':     pd.Timestamp(val_dates[-1]),
            'n_train':     train_mask.sum(),
            'n_val':       val_mask.sum()
        }

        logger.info(
            f"  Fold {fold_num + 1}: "
            f"Train [{train_dates[0]} → {train_dates[-1]}] "
            f"Val [{val_dates[0]} → {val_dates[-1]}]"
        )

        yield (
            df[train_mask].index,
            df[val_mask].index,
            fold_info
        )


# ─────────────────────────────────────────
# 2. Métricas de evaluación
# ─────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    in_log_scale: bool = True
) -> dict:
    """
    Calcula todas las métricas de evaluación.

    Parámetros:
        y_true:       valores reales
        y_pred:       valores predichos
        in_log_scale: si True, revierte log1p
                      antes de calcular métricas

    Retorna:
        dict con RMSE, MAE, MAPE, RMSLE
    """
    if in_log_scale:
        # Revertir log1p para métricas en escala real
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    # Clip para evitar predicciones negativas
    y_pred = np.clip(y_pred, 0, None)

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # MAPE — excluye ceros para evitar división por cero
    mask = y_true > 0
    mape = (
        np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100
        if mask.sum() > 0 else np.nan
    )

    # WAPE — Ideal para demanda intermitente y grandes volúmenes mezclados
    total_actual = np.sum(y_true)
    if total_actual > 0:
        wape = (np.sum(np.abs(y_true - y_pred)) / total_actual) * 100
    else:
        wape = np.nan

    # RMSLE — robusto con distribuciones sesgadas
    rmsle = np.sqrt(
        np.mean(
            (np.log1p(y_true) - np.log1p(y_pred)) ** 2
        )
    )

    return {
        'rmse':  round(float(rmse),  4),
        'mae':   round(float(mae),   4),
        'mape':  round(float(mape),  4),
        'rmsle': round(float(rmsle), 4),
        'wape':  round(float(wape),  4)
    }



# ─────────────────────────────────────────
# 3. Reporte de resultados
# ─────────────────────────────────────────
def summarize_validation(
    results: list[FoldResult]
) -> ValidationResult:
    """
    Calcula estadísticas finales de todos
    los folds de validación.
    """
    summary = ValidationResult(folds=results)

    for metric in ['rmse', 'mae', 'mape', 'rmsle','wape']:
        values = [getattr(r, metric) for r in results]
        setattr(summary, f'{metric}_mean', round(np.mean(values), 4))
        setattr(summary, f'{metric}_std',  round(np.std(values),  4))

    logger.info("\n" + "=" * 50)
    logger.info("RESULTADOS DE VALIDACIÓN")
    logger.info("=" * 50)
    logger.info(
        f"  RMSE:  {summary.rmse_mean:.4f} "
        f"(±{summary.rmse_std:.4f})"
    )
    logger.info(
        f"  MAE:   {summary.mae_mean:.4f} "
        f"(±{summary.mae_std:.4f})"
    )
    logger.info(
        f"  MAPE:  {summary.mape_mean:.4f}% "
        f"(±{summary.mape_std:.4f}%)"
    )
    logger.info(
        f"  RMSLE: {summary.rmsle_mean:.4f} "
        f"(±{summary.rmsle_std:.4f})"
    )
    logger.info(
        f"  WAPE: {summary.wape_mean:.4f} "
        f"(±{summary.wape_std:.4f})"
    )
    logger.info("=" * 50)

    return summary


# ─────────────────────────────────────────
# 4. Visualización de folds
# ─────────────────────────────────────────
def plot_folds(
    results: list[FoldResult],
    save_path: str = None
) -> None:
    """
    Visualiza los resultados por fold
    para detectar inestabilidad.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    metrics = ['rmse', 'mae', 'mape', 'rmsle','wape']
    colors  = ['steelblue', 'coral', 'seagreen', 'purple','orange']

    for ax, metric, color in zip(
        axes.flatten(), metrics, colors
    ):
        values = [getattr(r, metric) for r in results]
        folds  = [r.fold for r in results]

        ax.plot(folds, values,
                marker='o', color=color,
                linewidth=2, markersize=8)
        ax.axhline(
            np.mean(values),
            color='gray', linestyle='--',
            label=f'Media: {np.mean(values):.4f}'
        )
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.set_xticks(folds)

    plt.suptitle(
        'Métricas por Fold — Walk-Forward Validation',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info(f"Gráfica guardada: {save_path}")
        except Exception as e:
            logger.warning(f"No se pudo guardar la gráfica de folds: {e}")
            logger.info("Continuando con el entrenamiento del modelo final...")


    plt.close()