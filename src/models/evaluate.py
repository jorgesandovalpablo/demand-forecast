# src/models/evaluate.py
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import joblib
import matplotlib.pyplot as plt
#import shap
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import config
from src.models.validation import compute_metrics

logger = get_logger(__name__)


# ─────────────────────────────────────────
# 1. Preparar test set
# ─────────────────────────────────────────
def prepare_test_set(
    df: pd.DataFrame,
    feature_cols: list
) -> tuple:
    """
    Separa las últimas 8 semanas como test set.
    Estas semanas nunca fueron vistas por el modelo.

    Retorna:
        tuple: (X_test, y_test, test_df)
    """
    test_weeks = config['training']['test_size_weeks']
    cutoff     = (
        df['date'].max() -
        pd.Timedelta(weeks=test_weeks)
    )

    test_df = df[df['date'] > cutoff].copy()
    target  = config['data']['target']

    X_test  = test_df[feature_cols]
    y_test  = test_df[target]

    logger.info(
        f"Test set: {len(test_df):,} filas | "
        f"{test_df['date'].min()} → "
        f"{test_df['date'].max()}"
    )
    return X_test, y_test, test_df


# ─────────────────────────────────────────
# 2. Métricas globales
# ─────────────────────────────────────────
def evaluate_global(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int
) -> dict:
    """
    Calcula métricas globales sobre el test set
    y las loggea en MLflow.
    """
    metrics = compute_metrics(
        y_true, y_pred,
        in_log_scale=True
    )

    logger.info("\n" + "=" * 50)
    logger.info("MÉTRICAS GLOBALES — TEST SET")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k.upper()}: {v:.4f}")
    logger.info("=" * 50)

    # Loggear en MLflow con prefijo 'test_'
    for k, v in metrics.items():
        mlflow.log_metric(f"test_{k}", v)

    return metrics


# ─────────────────────────────────────────
# 3. Métricas por familia
# ─────────────────────────────────────────
def evaluate_by_family(
    test_df: pd.DataFrame,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Calcula métricas por familia de producto.
    Identifica qué categorías el modelo predice
    mejor y peor.
    """
    logger.info("Calculando métricas por familia...")
    target   = config['data']['target']
    test_df  = test_df.copy()
    test_df['y_pred'] = y_pred

    results = []
    for family, group in test_df.groupby('family'):
        metrics = compute_metrics(
            group[target].values,
            group['y_pred'].values,
            in_log_scale=True
        )
        metrics['family'] = family
        metrics['n_rows'] = len(group)
        results.append(metrics)

    family_metrics = (
        pd.DataFrame(results)
        .sort_values('rmse', ascending=False)
        .reset_index(drop=True)
    )

    logger.info("\nTop 5 familias con mayor RMSE:")
    logger.info(
        family_metrics[['family', 'rmse', 'mape','wape']]
        .head()
        .to_string(index=False)
    )

    return family_metrics


# ─────────────────────────────────────────
# 4. Métricas por tienda
# ─────────────────────────────────────────
def evaluate_by_store(
    test_df: pd.DataFrame,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Calcula métricas por tienda.
    """
    logger.info("Calculando métricas por tienda...")
    target  = config['data']['target']
    test_df = test_df.copy()
    test_df['y_pred'] = y_pred

    results = []
    for store, group in test_df.groupby('store_nbr'):
        metrics = compute_metrics(
            group[target].values,
            group['y_pred'].values,
            in_log_scale=True
        )
        metrics['store_nbr'] = store
        metrics['city']      = group['city'].iloc[0]
        metrics['n_rows']    = len(group)
        results.append(metrics)

    store_metrics = (
        pd.DataFrame(results)
        .sort_values('rmse', ascending=False)
        .reset_index(drop=True)
    )

    logger.info("\nTop 5 tiendas con mayor RMSE:")
    logger.info(
        store_metrics[
            ['store_nbr', 'city', 'rmse', 'mape']
        ]
        .head()
        .to_string(index=False)
    )

    return store_metrics


# ─────────────────────────────────────────
# 5. Análisis de errores temporales
# ─────────────────────────────────────────
def evaluate_by_time(
    test_df: pd.DataFrame,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Analiza cómo varía el error a lo largo
    del tiempo — identifica si el modelo
    se degrada con el tiempo.
    """
    logger.info("Analizando errores temporales...")
    target  = config['data']['target']
    test_df = test_df.copy()
    test_df['y_pred'] = y_pred
    test_df['error']  = np.abs(
        np.expm1(test_df[target]) -
        np.expm1(test_df['y_pred'])
    )

    time_metrics = (
        test_df.groupby('date')['error']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={
            'mean': 'mae_diario',
            'std':  'std_diario'
        })
    )

    return time_metrics


# ─────────────────────────────────────────
# 6. Feature importance
# ─────────────────────────────────────────
def evaluate_feature_importance(
    model: lgb.Booster,
    feature_cols: list,
    top_n: int = 30
) -> pd.DataFrame:
    """
    Analiza la importancia de cada feature
    usando gain — más robusto que split count.
    """
    logger.info("Calculando feature importance...")

    importance = pd.DataFrame({
        'feature':    feature_cols,
        'importance': model.feature_importance(
            importance_type='gain'
        )
    }).sort_values(
        'importance', ascending=False
    ).reset_index(drop=True)

    importance['importance_pct'] = (
        importance['importance'] /
        importance['importance'].sum() * 100
    ).round(2)

    logger.info(f"\nTop {top_n} features más importantes:")
    logger.info(
        importance.head(top_n)
        .to_string(index=False)
    )

    return importance


# ─────────────────────────────────────────
# 7. Visualizaciones
# ─────────────────────────────────────────
def plot_predictions(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    store_nbr: int = 1,
    family: str = 'GROCERY I',
    save_path: str = None
) -> None:
    """
    Visualiza predicciones vs reales para
    una tienda y familia específica.
    """
    target  = config['data']['target']
    mask    = (
        (test_df['store_nbr'] == store_nbr) &
        (test_df['family'] == family)
    )
    subset  = test_df[mask].copy()
    subset['y_pred'] = y_pred[mask.values]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Predicciones vs reales en escala log
    axes[0].plot(
        subset['date'],
        np.expm1(subset[target]),
        label='Real', color='steelblue', linewidth=2
    )
    axes[0].plot(
        subset['date'],
        np.expm1(subset['y_pred']),
        label='Predicho', color='coral',
        linewidth=2, linestyle='--'
    )
    axes[0].set_title(
        f'Predicciones vs Real — '
        f'Tienda {store_nbr} | {family}',
        fontweight='bold'
    )
    axes[0].legend()
    axes[0].set_ylabel('Ventas')

    # Error absoluto
    error = np.abs(
        np.expm1(subset[target]) -
        np.expm1(subset['y_pred'])
    )
    axes[1].bar(
        subset['date'], error,
        color='coral', alpha=0.7
    )
    axes[1].set_title('Error absoluto', fontweight='bold')
    axes[1].set_ylabel('Error')
    axes[1].set_xlabel('Fecha')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Gráfica guardada: {save_path}")
    plt.show()


def plot_feature_importance(
    importance: pd.DataFrame,
    top_n: int = 30,
    save_path: str = None
) -> None:
    """
    Visualiza las features más importantes.
    """
    top = importance.head(top_n)

    fig, ax = plt.subplots(figsize=(10, top_n * 0.35))
    ax.barh(
        top['feature'][::-1],
        top['importance_pct'][::-1],
        color='steelblue'
    )
    ax.set_title(
        f'Top {top_n} Features más importantes (Gain)',
        fontweight='bold'
    )
    ax.set_xlabel('Importancia (%)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Gráfica guardada: {save_path}")
    plt.show()


def plot_errors_by_family(
    family_metrics: pd.DataFrame,
    save_path: str = None
) -> None:
    """
    Visualiza RMSE por familia de producto.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors  = [
        'coral' if r > family_metrics['rmse'].mean()
        else 'steelblue'
        for r in family_metrics['rmse']
    ]
    ax.barh(
        family_metrics['family'][::-1],
        family_metrics['rmse'][::-1],
        color=colors[::-1]
    )
    ax.axvline(
        family_metrics['rmse'].mean(),
        color='gray', linestyle='--',
        label='RMSE promedio'
    )
    ax.set_title(
        'RMSE por familia de producto\n'
        '(rojo = sobre el promedio)',
        fontweight='bold'
    )
    ax.set_xlabel('RMSE')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Gráfica guardada: {save_path}")
    plt.show()


# ─────────────────────────────────────────
# Función principal — punto de entrada
# ─────────────────────────────────────────
def run_evaluation(horizon: int) -> dict:
    """
    Ejecuta la evaluación completa del modelo
    sobre el test set.

    Parámetros:
        horizon: 7 → modelo diario
                 30 → modelo mensual

    Retorna:
        dict con todas las métricas y análisis
    """
    logger.info("=" * 50)
    logger.info(
        f"Iniciando evaluación "
        f"{'DIARIA' if horizon == 7 else 'MENSUAL'}"
    )
    logger.info("=" * 50)

    # Cargar modelo y features
    model_path    = Path(f"models/lgbm_h{horizon}.pkl")
    features_path = (
        f"data/processed/"
        f"train_features_d{horizon}.parquet"
    )

    model = joblib.load(model_path)
    df    = pd.read_parquet(features_path)

    # Feature cols
    from src.models.train import get_feature_cols
    feature_cols = get_feature_cols(df)

    # Test set
    X_test, y_test, test_df = prepare_test_set(
        df, feature_cols
    )

    # Predicciones
    y_pred = model.predict(X_test)

    # Evaluaciones
    figures_path = Path("notebooks/figures")
    figures_path.mkdir(exist_ok=True)

    with mlflow.start_run(
        run_name=f"evaluation_h{horizon}"
    ):
        # Métricas globales
        global_metrics = evaluate_global(
            y_test.values, y_pred, horizon
        )

        # Por familia
        family_metrics = evaluate_by_family(
            test_df, y_pred
        )
        plot_errors_by_family(
            family_metrics,
            save_path=str(
                figures_path /
                f"errors_by_family_h{horizon}.png"
            )
        )

        # Por tienda
        store_metrics = evaluate_by_store(
            test_df, y_pred
        )

        # Temporal
        time_metrics = evaluate_by_time(
            test_df, y_pred
        )

        # Feature importance
        importance = evaluate_feature_importance(
            model, feature_cols
        )
        plot_feature_importance(
            importance,
            save_path=str(
                figures_path /
                f"feature_importance_h{horizon}.png"
            )
        )

        # Predicciones vs real (tienda 1, GROCERY I)
        plot_predictions(
            test_df, y_pred,
            store_nbr=1,
            family='GROCERY I',
            save_path=str(
                figures_path /
                f"predictions_h{horizon}.png"
            )
        )

        # Loggear artefactos
        for fig_path in figures_path.glob(
            f"*h{horizon}.png"
        ):
            mlflow.log_artifact(str(fig_path))

        # Guardar métricas por familia y tienda
        family_metrics.to_parquet(
            f"data/predictions/"
            f"family_metrics_h{horizon}.parquet",
            index=False
        )
        store_metrics.to_parquet(
            f"data/predictions/"
            f"store_metrics_h{horizon}.parquet",
            index=False
        )

    logger.info("=" * 50)
    logger.info(" Evaluación completada")
    logger.info("=" * 50)

    return {
        'global_metrics': global_metrics,
        'family_metrics': family_metrics,
        'store_metrics':  store_metrics,
        'time_metrics':   time_metrics,
        'importance':     importance
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
        help='Horizonte: 7 (diario) o 30 (mensual)'
    )
    args = parser.parse_args()

    run_evaluation(horizon=args.horizon)