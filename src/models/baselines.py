# src/models/baselines.py
"""
Baselines para comparación de valor ML.

Naive (último valor) y Seasonal Naive sobre el mismo test set
que evaluate.py. Script standalone — no integrado en el flujo principal.

Uso:
    python src/models/baselines.py                  # ambos horizontes
    python src/models/baselines.py --horizon 7      # solo diario
    python src/models/baselines.py --horizon 30     # solo mensual
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import config
from src.models.validation import compute_metrics

logger = get_logger(__name__)


# ─────────────────────────────────────────
# 1. Preparar test set (misma lógica que evaluate.py)
# ─────────────────────────────────────────
def prepare_test_set(df: pd.DataFrame) -> tuple:
    """
    Separa las últimas 8 semanas como test set.
    Mismos parámetros que evaluate.py.
    """
    test_weeks = config['training']['test_size_weeks']
    target = config['data']['target']
    group_cols = config['data']['group_cols']

    cutoff = df['date'].max() - pd.Timedelta(weeks=test_weeks)
    test_df = df[df['date'] > cutoff].copy()
    train_df = df[df['date'] <= cutoff].copy()

    logger.info(
        f"Test set: {len(test_df):,} filas | "
        f"Train set: {len(train_df):,} filas | "
        f"Cutoff: {cutoff.date()}"
    )

    return train_df, test_df, target, group_cols


# ─────────────────────────────────────────
# 2. Naive: último valor conocido
# ─────────────────────────────────────────
def naive_last_value(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    group_cols: list
) -> np.ndarray:
    """
    Para cada serie (store, family), la predicción es
    el último valor observado en el train set.
    """
    last_values = (
        train_df
        .groupby(group_cols, observed=False)[target]
        .last()
        .reset_index()
        .rename(columns={target: 'y_pred'})
    )

    merged = test_df[group_cols].merge(
        last_values, on=group_cols, how='left'
    )

    return merged['y_pred'].values


# ─────────────────────────────────────────
# 3. Seasonal Naive (vectorizado)
# ─────────────────────────────────────────
def seasonal_naive(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    group_cols: list,
    horizon: int
) -> np.ndarray:
    """
    Seasonal Naive adaptado por horizonte:
    - h7: predicción = valor del mismo weekday de la semana anterior
    - h30: predicción = valor del mismo mes del año anterior

    Implementación vectorizada con merge.
    """
    test = test_df.copy()

    if horizon == 7:
        test['ref_date'] = test['date'] - pd.Timedelta(days=7)
    else:
        test['ref_date'] = test['date'] - pd.Timedelta(days=365)

    # Merge train con test usando ref_date
    merged = test.merge(
        train_df[group_cols + ['date', target]],
        left_on=group_cols + ['ref_date'],
        right_on=group_cols + ['date'],
        how='left',
        suffixes=('_test', '_pred')
    )

    pred_col = f'{target}_pred'

    # Fallback: último valor de la serie si no hay match
    last_values = (
        train_df
        .groupby(group_cols, observed=False)[target]
        .last()
        .reset_index()
        .rename(columns={target: '_fallback'})
    )

    merged = merged.merge(
        last_values,
        on=group_cols,
        how='left'
    )

    merged[pred_col] = merged[pred_col].fillna(merged['_fallback'])
    merged[pred_col] = merged[pred_col].fillna(0.0)

    return merged[pred_col].values


# ─────────────────────────────────────────
# 4. Ejecutar baselines
# ─────────────────────────────────────────
def run_baselines(horizon: int) -> dict:
    """
    Ejecuta Naive y Seasonal Naive para un horizonte dado.
    Retorna dict con métricas de cada baseline.
    """
    processed_path = Path("data/processed/train_processed.parquet")

    if not processed_path.exists():
        raise FileNotFoundError(
            "No se encontró train_processed.parquet. "
            "Ejecuta primero: python src/data/preprocessing.py"
        )

    logger.info("=" * 50)
    logger.info(
        f"BASELINES — "
        f"{'DIARIO' if horizon == 7 else 'MENSUAL'} (h={horizon})"
    )
    logger.info("=" * 50)

    df = pd.read_parquet(processed_path)
    train_df, test_df, target, group_cols = prepare_test_set(df)

    y_true = test_df[target].values

    # Naive
    logger.info("Ejecutando Naive (último valor)...")
    y_naive = naive_last_value(train_df, test_df, target, group_cols)
    metrics_naive = compute_metrics(y_true, y_naive, in_log_scale=True)

    # Seasonal Naive
    logger.info(
        f"Ejecutando Seasonal Naive "
        f"({'7d' if horizon == 7 else '30d'})..."
    )
    y_snaive = seasonal_naive(
        train_df, test_df, target, group_cols, horizon
    )
    metrics_snaive = compute_metrics(y_true, y_snaive, in_log_scale=True)

    logger.info("=" * 50)
    logger.info("Resultados:")
    logger.info(
        f"  Naive:       WAPE={metrics_naive['wape']:.2f}% "
        f"MAE={metrics_naive['mae']:.2f}"
    )
    logger.info(
        f"  SNaive:      WAPE={metrics_snaive['wape']:.2f}% "
        f"MAE={metrics_snaive['mae']:.2f}"
    )
    logger.info(f"{'=' * 50}")

    return {
        'horizon': horizon,
        'naive': metrics_naive,
        'snaive': metrics_snaive,
    }


# ─────────────────────────────────────────
# 5. Generar reporte
# ─────────────────────────────────────────
def generate_report(results: list[dict]) -> str:
    """
    Genera markdown con tabla comparativa.
    """
    # Métricas ML del README (producción)
    ml_metrics = {
        7: {'wape': 10.51, 'mae': 49.73, 'rmse': 194.35},
        30: {'wape': 12.34, 'mae': 58.41, 'rmse': 218.19},
    }

    lines = [
        "# Baselines: Comparación Naive vs ML",
        "",
        "Comparación del modelo LightGBM contra dos baselines triviales",
        "para demostrar valor añadido del ML.",
        "",
        "---",
        "",
    ]

    for r in results:
        h = r['horizon']
        label = "Diario (h=7)" if h == 7 else "Mensual (h=30)"
        ml = ml_metrics[h]

        lines.extend([
            f"## Modelo {label}",
            "",
            "| Modelo | WAPE | MAE | RMSE |",
            "|---|---|---|---|",
            f"| Naive (último valor) | "
            f"{r['naive']['wape']:.2f}% | "
            f"{r['naive']['mae']:.2f} | "
            f"{r['naive']['rmse']:.2f} |",
            f"| Seasonal Naive | "
            f"{r['snaive']['wape']:.2f}% | "
            f"{r['snaive']['mae']:.2f} | "
            f"{r['snaive']['rmse']:.2f} |",
            f"| **LightGBM (Optuna)** | "
            f"**{ml['wape']:.2f}%** | "
            f"**{ml['mae']:.2f}** | "
            f"**{ml['rmse']:.2f}** |",
            "",
        ])

    # Resumen
    lines.extend([
        "---",
        "",
        "## Resumen de mejora relativa",
        "",
    ])

    for r in results:
        h = r['horizon']
        label = "h7" if h == 7 else "h30"
        ml = ml_metrics[h]

        wape_naive = r['naive']['wape']
        wape_snaive = r['snaive']['wape']
        wape_ml = ml['wape']

        if wape_naive > 0:
            gain_vs_naive = ((wape_naive - wape_ml) / wape_naive) * 100
        else:
            gain_vs_naive = 0

        if wape_snaive > 0:
            gain_vs_snaive = ((wape_snaive - wape_ml) / wape_snaive) * 100
        else:
            gain_vs_snaive = 0

        lines.extend([
            f"### {label}",
            f"- ML vs Naive: **{gain_vs_naive:+.1f}%** en WAPE "
            f"({wape_naive:.2f}% → {wape_ml:.2f}%)",
            f"- ML vs SNaive: **{gain_vs_snaive:+.1f}%** en WAPE "
            f"({wape_snaive:.2f}% → {wape_ml:.2f}%)",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*Generado por `src/models/baselines.py`*",
    ])

    return "\n".join(lines)


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Baselines Naive y Seasonal Naive"
    )
    parser.add_argument(
        '--horizon',
        type=int,
        choices=[7, 30],
        default=None,
        help='Horizonte: 7 (diario) o 30 (mensual). '
             'Por defecto: ambos.'
    )
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else [7, 30]
    results = []

    for h in horizons:
        r = run_baselines(h)
        results.append(r)

    report = generate_report(results)

    report_path = Path("reports/baselines.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    logger.info(f"\nReporte guardado en: {report_path}")
    print(f"\n{report}")
