# src/models/shap_analysis.py
"""
Análisis de SHAP values para interpretabilidad de features.

Uso:
    python src/models/shap_analysis.py --horizon 7
    python src/models/shap_analysis.py --horizon 30 --sample 50000
"""
import argparse
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config import config
from src.models.predict import ModelRegistry
from src.features.build_features import DemandFeatureEngineer

logger = get_logger(__name__)


# ─────────────────────────────────────────
# 1. Cargar datos y modelo
# ─────────────────────────────────────────
def load_data_and_model(
    horizon: int,
) -> tuple:
    """
    Carga modelo, pipeline, features y datos procesados.

    Retorna: (model, X, feature_cols, df_full)
    """
    pipeline_path = Path(
        f"models/feature_pipeline_h{horizon}.pkl"
    )
    features_path = Path(
        f"models/features_h{horizon}.pkl"
    )
    processed_path = Path(
        "data/processed/train_processed.parquet"
    )

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline no encontrado: {pipeline_path}"
        )
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features no encontradas: {features_path}"
        )
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Datos procesados no encontrados: {processed_path}"
        )

    logger.info(
        f"Cargando modelo y pipeline para horizon={horizon}..."
    )
    model = ModelRegistry.load(horizon)
    pipeline = DemandFeatureEngineer.load(pipeline_path)
    feature_cols = joblib.load(features_path)

    logger.info("Cargando y transformando datos procesados...")
    df_processed = pd.read_parquet(processed_path)
    df = pipeline.transform(df_processed, is_train=True)

    X = df[feature_cols].copy()

    logger.info(
        f"Datos cargados: {X.shape[0]} filas, "
        f"{X.shape[1]} features"
    )
    return model, X, feature_cols, df


# ─────────────────────────────────────────
# 2. Calcular SHAP values
# ─────────────────────────────────────────
def compute_shap_values(
    model,
    X: pd.DataFrame,
    n_sample: int = 30000,
) -> pd.DataFrame:
    """
    Calcula SHAP values usando TreeExplainer.

    Retorna DataFrame con columns:
    [feature, shap_mean, shap_ratio]
    """
    # Subsampleo si hay más filas que n_sample
    if len(X) > n_sample:
        logger.info(
            f"Subsampleando {n_sample} filas "
            f"de {len(X)} totales..."
        )
        X_sample = X.sample(
            n=n_sample, random_state=config['project']['seed']
        )
    else:
        X_sample = X

    logger.info(
        f"Calculando SHAP values con "
        f"{len(X_sample)} muestras..."
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": X.columns,
        "shap_mean": mean_abs_shap
    }).sort_values("shap_mean", ascending=False)

    shap_df["shap_ratio"] = (
        shap_df["shap_mean"] / shap_df["shap_mean"].max()
    )

    logger.info(
        f"SHAP completado. "
        f"Top feature: {shap_df.iloc[0]['feature']} "
        f"({shap_df.iloc[0]['shap_mean']:.4f})"
    )
    return shap_df


# ─────────────────────────────────────────
# 3. Clasificar features
# ─────────────────────────────────────────
def classify_features(
    shap_df: pd.DataFrame,
    low_percentile: int = 20,
    min_ratio: float = 0.05,
) -> pd.DataFrame:
    """
    Clasifica features en KEEP / DROP_CANDIDATE.

    Criterio: percentil bajo + ratio mínimo de SHAP.
    """
    threshold = np.percentile(
        shap_df["shap_mean"], low_percentile
    )

    shap_df["decision"] = "KEEP"
    shap_df.loc[
        (shap_df["shap_mean"] < threshold) &
        (shap_df["shap_ratio"] < min_ratio),
        "decision"
    ] = "DROP_CANDIDATE"

    n_keep = (shap_df["decision"] == "KEEP").sum()
    n_drop = (shap_df["decision"] == "DROP_CANDIDATE").sum()
    logger.info(
        f"Clasificación: {n_keep} KEEP, {n_drop} DROP_CANDIDATE"
    )
    return shap_df


# ─────────────────────────────────────────
# 4. Guardar resultados
# ─────────────────────────────────────────
def save_results(
    shap_df: pd.DataFrame,
    horizon: int,
    output_dir: Path,
) -> None:
    """Guarda CSV resumen, listas keep/drop y gráfico."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV resumen
    shap_df.to_csv(
        output_dir / f"shap_summary_h{horizon}.csv",
        index=False
    )

    # Listas keep/drop
    shap_df.query("decision == 'KEEP'")["feature"].to_csv(
        output_dir / f"features_keep_h{horizon}.txt",
        index=False, header=False
    )
    shap_df.query("decision == 'DROP_CANDIDATE'")["feature"].to_csv(
        output_dir / f"features_drop_h{horizon}.txt",
        index=False, header=False
    )

    # Gráfico top-30
    top30 = shap_df.head(30)
    plt.figure(figsize=(10, 12))
    plt.barh(
        top30["feature"][::-1],
        top30["shap_mean"][::-1]
    )
    plt.title(f"Top 30 SHAP Features | Horizon {horizon}")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"shap_top30_h{horizon}.png",
        dpi=150
    )
    plt.close()

    logger.info(
        f"Resultados guardados en {output_dir}/"
    )


# ─────────────────────────────────────────
# 5. Función principal
# ─────────────────────────────────────────
def run_shap_analysis(
    horizon: int,
    n_sample: int = 30000,
    output_dir: str = "reports/shap",
) -> pd.DataFrame:
    """
    Ejecuta el análisis completo de SHAP.

    Retorna DataFrame con ranking de features.
    """
    logger.info("=" * 50)
    logger.info(f"SHAP Analysis — Horizon {horizon}")
    logger.info("=" * 50)

    model, X, feature_cols, df = load_data_and_model(horizon)

    # Filtrar ventana temporal: últimos 6 meses antes del cutoff
    cutoff = df['date'].max() - pd.Timedelta(weeks=8)
    mask = (
        (df['date'] < cutoff) &
        (df['date'] >= cutoff - pd.Timedelta(days=180))
    )
    X_window = X[mask].copy()
    logger.info(
        f"Ventana SHAP: {mask.sum()} filas "
        f"(últimos 6 meses antes del cutoff)"
    )

    shap_df = compute_shap_values(model, X_window, n_sample)
    shap_df = classify_features(shap_df)

    save_results(
        shap_df, horizon, Path(output_dir)
    )

    logger.info("=" * 50)
    logger.info("SHAP Analysis completado")
    logger.info("=" * 50)

    return shap_df


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Análisis SHAP de features"
    )
    parser.add_argument(
        '--horizon',
        type=int,
        choices=[7, 30],
        required=True,
        help='Horizonte: 7 (diario) o 30 (mensual)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=30000,
        help='Número de muestras para SHAP (default: 30000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/shap',
        help='Directorio de salida (default: reports/shap)'
    )
    args = parser.parse_args()

    run_shap_analysis(
        horizon=args.horizon,
        n_sample=args.sample,
        output_dir=args.output_dir,
    )
