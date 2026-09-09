#!/usr/bin/env python
# scripts/build_demo_bundle.py
"""Materializa assets para la demo de Streamlit Community Cloud.

Pre-calcula predicciones con predict() (paridad real con el pipeline de
entrenamiento) para cada tienda x horizonte y las escribe como parquet
en deploy/assets/predictions/. Tambien genera stores.json, families.json,
copia backtest + metricas.
"""

import json
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "deploy" / "assets"
PRED_DIR = ASSETS / "predictions"
HORIZONS = [7, 30]
METRICS_PATTERN = "data/predictions/global_metrics_h{h}.parquet"
BACKTEST_PATTERN = "data/predictions/backtest_predictions_h{h}.parquet"


def _copy_backtest_and_metrics() -> None:
    """Copia backtest_predictions y global_metrics por horizonte."""
    for h in HORIZONS:
        for pattern in [BACKTEST_PATTERN, METRICS_PATTERN]:
            src = ROOT / pattern.format(h=h)
            if src.exists():
                dst = ASSETS / src.name
                shutil.copy2(src, dst)
                print(f"  [data] {src.name}")
            else:
                print(f"  [data] SKIP {src.name} (no existe)")


def _generate_metadata(
    historical: pd.DataFrame, pipeline
) -> None:
    """Genera stores.json y families.json."""
    stores = sorted(historical["store_nbr"].unique().tolist())
    with open(ASSETS / "stores.json", "w") as f:
        json.dump(stores, f)
    print(f"  [meta] stores.json ({len(stores)} tiendas)")

    cats = pipeline.categories_mapping.get("family", [])
    families = {i: str(name) for i, name in enumerate(cats)}
    with open(ASSETS / "families.json", "w") as f:
        json.dump(families, f, ensure_ascii=False, indent=2)
    print(f"  [meta] families.json ({len(families)} familias)")


def build() -> None:
    """Ejecuta la materializacion completa del bundle."""
    print("=== Build demo bundle ===")

    from src.models.predict import predict, ModelRegistry

    # Cargar historial procesado
    proc_path = ROOT / "data" / "processed" / "train_processed.parquet"
    if not proc_path.exists():
        sys.exit(f"ERROR: {proc_path} no encontrado")
    historical = pd.read_parquet(proc_path)

    # Cargar pipeline para metadata de familias
    pipeline_path = ROOT / "models" / "feature_pipeline_h7.pkl"
    if not pipeline_path.exists():
        sys.exit(f"ERROR: {pipeline_path} no encontrado")
    pipeline = joblib.load(pipeline_path)

    stores = sorted(historical["store_nbr"].unique().tolist())
    print(f"Historial: {len(historical):,} filas, {len(stores)} tiendas")

    # Pre-cargar modelos
    for h in HORIZONS:
        ModelRegistry.load(h)
        print(f"Modelo h={h} cargado")

    # Crear directorios
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Generar metadata
    print("\nGenerando metadata...")
    _generate_metadata(historical, pipeline)

    # Pre-calcula predicciones: UNA llamada por horizonte (todas las tiendas)
    # predict_by_store() por tienda provoca test DataFrame vacio en
    # prepare_prediction_data → holiday_impact_type sin categorías →
    # categorical_feature mismatch con LightGBM. predict() con el
    # historial completo evita este problema.
    print("\nPre-calculando predicciones...")
    for h in HORIZONS:
        print(f"\n--- Horizonte {h} ---")
        all_preds = predict(historical, h)
        for store_nbr in stores:
            store_pred = all_preds[all_preds["store_nbr"] == store_nbr]
            fname = f"store_{store_nbr}_h{h}.parquet"
            store_pred.to_parquet(PRED_DIR / fname, index=False)
        print(
            f"  h={h}: {len(all_preds):,} filas "
            f"({len(stores)} tiendas)"
        )

    # Copiar backtest y metricas
    print("\nCopiando backtest y metricas...")
    _copy_backtest_and_metrics()

    # Resumen de tamano
    total_bytes = sum(f.stat().st_size for f in ASSETS.rglob("*") if f.is_file())
    print(f"\nBundle generado en deploy/assets/ ({total_bytes / 1024 / 1024:.1f} MB)")
    print("Listo para deploy en Streamlit Community Cloud.")


if __name__ == "__main__":
    build()
