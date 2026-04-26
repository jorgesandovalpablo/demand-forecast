# test_validation.py (bórralo después)
import pandas as pd
import numpy as np

from src.models.validation import (
    walk_forward_splits,
    compute_metrics,
    FoldResult
)

# Cargar features ya procesadas
train = pd.read_parquet('data/processed/train_features_d7.parquet')

print("🔍 Probando walk-forward splits:")
folds = list(walk_forward_splits(train, n_folds=3))
print(f"  Folds generados: {len(folds)}")

# Probar métricas

y_true = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 18, 33, 38, 52])

metrics = compute_metrics(y_true, y_pred, in_log_scale=False)
print(f"\n✅ Métricas de prueba:")
for k, v in metrics.items():
    print(f"  {k}: {v}")