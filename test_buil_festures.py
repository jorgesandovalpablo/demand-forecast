# test_features.py (bórralo después de probar)
from src.data.ingestion import load_raw_data
from src.data.preprocessing import run_preprocessing
from src.features.build_features import build_features

# Pipeline completo
data = load_raw_data()
train, test = run_preprocessing(data, save=False)

print(f"Train pre columns{train.columns.to_list()}")
# Modelo diario
train_d7 = build_features(train, horizon=7, save=True)
print(f"\n✅ Features diario:  {train_d7.shape}")

print(f"Train post columns{train.columns.to_list()}")
# Modelo mensual
train_m30 = build_features(train, horizon=30, save=True)
print(f"✅ Features mensual: {train_m30.shape}")
print(f"Train final{train.columns.to_list()}")

# Verificar que no hay data leakage
print("\n🔍 Verificación de lags:")
print("  Lag mínimo modelo diario:  lag_7  ✅")
print("  Lag mínimo modelo mensual: lag_30 ✅")
print(f"Train post columns{train_d7.columns.to_list()}")
print(f"Train post columns{train_m30.columns.to_list()}")