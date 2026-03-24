#! /bin/bash
echo "Iniciando Proceso..."
mkdir -p data/{raw,processed,predictions}
mkdir -p src/{data,features,models,api,utils}
mkdir -p tests
mkdir -p notebooks
mkdir -p configs
mkdir -p mlflow
mkdir -p .github/workflows

echo "Directorios creados..."
#crear inits
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/api/__init__.py
touch src/utils/__init__.py
touch src/__init__.py

echo "inits creados..."

touch src/data/ingestion.py
touch src/data/preprocessing.py
touch src/features/build_features.py
touch src/models/train.py
touch src/models/evaluate.py
touch src/models/predict.py
touch src/models/retrain.py
touch src/models/validation.py

touch src/api/main.py
touch src/api/schemas.py
touch src/utils/seed.py
touch src/utils/logger.py
touch configs/config.yaml
touch tests/test_features.py
touch tests/test_api.py
touch .github/workflows/ci.yml
touch Dockerfile
touch .env.example

echo "archivos base creados y proceso finalizado"