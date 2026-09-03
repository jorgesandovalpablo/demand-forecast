#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# sync_hf_space.sh — Prepara y empuja el subset curado al HuggingFace Space.
#
# El Space es un repositorio independiente. Este script copia solo lo que la
# demo necesita (código + artefactos) y versiona los binarios con git LFS.
#
# Uso:
#   HF_SPACE_DIR=/ruta/al/repo-del-space bash scripts/sync_hf_space.sh
#
# Variables:
#   HF_SPACE_DIR  (obligatoria)  Directorio local del Space clonado (git).
#   COMMIT_MSG    (opcional)     Mensaje de commit. Default descriptivo.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_DIR="${HF_SPACE_DIR:-}"

if [[ -z "$SPACE_DIR" ]]; then
    echo "ERROR: Define HF_SPACE_DIR apuntando al repo local del Space."
    echo '  Ej: HF_SPACE_DIR=~/demand-forecast-demo bash scripts/sync_hf_space.sh'
    exit 1
fi

if [[ ! -d "$SPACE_DIR/.git" ]]; then
    echo "ERROR: $SPACE_DIR no es un repositorio git."
    exit 1
fi

echo "→ Sincronizando desde $ROOT_DIR hacia $SPACE_DIR"

# ── Código fuente ────────────────────────────────────────────────────────────
mkdir -p "$SPACE_DIR"/{src,dashboard,configs,scripts}
cp -R "$ROOT_DIR/src/."         "$SPACE_DIR/src/"
cp -R "$ROOT_DIR/dashboard/."   "$SPACE_DIR/dashboard/"
cp -R "$ROOT_DIR/configs/."     "$SPACE_DIR/configs/"
find "$SPACE_DIR" -name "*egg-info" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# ── Artefactos: modelos ──────────────────────────────────────────────────────
mkdir -p "$SPACE_DIR/models"
for f in \
    lgbm_h7.pkl lgbm_h30.pkl \
    features_h7.pkl features_h30.pkl \
    feature_pipeline_h7.pkl feature_pipeline_h30.pkl; do
    if [[ -f "$ROOT_DIR/models/$f" ]]; then
        cp "$ROOT_DIR/models/$f" "$SPACE_DIR/models/$f"
    else
        echo "  ⚠ falta models/$f"
    fi
done

# ── Artefactos: datos ────────────────────────────────────────────────────────
mkdir -p "$SPACE_DIR/data/raw" "$SPACE_DIR/data/processed" "$SPACE_DIR/data/predictions"
cp "$ROOT_DIR/data/processed/train_processed.parquet" "$SPACE_DIR/data/processed/" 2>/dev/null \
    || echo "  ⚠ falta train_processed.parquet"

for f in stores.csv oil.csv holidays_events.csv transactions.csv; do
    cp "$ROOT_DIR/data/raw/$f" "$SPACE_DIR/data/raw/" 2>/dev/null \
        || echo "  ⚠ falta data/raw/$f"
done

for h in 7 30; do
    for kind in family_metrics_h${h} global_metrics_h${h}; do
        cp "$ROOT_DIR/data/predictions/${kind}.parquet" \
           "$SPACE_DIR/data/predictions/" 2>/dev/null \
            || echo "  ⚠ falta ${kind}.parquet"
    done
done

# ── Configuración del Space (app raíz + deps + gitignore) ───────────────────
cp "$ROOT_DIR/dashboard/app.py" "$SPACE_DIR/app.py"
cp "$ROOT_DIR/dashboard/requirements.txt" "$SPACE_DIR/requirements.txt"
cat > "$SPACE_DIR/.gitignore" <<'EOF'
__pycache__/
*.pyc
.venv/
venv/
logs/
mlflow/
mlruns/
EOF

# ── git LFS: binarios grandes ────────────────────────────────────────────────
cd "$SPACE_DIR"
git lfs track 'models/*.pkl' 'data/**/*.parquet' 'data/raw/*.csv'
git add .gitattributes

git add -A
git status --short | head -30

COMMIT_MSG="${COMMIT_MSG:-feat: sincronizar dashboard + artefactos al Space}"
git commit -m "$COMMIT_MSG" || echo "  (sin cambios para commit)"

echo ""
echo "✅ Listo. Revisa el diff y empuja con:"
echo "  cd $SPACE_DIR && git push"