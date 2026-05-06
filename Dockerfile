# ─────────────────────────────────────────
# Etapa 1 — Builder
# Instala dependencias en una imagen temporal
# ─────────────────────────────────────────
FROM python:3.10.12-slim AS builder

# Evita que Python genere archivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1
# Evita buffering en logs — los ves en tiempo real
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema
# necesarias para compilar algunas librerías
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo requirements primero
# Docker cachea esta capa — si no cambia
# requirements.txt no reinstala todo
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────
# Etapa 2 — Runtime
# Imagen final ligera solo con lo necesario
# ─────────────────────────────────────────
FROM python:3.10.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Usuario no-root por seguridad
# Nunca corras contenedores como root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copiar librerías instaladas del builder
COPY --from=builder /usr/local/lib/python3.10/site-packages \
                    /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin \
                    /usr/local/bin

# Copiar código fuente
COPY src/       ./src/
COPY configs/   ./configs/
COPY models/    ./models/
COPY pyproject.toml .

# Copiar variables de entorno de ejemplo
COPY .env.example .env

# Crear carpetas necesarias
RUN mkdir -p data/processed \
             data/predictions \
             logs \
    && chown -R appuser:appuser /app

# Cambiar a usuario no-root
USER appuser

# Puerto que expone la API
EXPOSE 8000

# Health check — Docker verifica que
# el contenedor está funcionando
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=40s \
            --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para iniciar la API
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]