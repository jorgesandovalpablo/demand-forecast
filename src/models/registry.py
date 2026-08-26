"""
Integración con el MLflow Model Registry (DagsHub).

Fuente de verdad de los modelos productivos. Estrategia
"local primero": los artefactos en models/ tienen prioridad;
el registry actúa como respaldo versionado y mecanismo de
recuperación (ej. runners efímeros de CI o máquinas nuevas).

Convención:
    - Un modelo registrado por horizonte:
        7 → demand-forecast-daily
       30 → demand-forecast-monthly
    - Alias 'production' sobre la versión vigente.
      Rollback = reasignar el alias a una versión anterior.
"""
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.utils.logger import get_logger

logger = get_logger(__name__)

PRODUCTION_ALIAS = "production"
ARTIFACT_SUBDIR = "model"

REGISTRY_MODELS: dict[int, str] = {
    7: "demand-forecast-daily",
    30: "demand-forecast-monthly",
}


def _artifact_paths(horizon: int, models_dir: str = "models") -> list[Path]:
    """Rutas locales de los 3 artefactos obligatorios de un horizonte."""
    return [
        Path(models_dir) / f"lgbm_h{horizon}.pkl",
        Path(models_dir) / f"features_h{horizon}.pkl",
        Path(models_dir) / f"feature_pipeline_h{horizon}.pkl",
    ]


def get_model_name(horizon: int) -> str:
    """Nombre del modelo registrado para un horizonte."""
    if horizon not in REGISTRY_MODELS:
        raise ValueError(f"Horizonte no soportado por el registry: {horizon}")
    return REGISTRY_MODELS[horizon]


def promote_local_artifacts(
    horizon: int,
    metrics: dict | None = None,
    models_dir: str = "models"
) -> str | None:
    """
    Versiona los artefactos de producción local en el registry.

    Loguea los 3 artefactos en un run dedicado, registra una nueva
    versión del modelo del horizonte y mueve el alias 'production'
    a esa versión. Si el registro falla se retorna None y la
    promoción local previa queda intacta (el serving sigue operativo).

    Args:
        horizon: horizonte del modelo promovido (7 o 30).
        metrics: métricas opcionales para tags de la versión.
        models_dir: directorio local de artefactos.

    Returns:
        Descripción de la versión promovida o None si falló.
    """
    from src.models.train import setup_mlflow

    paths = _artifact_paths(horizon, models_dir)
    missing = [p for p in paths if not p.exists()]
    if missing:
        logger.warning(
            f"No se puede registrar: faltan artefactos locales {missing}"
        )
        return None

    name = get_model_name(horizon)
    try:
        setup_mlflow()
        with mlflow.start_run(
            run_name=f"registry_promotion_h{horizon}_"
                     f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            run_id = mlflow.active_run().info.run_id
            for path in paths:
                mlflow.log_artifact(str(path), ARTIFACT_SUBDIR)
            if metrics:
                clean = {
                    k: float(v) for k, v in metrics.items()
                    if isinstance(v, (int, float))
                }
                mlflow.log_metrics(clean)

        client = MlflowClient()
        run = client.get_run(run_id)
        source = (
            f"{run.info.artifact_uri}/{ARTIFACT_SUBDIR}"
            f"/lgbm_h{horizon}.pkl"
        )
        version = client.create_model_version(
            name=name, source=source, run_id=run_id
        )
        client.set_registered_model_alias(
            name, PRODUCTION_ALIAS, version.version
        )

        tags = {
            "horizon": str(horizon),
            "promoted_at": datetime.now().isoformat(),
        }
        if metrics:
            for key in ("mae", "rmse", "wape"):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    tags[f"test_{key}"] = f"{value:.4f}"
        for tag_key, tag_value in tags.items():
            client.set_model_version_tag(
                name, version.version, tag_key, tag_value
            )

        promoted = f"{name}@{PRODUCTION_ALIAS} (v{version.version})"
        logger.info(f"Modelo registrado en el registry: {promoted}")
        return promoted

    except MlflowException as exc:
        logger.warning(
            f"Fallo el registro en MLflow ({exc}); la promoción "
            f"local se mantiene operativa."
        )
        return None


def ensure_local_artifacts(horizon: int, models_dir: str = "models") -> bool:
    """
    Garantiza que existan los artefactos locales de un horizonte.

    Si falta alguno, descarga la versión con alias 'production'
    desde el registry y la cachea en models_dir. Es un no-op cuando
    todo ya está en disco.

    Returns:
        True si los 3 artefactos están disponibles tras la llamada.
    """
    paths = _artifact_paths(horizon, models_dir)
    if all(p.exists() for p in paths):
        return True

    from src.models.train import setup_mlflow

    name = get_model_name(horizon)
    try:
        setup_mlflow()
        client = MlflowClient()
        version = client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    except MlflowException as exc:
        logger.error(
            f"No hay versión '{PRODUCTION_ALIAS}' de '{name}' en el "
            f"registry ({exc}). Ejecuta train.py --horizon {horizon} "
            f"localmente."
        )
        return False

    try:
        source_dir = Path(
            mlflow.artifacts.download_artifacts(
                run_id=version.run_id,
                artifact_path=ARTIFACT_SUBDIR,
            )
        )
    except MlflowException as exc:
        logger.error(f"Descarga desde el registry falló: {exc}")
        return False

    downloaded = Path(source_dir)
    target_dir = Path(models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        origin = downloaded / path.name
        if not origin.exists():
            logger.warning(
                f"Artefacto ausente en la versión registrada: {path.name}"
            )
            continue
        destination = target_dir / path.name
        destination.write_bytes(origin.read_bytes())
        logger.info(f"Artefacto recuperado del registry: {destination}")

    return all(p.exists() for p in paths)


def rollback_production(horizon: int, version: int | str) -> None:
    """
    Reasigna el alias 'production' a una versión anterior.

    Útil para revertir una promoción defectuosa sin reentrenar.
    """
    name = get_model_name(horizon)
    client = MlflowClient()
    client.set_registered_model_alias(name, PRODUCTION_ALIAS, str(version))
    logger.info(f"Rollback aplicado: {name}@{PRODUCTION_ALIAS} → v{version}")
