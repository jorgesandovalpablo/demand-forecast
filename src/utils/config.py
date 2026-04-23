from pathlib import Path
import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    """
    Carga el archivo de configuración del proyecto.

    Parámetros:
        path: ruta al archivo config.yaml

    Retorna:
        dict con toda la configuración del proyecto

    Raises:
        FileNotFoundError: si el archivo no existe
        yaml.YAMLError: si el archivo tiene errores de sintaxis
    """
    config_path = Path(path)

    if not config_path.exists():
        logger.error(f"Archivo de configuración no encontrado: {path}")
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {path}"
        )

    try:
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuración cargada desde: {path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error de sintaxis en config.yaml: {e}")
        raise


# Instancia global — se carga una sola vez
# y se reutiliza en todos los módulos
config = load_config()