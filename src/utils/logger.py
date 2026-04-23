import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Crea y retorna un logger configurado para el módulo
    que lo solicita.

    Parámetros:
        name: nombre del módulo, usualmente __name__

    Retorna:
        logger configurado con handlers de consola y archivo

    Uso:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Cargando datos...")
        logger.warning("Nulos detectados")
        logger.error("Archivo no encontrado")
    """

    # Crear carpeta de logs si no existe
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Nombre del archivo de log con fecha
    log_file = logs_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

    # Crear logger
    logger = logging.getLogger(name)

    # Evitar duplicar handlers si el logger ya existe
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Formato del mensaje
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler 1 — Consola (INFO y superior)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Handler 2 — Archivo (DEBUG y superior)
    # Guarda todo, incluyendo mensajes de debug detallados
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

