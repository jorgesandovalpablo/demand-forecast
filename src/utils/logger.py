import logging 
import sys
from pathlib import Path
from datetime import datetime

def get_logger(name:str) -> logging.Logger:
    """
    Crear y retorna lkogger configurado para el modulo solicitado

    parametros: 
        name: nombre del modulo

    uso:
        from scr.utils.logger import get_logger(__name__)
        logger = get_logger(__name__)
        logger.info("cargando acrhivos")
        logger.warning("nulos detectados)
        logger.error("archivo no encontrado")
    """

    #crear path
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    #nombre y fecha
    log_file = logs_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

    #crear logger
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    #formato
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    #Handler 1 
    console_handler =  logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    #Handler 2
    file_handler =  logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger



