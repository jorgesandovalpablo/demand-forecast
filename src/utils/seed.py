import random
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """
    Fija el seed global para todos los
    generadores de números aleatorios
    del proyecto.

    Afecta:
        - Python random
        - NumPy
        - LightGBM (vía parámetro random_state)

    Parámetros:
        seed: valor del seed (default: 42)
              debe coincidir con config.yaml
    """
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Seed global fijado: {seed}")