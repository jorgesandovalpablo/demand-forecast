from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import date


# ─────────────────────────────────────────
# Requests — lo que recibe la API
# ─────────────────────────────────────────
class PredictionRequest(BaseModel):
    """
    Schema de entrada para solicitar predicciones.
    """
    store_nbr: int = Field(
        ...,
        ge=1,
        le=54,
        description="Número de tienda (1-54)"
    )
    horizon: int = Field(
        ...,
        description="Horizonte de predicción: 7 (diario) o 30 (mensual)"
    )
    family: Optional[str] = Field(
        default=None,
        description="Familia de producto. Si es None retorna todas."
    )

    @validator('horizon')
    def horizon_must_be_valid(cls, v):
        if v not in [7, 30]:
            raise ValueError(
                "horizon debe ser 7 (diario) o 30 (mensual)"
            )
        return v

    class Config:
        schema_extra = {
            "example": {
                "store_nbr": 1,
                "horizon":   7,
                "family":    "GROCERY I"
            }
        }


class RetrainingRequest(BaseModel):
    """
    Schema de entrada para solicitar reentrenamiento.
    """
    horizon: int = Field(
        ...,
        description="Horizonte a reentrenar: 7 o 30"
    )
    force: bool = Field(
        default=False,
        description="Si True, reentrena aunque métricas no mejoren"
    )

    @validator('horizon')
    def horizon_must_be_valid(cls, v):
        if v not in [7, 30]:
            raise ValueError(
                "horizon debe ser 7 o 30"
            )
        return v


# ─────────────────────────────────────────
# Responses — lo que retorna la API
# ─────────────────────────────────────────
class PredictionItem(BaseModel):
    """
    Una predicción individual por
    fecha, tienda y familia.
    """
    date:            date
    store_nbr:       int
    family:          str
    predicted_sales: float
    lower_bound:     float
    upper_bound:     float


class PredictionResponse(BaseModel):
    """
    Respuesta completa de predicción.
    """
    store_nbr:    int
    horizon:      int
    n_predictions: int
    predictions:  list[PredictionItem]

    class Config:
        schema_extra = {
            "example": {
                "store_nbr":     1,
                "horizon":       7,
                "n_predictions": 7,
                "predictions": [
                    {
                        "date":            "2024-01-16",
                        "store_nbr":       1,
                        "family":          "GROCERY I",
                        "predicted_sales": 245.30,
                        "lower_bound":     198.20,
                        "upper_bound":     292.40
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """
    Respuesta del endpoint de salud.
    """
    status:          str
    models_loaded:   list[int]
    version:         str


class MetricsResponse(BaseModel):
    """
    Respuesta con métricas del modelo.
    """
    horizon: int
    rmse:    float
    mae:     float
    mape:    float
    rmsle:   float
    wape:    float


class RetrainingResponse(BaseModel):
    """
    Respuesta del endpoint de reentrenamiento.
    """
    horizon:         int
    status:          str
    metrics_before:  dict
    metrics_after:   dict
    model_updated:   bool