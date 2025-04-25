from typing import Optional

from pydantic import BaseModel, Field


class ValidationRequestModel(BaseModel):
    url: str = Field(..., description="Parameter to provide url for image scraping.")


class ClassificationRequestModel(BaseModel):
    url: str = Field(..., description="Parameter to provide url for image scraping.")
    user_id: str = Field(..., description="Parameter to provide a user identifier.")
    timestamp: str = Field(
        ..., description="Parameter to provide a timestamp of request."
    )


class RetrainingRequestModel(BaseModel):
    urls: list[str] = Field(..., description="List of S3 URLs for images to retrain on")
    user_id: str = Field(..., description="Parameter to provide a user identifier.")
    timestamp: str = Field(
        ..., description="Parameter to provide a timestamp of request."
    )


class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float


class RetrainingResponse(BaseModel):
    message: str
    old_metrics: MetricsResponse
    new_metrics: MetricsResponse
    model_path: Optional[str] = None
