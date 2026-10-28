"""Pydantic request/response schemas."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Each feature must be a finite number in a sane range (rejects NaN / Infinity / absurd values).
Feature = Annotated[float, Field(allow_inf_nan=False, ge=-1_000_000, le=1_000_000)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"features": [0.5, -1.2, 0.3, 2.1, -0.7, 0.0, 1.4, -0.3, 0.9, -2.0, 0.1, 0.6, -0.8, 1.1, 0.2, -1.5]}]},
    )

    features: list[Feature] = Field(
        min_length=1,
        max_length=1024,
        description="Numeric feature vector. Its length must equal the model's input_dim (see GET /model/info).",
    )


class Prediction(BaseModel):
    class_index: int
    label: str
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    prediction: Prediction
    cached: bool = Field(description="True if the prediction came from Redis instead of running the model.")
    instance: str = Field(description="Which API replica served this request.")
    server_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    instance: str
    uptime_seconds: float


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, str]


class InferenceInfo(BaseModel):
    device: str
    inference_threads: int
    torch_intra_op_threads: int
    timeout_seconds: float
    extra_forward_passes: int
    cache_enabled: bool
    cache_ttl_seconds: int
    cache_failure_policy: str


class ModelInfoResponse(BaseModel):
    model_version: str
    architecture: str
    input_dim: int
    hidden_dim: int
    num_classes: int
    class_names: list[str]
    parameter_count: int
    inference: InferenceInfo
    note: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | list[dict]
    request_id: str
