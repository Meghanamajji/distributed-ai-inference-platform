"""Application configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces=() lets us name a field `model_path` without a pydantic warning.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    log_level: str = "INFO"
    # In Kubernetes HOSTNAME is the pod name; in Compose we set INSTANCE_NAME explicitly.
    instance_name: str = Field(default_factory=lambda: os.environ.get("HOSTNAME", "local"))

    # Model
    model_path: str = "model/artifacts/model.pt"
    extra_forward_passes: int = Field(default=0, ge=0, le=1000)

    # Redis / cache
    redis_url: str = "redis://localhost:6379/0"
    redis_timeout_seconds: float = Field(default=0.25, ge=0.01, le=10)
    redis_max_connections: int = Field(default=64, ge=1)
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(default=300, ge=1)
    cache_namespace: str = Field(default="infer", pattern=r"^[A-Za-z0-9_-]+$")
    cache_failure_policy: Literal["bypass", "fail"] = "bypass"
    cache_retry_after_seconds: float = Field(default=5.0, ge=0)

    # Concurrency
    inference_threads: int = Field(default=2, ge=1, le=64)
    torch_num_threads: int = Field(default=1, ge=1, le=64)
    inference_timeout_seconds: float = Field(default=2.0, gt=0)
    max_pending_inferences: int = Field(default=256, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
