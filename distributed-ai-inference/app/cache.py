"""Redis prediction cache with graceful degradation.

Key design:  <namespace>:<model_version>:<sha256(canonical features)>
 - The model version is part of the key AND stored in the value, and both are re-checked on read,
   so a prediction made by a different model version (or for different features) is never returned.
Failure design:
 - Any Redis problem is counted, logged and (policy "bypass") treated as "no cache" so predictions
   still work. After a failure we skip Redis for `retry_after_seconds` (a tiny circuit breaker) so a
   dead Redis does not add a timeout to every request.
 - Policy "fail" raises CacheUnavailableError (HTTP 503) instead.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Sequence

from redis.exceptions import RedisError

from app.errors import CacheUnavailableError
from app.metrics import CACHE_ERRORS, CACHE_HITS, CACHE_MISSES

logger = logging.getLogger(__name__)

# Return statuses from get(), also exposed to clients in the X-Cache header.
HIT, MISS, UNAVAILABLE = "HIT", "MISS", "UNAVAILABLE"

_REDIS_ERRORS = (RedisError, OSError, TimeoutError)  # asyncio.TimeoutError is TimeoutError on 3.11+


class PredictionCache:
    def __init__(
        self,
        client: Any,
        *,
        ttl_seconds: int,
        namespace: str,
        failure_policy: str = "bypass",
        retry_after_seconds: float = 5.0,
        op_timeout_seconds: float = 0.25,
        enabled: bool = True,
    ) -> None:
        self._client = client
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace
        self.failure_policy = failure_policy
        self.enabled = enabled
        self._retry_after = retry_after_seconds
        self._timeout = op_timeout_seconds
        self._down_until = 0.0

    @classmethod
    def from_settings(cls, settings) -> "PredictionCache":
        import redis.asyncio as aioredis

        client = aioredis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=settings.redis_timeout_seconds,
            socket_connect_timeout=settings.redis_timeout_seconds,
            max_connections=settings.redis_max_connections,
            health_check_interval=30,
        )
        return cls(
            client,
            ttl_seconds=settings.cache_ttl_seconds,
            namespace=settings.cache_namespace,
            failure_policy=settings.cache_failure_policy,
            retry_after_seconds=settings.cache_retry_after_seconds,
            op_timeout_seconds=settings.redis_timeout_seconds,
            enabled=settings.cache_enabled,
        )

    # ---- key generation -------------------------------------------------------------------
    @staticmethod
    def canonical_features(features: Sequence[float]) -> str:
        """Same numbers => same string. float() makes 1 and 1.0 identical; json uses repr()."""
        return json.dumps([float(f) for f in features], separators=(",", ":"))

    @classmethod
    def input_digest(cls, features: Sequence[float]) -> str:
        return hashlib.sha256(cls.canonical_features(features).encode()).hexdigest()

    def make_key(self, features: Sequence[float], model_version: str) -> str:
        return f"{self.namespace}:{model_version}:{self.input_digest(features)}"

    # ---- failure bookkeeping --------------------------------------------------------------
    def _circuit_open(self) -> bool:
        return time.monotonic() < self._down_until

    def _on_failure(self, operation: str, exc: BaseException) -> None:
        CACHE_ERRORS.labels(operation=operation).inc()
        self._down_until = time.monotonic() + self._retry_after
        logger.warning("Redis %s failed (%s); skipping cache for %.1fs", operation, type(exc).__name__, self._retry_after)

    def _on_success(self) -> None:
        self._down_until = 0.0

    def _unavailable(self) -> tuple[str, None]:
        if self.failure_policy == "fail":
            raise CacheUnavailableError()
        return UNAVAILABLE, None

    # ---- operations -----------------------------------------------------------------------
    async def get(self, features: Sequence[float], model_version: str) -> tuple[str, dict | None]:
        """Returns (HIT, prediction) | (MISS, None) | (UNAVAILABLE, None)."""
        if self._circuit_open():
            CACHE_ERRORS.labels(operation="circuit_open").inc()
            return self._unavailable()
        key = self.make_key(features, model_version)
        try:
            raw = await asyncio.wait_for(self._client.get(key), timeout=self._timeout)
        except _REDIS_ERRORS as exc:
            self._on_failure("get", exc)
            return self._unavailable()
        self._on_success()
        if raw is None:
            CACHE_MISSES.inc()
            return MISS, None
        try:
            doc = json.loads(raw)
            prediction = doc["prediction"]
            stored_version, stored_digest = doc["model_version"], doc["input_sha256"]
            if not isinstance(prediction, dict):
                raise TypeError("prediction must be an object")
        except (ValueError, KeyError, TypeError):
            CACHE_ERRORS.labels(operation="decode").inc()
            CACHE_MISSES.inc()
            return MISS, None
        if stored_version != model_version or stored_digest != self.input_digest(features):
            CACHE_ERRORS.labels(operation="mismatch").inc()
            CACHE_MISSES.inc()
            return MISS, None
        CACHE_HITS.inc()
        return HIT, prediction

    async def set(self, features: Sequence[float], model_version: str, prediction: dict) -> bool:
        """Best-effort store. Returns True if written."""
        if self._circuit_open():
            return False
        value = json.dumps(
            {"model_version": model_version, "input_sha256": self.input_digest(features), "prediction": prediction},
            separators=(",", ":"),
        )
        try:
            await asyncio.wait_for(
                self._client.set(self.make_key(features, model_version), value, ex=self.ttl_seconds),
                timeout=self._timeout,
            )
        except _REDIS_ERRORS as exc:
            self._on_failure("set", exc)
            if self.failure_policy == "fail":
                raise CacheUnavailableError() from exc
            return False
        self._on_success()
        return True

    async def ping(self) -> bool:
        try:
            await asyncio.wait_for(self._client.ping(), timeout=self._timeout)
        except _REDIS_ERRORS as exc:
            self._on_failure("ping", exc)
            return False
        self._on_success()
        return True

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # shutting down; nothing useful to do
            logger.debug("Error while closing Redis client", exc_info=True)
