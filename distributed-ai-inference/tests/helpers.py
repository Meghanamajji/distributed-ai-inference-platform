"""Shared test helpers (plain module so tests can import them without importing conftest)."""
from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache import PredictionCache

FEATURES = [0.5, -1.2, 0.3, 2.1, -0.7, 0.0, 1.4, -0.3, 0.9, -2.0, 0.1, 0.6, -0.8, 1.1, 0.2, -1.5]
OTHER_FEATURES = [-1.0] * 16


class FailingRedis:
    """Stand-in for a Redis server that is down."""

    async def get(self, *a, **k):
        raise RedisConnectionError("simulated outage")

    set = ping = get

    async def aclose(self):
        return None


def make_cache(client, **overrides):
    params = dict(ttl_seconds=60, namespace="test", failure_policy="bypass", retry_after_seconds=30.0, op_timeout_seconds=1.0)
    params.update(overrides)
    return PredictionCache(client, **params)
