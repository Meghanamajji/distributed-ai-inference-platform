import asyncio

import pytest
from prometheus_client import REGISTRY

from app.cache import HIT, MISS, UNAVAILABLE
from app.errors import CacheUnavailableError
from tests.helpers import FEATURES, OTHER_FEATURES, FailingRedis, make_cache

PRED = {"class_index": 1, "label": "class_b", "probabilities": {"class_a": 0.1, "class_b": 0.8, "class_c": 0.1}}


def metric(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_key_is_deterministic_and_input_sensitive(cache):
    k1 = cache.make_key(FEATURES, "v1")
    assert k1 == cache.make_key(list(FEATURES), "v1")
    assert cache.make_key([1, 2], "v1") == cache.make_key([1.0, 2.0], "v1")  # 1 == 1.0
    assert k1 != cache.make_key(OTHER_FEATURES, "v1")
    assert k1 != cache.make_key(FEATURES, "v2")  # different model version => different key
    assert " " not in k1


async def test_miss_then_hit(cache):
    hits, misses = metric("cache_hits_total"), metric("cache_misses_total")
    assert await cache.get(FEATURES, "v1") == (MISS, None)
    assert await cache.set(FEATURES, "v1", PRED) is True
    assert await cache.get(FEATURES, "v1") == (HIT, PRED)
    assert metric("cache_hits_total") == hits + 1
    assert metric("cache_misses_total") == misses + 1


async def test_never_returns_other_input_or_other_version(cache):
    await cache.set(FEATURES, "v1", PRED)
    assert (await cache.get(OTHER_FEATURES, "v1"))[0] == MISS
    assert (await cache.get(FEATURES, "v2"))[0] == MISS


async def test_entries_expire(fake_redis):
    short = make_cache(fake_redis, ttl_seconds=1)
    await short.set(FEATURES, "v1", PRED)
    assert (await short.get(FEATURES, "v1"))[0] == HIT
    assert 0 < await fake_redis.ttl(short.make_key(FEATURES, "v1")) <= 1
    await asyncio.sleep(1.2)
    assert (await short.get(FEATURES, "v1"))[0] == MISS


async def test_tampered_value_with_wrong_version_is_rejected(cache, fake_redis):
    key = cache.make_key(FEATURES, "v1")
    await fake_redis.set(key, '{"model_version":"OLD","input_sha256":"x","prediction":{}}')
    before = metric("cache_errors_total", {"operation": "mismatch"})
    assert (await cache.get(FEATURES, "v1"))[0] == MISS
    assert metric("cache_errors_total", {"operation": "mismatch"}) == before + 1


async def test_corrupt_value_is_a_miss(cache, fake_redis):
    await fake_redis.set(cache.make_key(FEATURES, "v1"), "not json")
    assert (await cache.get(FEATURES, "v1"))[0] == MISS


async def test_redis_down_bypass_policy_degrades_gracefully():
    c = make_cache(FailingRedis(), retry_after_seconds=30.0)
    before = metric("cache_errors_total", {"operation": "get"})
    assert await c.get(FEATURES, "v1") == (UNAVAILABLE, None)
    assert metric("cache_errors_total", {"operation": "get"}) == before + 1
    assert await c.set(FEATURES, "v1", PRED) is False  # circuit open: skipped, no exception
    assert await c.ping() is False


async def test_circuit_breaker_skips_redis_after_failure():
    calls = {"n": 0}

    class Counting(FailingRedis):
        async def get(self, *a, **k):
            calls["n"] += 1
            return await super().get()

    c = make_cache(Counting(), retry_after_seconds=30.0)
    await c.get(FEATURES, "v1")
    await c.get(FEATURES, "v1")
    await c.get(FEATURES, "v1")
    assert calls["n"] == 1  # later lookups skipped Redis entirely


async def test_redis_down_fail_policy_raises():
    c = make_cache(FailingRedis(), failure_policy="fail")
    with pytest.raises(CacheUnavailableError):
        await c.get(FEATURES, "v1")


async def test_recovers_after_retry_window(fake_redis):
    c = make_cache(fake_redis, retry_after_seconds=0.0)
    c._on_failure("get", RuntimeError("x"))
    assert (await c.get(FEATURES, "v1"))[0] == MISS  # window of 0s => Redis is tried again
