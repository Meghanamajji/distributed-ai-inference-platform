import asyncio
import time

from prometheus_client import REGISTRY

from app.inference import InferenceService
from tests.helpers import FEATURES, OTHER_FEATURES, FailingRedis, make_cache


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["instance"] == "test-instance"


async def test_ready_ok(client):
    r = await client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "checks": {"model": "ok", "cache": "ok"}}


async def test_ready_degraded_when_redis_down_and_policy_bypass(make_client):
    c = await make_client(make_cache(FailingRedis()))
    r = await c.get("/ready")
    assert r.status_code == 200
    assert "degraded" in r.json()["checks"]["cache"]


async def test_not_ready_when_redis_down_and_policy_fail(make_client, settings):
    c = await make_client(make_cache(FailingRedis(), failure_policy="fail"), settings_override=settings.model_copy(update={"cache_failure_policy": "fail"}))
    assert (await c.get("/ready")).status_code == 503


async def test_model_info(client):
    body = (await client.get("/model/info")).json()
    assert body["input_dim"] == 16 and body["num_classes"] == 3
    assert body["inference"]["device"] == "cpu"


async def test_predict_miss_then_hit_with_identical_prediction(client):
    first = await client.post("/predict", json={"features": FEATURES})
    second = await client.post("/predict", json={"features": FEATURES})
    assert first.status_code == second.status_code == 200
    assert first.headers["x-cache"] == "MISS" and second.headers["x-cache"] == "HIT"
    assert first.json()["cached"] is False and second.json()["cached"] is True
    assert first.json()["prediction"] == second.json()["prediction"]
    assert abs(sum(first.json()["prediction"]["probabilities"].values()) - 1) < 1e-4


async def test_cached_response_belongs_to_its_own_input(client):
    a = (await client.post("/predict", json={"features": FEATURES})).json()["prediction"]
    b = (await client.post("/predict", json={"features": OTHER_FEATURES})).json()["prediction"]
    assert a != b
    assert (await client.post("/predict", json={"features": FEATURES})).json()["prediction"] == a
    assert (await client.post("/predict", json={"features": OTHER_FEATURES})).json()["prediction"] == b


async def test_baseline_bypasses_cache_but_matches_prediction(client, fake_redis):
    cached_path = (await client.post("/predict", json={"features": FEATURES})).json()["prediction"]
    n_keys = len(await fake_redis.keys("*"))
    r = await client.post("/predict/baseline", json={"features": OTHER_FEATURES})
    assert r.headers["x-cache"] == "BYPASS" and r.json()["cached"] is False
    assert len(await fake_redis.keys("*")) == n_keys  # baseline wrote nothing
    base = (await client.post("/predict/baseline", json={"features": FEATURES})).json()["prediction"]
    assert base == cached_path  # same model, same answer


async def test_invalid_payloads(client):
    cases = [
        {"features": [1.0, 2.0]},  # wrong dimension -> handled by route
        {"features": []},
        {"features": ["a"] * 16},
        {"features": FEATURES, "extra": 1},
        {},
    ]
    for payload in cases:
        r = await client.post("/predict", json=payload)
        assert r.status_code == 422, payload
        body = r.json()
        assert "request_id" in body and "Traceback" not in r.text


async def test_nan_and_malformed_json_rejected(client):
    nan_body = b'{"features": [NaN, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}'
    r = await client.post("/predict", content=nan_body, headers={"content-type": "application/json"})
    assert r.status_code == 422
    r = await client.post("/predict", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


async def test_redis_outage_still_serves_predictions(make_client):
    c = await make_client(make_cache(FailingRedis()))
    r = await c.post("/predict", json={"features": FEATURES})
    assert r.status_code == 200
    assert r.headers["x-cache"] == "UNAVAILABLE" and r.json()["cached"] is False


async def test_redis_outage_with_fail_policy_returns_503(make_client):
    c = await make_client(make_cache(FailingRedis(), failure_policy="fail"))
    r = await c.post("/predict", json={"features": FEATURES})
    assert r.status_code == 503 and r.json()["error"] == "cache_unavailable"
    # baseline never needs Redis, so it keeps working
    assert (await c.post("/predict/baseline", json={"features": FEATURES})).status_code == 200


class _Slow(InferenceService):
    def _predict_sync(self, features, record_metrics=True):
        time.sleep(0.3)
        return super()._predict_sync(features, record_metrics)


async def test_inference_timeout_returns_504_without_internals(make_client, loaded_model, cache):
    slow = _Slow(loaded_model, threads=1, timeout_seconds=0.05, max_pending=10)
    c = await make_client(cache, inference_override=slow)
    r = await c.post("/predict/baseline", json={"features": FEATURES})
    slow.shutdown()
    assert r.status_code == 504 and r.json()["error"] == "inference_timeout"


async def test_metrics_endpoint(client):
    await client.post("/predict", json={"features": FEATURES})
    text = (await client.get("/metrics")).text
    for name in ("http_requests_total", "http_request_duration_seconds_bucket", "inference_duration_seconds_bucket",
                 "cache_hits_total", "cache_misses_total", "cache_errors_total", "prediction_failures_total"):
        assert name in text, name


async def test_unknown_paths_do_not_create_new_label_values(client):
    await client.get("/this/does/not/exist/12345")
    assert REGISTRY.get_sample_value("http_requests_total", {"method": "GET", "endpoint": "unmatched", "status": "404"}) >= 1
    assert "12345" not in (await client.get("/metrics")).text


async def test_request_id_is_generated_and_echoed(client):
    assert len((await client.get("/health")).headers["x-request-id"]) == 32
    r = await client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    r = await client.get("/health", headers={"X-Request-ID": "bad id with spaces!"})
    assert r.headers["x-request-id"] != "bad id with spaces!"


async def test_concurrent_predictions(client):
    payloads = [FEATURES] * 20 + [[float(i) / 10] * 16 for i in range(30)]
    responses = await asyncio.gather(*[client.post("/predict", json={"features": p}) for p in payloads])
    assert all(r.status_code == 200 for r in responses)
    same = {str(r.json()["prediction"]) for r in responses[:20]}
    assert len(same) == 1  # identical input => identical prediction, cached or not
