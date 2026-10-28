"""Integration tests against the running Docker Compose stack.

Start it:  docker compose up -d --build
Run it:    pytest -m integration
Override the URLs with INTEGRATION_BASE_URL / INTEGRATION_API1_URL / INTEGRATION_API2_URL.
"""
import os
import uuid

import httpx
import pytest

BASE = os.environ.get("INTEGRATION_BASE_URL", "http://localhost:8080")
API1 = os.environ.get("INTEGRATION_API1_URL", "http://localhost:8001")
API2 = os.environ.get("INTEGRATION_API2_URL", "http://localhost:8002")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def stack_up():
    try:
        httpx.get(f"{BASE}/ready", timeout=3).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Compose stack not reachable at {BASE}: {exc}")


def unique_features():
    seed = uuid.uuid4().int % 10_000
    return [round((seed % 97) / 10 - 4, 4) + i * 0.001 for i in range(16)]


def test_ready_through_load_balancer():
    assert httpx.get(f"{BASE}/ready").json()["status"] == "ready"


def test_second_identical_request_is_a_cache_hit():
    body = {"features": unique_features()}
    first = httpx.post(f"{BASE}/predict", json=body)
    second = httpx.post(f"{BASE}/predict", json=body)
    assert first.headers["x-cache"] == "MISS" and second.headers["x-cache"] == "HIT"
    assert first.json()["prediction"] == second.json()["prediction"]


def test_cache_is_shared_between_replicas():
    body = {"features": unique_features()}
    assert httpx.post(f"{API1}/predict", json=body).headers["x-cache"] == "MISS"
    assert httpx.post(f"{API2}/predict", json=body).headers["x-cache"] == "HIT"


def test_both_replicas_serve_traffic_through_nginx():
    with httpx.Client() as client:
        served = {client.post(f"{BASE}/predict/baseline", json={"features": unique_features()}).headers["x-served-by"] for _ in range(30)}
    assert served == {"api1", "api2"}


def test_metrics_blocked_at_load_balancer_but_available_per_replica():
    assert httpx.get(f"{BASE}/metrics").status_code == 404
    for url in (API1, API2):
        assert "cache_hits_total" in httpx.get(f"{url}/metrics").text
