"""Helpers (and a small CLI) to put the system in a known state before a benchmark.

  python scripts/warmup.py --flush --fill-cache --hot-keys 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "loadtests"))
from payloads import hot_set  # noqa: E402


def flush_cache(redis_url: str, namespace: str) -> int:
    """Delete ONLY keys under '<namespace>:' (never FLUSHALL). Returns how many were removed."""
    import redis

    client = redis.Redis.from_url(redis_url, decode_responses=True)
    removed = 0
    for key in client.scan_iter(match=f"{namespace}:*", count=500):
        removed += client.delete(key)
    client.close()
    return removed


def warm_connections(host: str, requests: int = 50, dim: int = 16) -> None:
    """Uses /predict/baseline (never writes to Redis) to warm TCP connections, thread pools and PyTorch."""
    with httpx.Client(base_url=host, timeout=10) as client:
        for vec in hot_set(requests, dim, seed=999):
            client.post("/predict/baseline", json={"features": vec}).raise_for_status()


def fill_cache(host: str, vectors: list[list[float]]) -> dict[str, int]:
    """Send every vector once to /predict so each becomes a cache entry."""
    seen: dict[str, int] = {}
    with httpx.Client(base_url=host, timeout=10) as client:
        for vec in vectors:
            r = client.post("/predict", json={"features": vec})
            r.raise_for_status()
            seen[r.headers.get("x-cache", "?")] = seen.get(r.headers.get("x-cache", "?"), 0) + 1
    return seen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="http://localhost:8080")
    p.add_argument("--redis-url", default="redis://localhost:6379/0")
    p.add_argument("--namespace", default="infer")
    p.add_argument("--hot-keys", type=int, default=20)
    p.add_argument("--hot-seed", type=int, default=1234)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--flush", action="store_true", help="delete cached predictions under the namespace first")
    p.add_argument("--fill-cache", action="store_true", help="pre-populate the cache with the hot set")
    args = p.parse_args()

    if args.flush:
        print(f"Removed {flush_cache(args.redis_url, args.namespace)} cache keys.")
    warm_connections(args.host, dim=args.dim)
    print("Connections / model warmed via /predict/baseline.")
    if args.fill_cache:
        print("Cache fill result (X-Cache counts):", fill_cache(args.host, hot_set(args.hot_keys, args.dim, args.hot_seed)))


if __name__ == "__main__":
    main()
