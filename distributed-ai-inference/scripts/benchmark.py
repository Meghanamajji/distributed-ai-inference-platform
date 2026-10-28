"""Reproducible benchmark: runs Locust scenarios A-E, collects measurements, writes JSON/CSV/Markdown.

Scenarios
  A  baseline          /predict/baseline, repeated inputs
  B  cache, cold       /predict, repeated inputs, cache flushed first (NOT pre-filled)
  C  cache, warm       /predict, repeated inputs, cache pre-filled with exactly those inputs
  D  cache, varied     /predict, unique inputs (cache can't help)
  E  baseline, varied  /predict/baseline, unique inputs (fair comparison partner for D)

Example:
  python scripts/benchmark.py --users 100 --spawn-rate 25 --run-time 30 --scenarios A,B,C,D,E

Every number in the output comes from Locust's own CSV and from the replicas' /metrics counters.
Nothing is estimated or hardcoded. The 30% / 2x numbers below are only the *targets* used to label results.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warmup  # noqa: E402
from payloads import hot_set  # noqa: E402

TARGET_P95_REDUCTION_PCT = 30.0
TARGET_THROUGHPUT_MULTIPLIER = 2.0

SCENARIOS = {
    "A": ("baseline", "Baseline: /predict/baseline, repeated inputs"),
    "B": ("cold", "Cache enabled, COLD cache (flushed, not pre-filled), repeated inputs"),
    "C": ("warm", "Cache enabled, WARM cache (pre-filled), repeated inputs"),
    "D": ("varied", "Cache enabled, varied (unique) inputs"),
    "E": ("baseline_varied", "Baseline with varied (unique) inputs - comparison partner for D"),
}


# ---- pure helpers (unit-tested) -------------------------------------------------------------
def latency_reduction_pct(baseline_p95: float, optimized_p95: float) -> float | None:
    """(baseline p95 - optimized p95) / baseline p95 * 100"""
    if not baseline_p95:
        return None
    return (baseline_p95 - optimized_p95) / baseline_p95 * 100.0


def throughput_multiplier(baseline_rps: float, optimized_rps: float) -> float | None:
    """optimized requests/s / baseline requests/s"""
    if not baseline_rps:
        return None
    return optimized_rps / baseline_rps


def parse_cache_counters(metrics_text: str) -> tuple[float, float]:
    def grab(name: str) -> float:
        m = re.search(rf"^{name}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$", metrics_text, re.MULTILINE)
        return float(m.group(1)) if m else 0.0

    return grab("cache_hits_total"), grab("cache_misses_total")


def parse_locust_stats(stats_csv: Path) -> dict:
    with open(stats_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    agg = next((r for r in rows if r.get("Name") == "Aggregated"), None)
    if agg is None:
        raise RuntimeError(f"No 'Aggregated' row in {stats_csv}")
    requests, failures = int(agg["Request Count"]), int(agg["Failure Count"])
    return {
        "requests": requests,
        "failures": failures,
        "error_rate_pct": (failures / requests * 100.0) if requests else 0.0,
        "rps": float(agg["Requests/s"]),
        "avg_ms": float(agg["Average Response Time"]),
        "p50_ms": float(agg["50%"]),
        "p95_ms": float(agg["95%"]),
        "p99_ms": float(agg["99%"]),
    }


# ---- orchestration --------------------------------------------------------------------------
def scrape_cache_counters(urls: list[str]) -> tuple[float, float] | None:
    hits = misses = 0.0
    try:
        for url in urls:
            h, m = parse_cache_counters(httpx.get(url, timeout=5).text)
            hits, misses = hits + h, misses + m
    except httpx.HTTPError:
        return None
    return hits, misses


def run_scenario(sid: str, args, out_dir: Path) -> dict:
    key, description = SCENARIOS[sid]
    print(f"\n=== Scenario {sid}: {description} ===")
    warmup.warm_connections(args.host, dim=args.dim)
    if key in ("cold", "warm", "varied"):
        print(f"  flushed {warmup.flush_cache(args.redis_url, args.namespace)} cache keys")
    if key == "warm":
        print("  pre-filled cache:", warmup.fill_cache(args.host, hot_set(args.hot_keys, args.dim, args.hot_seed)))

    before = scrape_cache_counters(args.metrics_urls)
    prefix = out_dir / f"locust_{sid}_{key}"
    env = dict(
        os.environ,
        SCENARIO=key, HOT_KEYS=str(args.hot_keys), HOT_SEED=str(args.hot_seed), INPUT_DIM=str(args.dim),
        VARIED_SEED=str(time.time_ns()), MAX_REQUESTS=str(args.max_requests),
        THINK_TIME_SECONDS=str(args.think_time), PYTHONPATH=str(ROOT / "loadtests"),
    )
    cmd = [
        sys.executable, "-m", "locust", "-f", str(ROOT / "loadtests" / "locustfile.py"), "--headless",
        "--host", args.host, "--users", str(args.users), "--spawn-rate", str(args.spawn_rate),
        "--run-time", f"{args.run_time}s", "--csv", str(prefix), "--reset-stats", "--stop-timeout", "5",
    ]
    print("  running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=False)  # Locust exits non-zero if any request failed; we still want the CSV
    stats_csv = Path(f"{prefix}_stats.csv")
    if not stats_csv.exists():
        raise SystemExit(f"Locust produced no results for scenario {sid}; see its output above.")
    after = scrape_cache_counters(args.metrics_urls)

    result = {"scenario": sid, "name": key, "description": description, "users": args.users,
              "spawn_rate": args.spawn_rate, "run_time_s": args.run_time, **parse_locust_stats(stats_csv)}
    if before and after:
        hits, misses = after[0] - before[0], after[1] - before[1]
        result.update(cache_hits=int(hits), cache_misses=int(misses),
                      cache_hit_ratio=(hits / (hits + misses)) if (hits + misses) else None)
    else:
        result.update(cache_hits=None, cache_misses=None, cache_hit_ratio=None)
    return result


def compare(results: dict[str, dict], base: str, opt: str) -> dict | None:
    if base not in results or opt not in results:
        return None
    b, o = results[base], results[opt]
    red, mult = latency_reduction_pct(b["p95_ms"], o["p95_ms"]), throughput_multiplier(b["rps"], o["rps"])
    return {"baseline": base, "optimized": opt, "p95_baseline_ms": b["p95_ms"], "p95_optimized_ms": o["p95_ms"],
            "p95_reduction_pct": red, "rps_baseline": b["rps"], "rps_optimized": o["rps"], "throughput_multiplier": mult,
            "meets_p95_target": red is not None and red >= TARGET_P95_REDUCTION_PCT,
            "meets_throughput_target": mult is not None and mult >= TARGET_THROUGHPUT_MULTIPLIER}


def fmt(v, spec=".2f", suffix=""):
    return "n/a" if v is None else f"{v:{spec}}{suffix}"


def write_outputs(results: list[dict], comparisons: list[dict], args, out_dir: Path) -> None:
    meta = {"timestamp": datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(),
            "python": platform.python_version(), "cpu_count": os.cpu_count(), "host": args.host,
            "users": args.users, "spawn_rate": args.spawn_rate, "run_time_s": args.run_time,
            "max_requests": args.max_requests, "hot_keys": args.hot_keys, "think_time_s": args.think_time}
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "results": results, "comparisons": comparisons}, indent=2))
    cols = ["scenario", "name", "users", "spawn_rate", "run_time_s", "requests", "failures", "error_rate_pct", "rps",
            "avg_ms", "p50_ms", "p95_ms", "p99_ms", "cache_hits", "cache_misses", "cache_hit_ratio"]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    lines = [f"# Benchmark report ({meta['timestamp']})", "",
             f"- Machine: {meta['platform']}, {meta['cpu_count']} logical CPUs, Python {meta['python']}",
             f"- Target: {args.host} | users={args.users} spawn_rate={args.spawn_rate}/s run_time={args.run_time}s "
             f"max_requests={args.max_requests or 'unlimited'} hot_keys={args.hot_keys} think_time={args.think_time}s",
             "- Latencies are measured by Locust (client side, through Nginx), in milliseconds.",
             "- Stats exclude the ramp-up period (`--reset-stats`). Locust and the services share this machine.", "",
             "| Scn | Description | Requests | Req/s | Avg ms | p50 | p95 | p99 | Error % | Cache hit ratio |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        ratio = None if r["cache_hit_ratio"] is None else r["cache_hit_ratio"] * 100
        lines.append(f"| {r['scenario']} | {r['description']} | {r['requests']} | {fmt(r['rps'])} | {fmt(r['avg_ms'])} | "
                     f"{fmt(r['p50_ms'], '.0f')} | {fmt(r['p95_ms'], '.0f')} | {fmt(r['p99_ms'], '.0f')} | "
                     f"{fmt(r['error_rate_pct'])} | {fmt(ratio, '.1f', '%')} |")
    lines += ["", "## Comparisons (computed from the measurements above)", "",
              "p95 reduction % = (baseline p95 - optimized p95) / baseline p95 x 100; throughput x = optimized req/s / baseline req/s.", "",
              "| Baseline -> Optimized | p95 base | p95 opt | p95 reduction | Req/s base | Req/s opt | Throughput x | "
              f"Meets {TARGET_P95_REDUCTION_PCT:.0f}% p95 goal? | Meets {TARGET_THROUGHPUT_MULTIPLIER:.0f}x goal? |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in comparisons:
        lines.append(f"| {c['baseline']} -> {c['optimized']} | {fmt(c['p95_baseline_ms'], '.0f')} | {fmt(c['p95_optimized_ms'], '.0f')} | "
                     f"{fmt(c['p95_reduction_pct'], '.1f', '%')} | {fmt(c['rps_baseline'])} | {fmt(c['rps_optimized'])} | "
                     f"{fmt(c['throughput_multiplier'], '.2f', 'x')} | {'yes' if c['meets_p95_target'] else 'no'} | "
                     f"{'yes' if c['meets_throughput_target'] else 'no'} |")
    lines += ["", "## How to read this honestly", "",
              "- A->C is the repeated-input case the cache is designed for. D vs E (unique inputs) shows the cost when caching cannot help.",
              "- B (cold) is expected to sit between A and C; it fills as the run proceeds.",
              "- If a goal is 'no', do not claim it. See README 'Interpreting results' for likely bottlenecks "
              "(tiny model vs. Redis round trip, Nginx/Docker networking, Locust sharing the CPU, worker counts).",
              "- `users` is simulated clients, not requests and not simultaneous concurrency (each user has at most one request in flight)."]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="http://localhost:8080")
    p.add_argument("--users", type=int, default=100)
    p.add_argument("--spawn-rate", type=float, default=25)
    p.add_argument("--run-time", type=int, default=30, help="seconds per scenario")
    p.add_argument("--max-requests", type=int, default=0, help="stop a scenario after ~N requests (0 = time only)")
    p.add_argument("--think-time", type=float, default=0.0)
    p.add_argument("--scenarios", default="A,B,C,D,E")
    p.add_argument("--hot-keys", type=int, default=20)
    p.add_argument("--hot-seed", type=int, default=1234)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--redis-url", default="redis://localhost:6379/0")
    p.add_argument("--namespace", default="infer")
    p.add_argument("--metrics-urls", default="http://localhost:8001/metrics,http://localhost:8002/metrics",
                   help="comma-separated replica /metrics URLs (used for cache hit ratio)")
    p.add_argument("--output-dir", default=str(ROOT / "results"))
    args = p.parse_args()
    args.metrics_urls = [u for u in args.metrics_urls.split(",") if u]

    try:
        httpx.get(f"{args.host}/ready", timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(f"Target {args.host} is not ready ({exc}). Start the stack: docker compose up -d --build")

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenarios {unknown}; choose from {list(SCENARIOS)}")

    results = [run_scenario(s, args, out_dir) for s in wanted]
    by_id = {r["scenario"]: r for r in results}
    comparisons = [c for c in (compare(by_id, "A", "C"), compare(by_id, "A", "B"), compare(by_id, "E", "D")) if c]
    write_outputs(results, comparisons, args, out_dir)
    print(f"\nDone. Results in {out_dir}\n  report.md  summary.json  summary.csv")
    print((out_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
