# Interview guide

## 30-second pitch
"I built an inference platform around a small PyTorch model. A FastAPI service validates requests, checks a shared Redis cache, and otherwise runs the model in a thread pool so the event loop stays free.
Two replicas run behind Nginx; Prometheus scrapes them and Grafana shows rate, latency percentiles, errors and cache hit ratio. I load-tested with Locust and wrote a benchmark script that compares cached and uncached paths, so every performance claim comes from my own measurements. The model is synthetic - the project is about the serving infrastructure."

## Decisions and the reasoning behind them
- **Load the model once at startup.** Loading is slow; per-request loading would dominate latency. `eval()` + `inference_mode()` disable training behaviour and gradient tracking.
- **Thread pool for inference.** Inference is CPU-bound; async doesn't speed it up, and running it inline blocks every other request. `run_in_executor` keeps the loop responsive (I have a test that proves the loop keeps ticking). Timeouts and a pending cap prevent unbounded queues.
- **Redis cache design.** Key = namespace + model version + SHA-256 of canonical features; TTL for freshness; version re-checked on read so a model update can't serve stale answers. Redis outage policy is configurable: `bypass` (availability) or `fail` (strictness), plus a small circuit breaker so a dead Redis doesn't add a timeout to every request.
- **When caching doesn't help.** Unique inputs never hit; a hit still pays Nginx + Redis round trip; with a tiny model, compute may be cheaper than the cache hop. That's why there's a baseline endpoint and a varied-input scenario.
- **Load balancing.** Nginx round-robin across identical replicas, shared Redis so any replica benefits from another's cached work. Replicas add capacity, not automatically lower latency, especially when everything shares one laptop.
- **Monitoring.** RED-style metrics (rate, errors, duration) plus cache and inference metrics. Labels are low-cardinality (route templates, not raw URLs/IDs) so Prometheus doesn't blow up. Percentiles come from histograms.
- **Benchmarking fairness.** Same payloads, warm-up, ramp-up excluded, cache state controlled per scenario, hit ratio from server counters, improvements computed with explicit formulas.
- **Liveness vs readiness.** Restart vs. stop-sending-traffic; used by Compose and Kubernetes.

## Questions to be ready for
- *Why didn't you use multiple Uvicorn workers?* Each is a full process with its own model copy and its own Prometheus registry; replicas give the same scaling with correct per-instance metrics.
- *What happens if Redis dies?* Depends on `CACHE_FAILURE_POLICY`; default serves uncached predictions and surfaces it in metrics/readiness.
- *Cache stampede?* Concurrent identical misses each compute once; request coalescing is a listed improvement.
- *What would make this production-grade?* Auth, TLS, rate limiting, HA Redis, autoscaling, tracing, CI, secrets management.
- *Did you hit the 30% / 2x goals?* Answer only from your own `results/` report, and explain the bottleneck either way.
