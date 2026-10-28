"""Prometheus metrics. Labels are deliberately low-cardinality (never request ids or payloads)."""
from prometheus_client import Counter, Gauge, Histogram

# prometheus_client appends "_total" to Counter names on exposition.
HTTP_REQUESTS = Counter(
    "http_requests", "Total HTTP requests handled.", ["method", "endpoint", "status"]
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "End-to-end HTTP request latency measured inside the app.",
    ["method", "endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
INFERENCE_DURATION = Histogram(
    "inference_duration_seconds",
    "Pure model compute time (inside the worker thread, excludes queueing).",
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
CACHE_HITS = Counter("cache_hits", "Lookups that found a valid cached prediction.")
CACHE_MISSES = Counter("cache_misses", "Successful lookups that found nothing.")
CACHE_ERRORS = Counter("cache_errors", "Cache problems by operation.", ["operation"])
PREDICTION_FAILURES = Counter("prediction_failures", "Failed predictions by reason.", ["reason"])
MODEL_INFO = Gauge("model_info", "Loaded model version (value is always 1).", ["model_version"])

# Create the known label values up front so dashboards show 0 instead of "no data".
for _op in ("get", "set", "ping", "decode", "mismatch", "circuit_open"):
    CACHE_ERRORS.labels(operation=_op)
for _reason in ("timeout", "overloaded", "error"):
    PREDICTION_FAILURES.labels(reason=_reason)
