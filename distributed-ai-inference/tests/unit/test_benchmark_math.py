"""Tests the benchmark helpers with made-up numbers (these are test inputs, NOT results)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import benchmark  # noqa: E402


def test_latency_reduction_formula():
    assert benchmark.latency_reduction_pct(100.0, 70.0) == pytest.approx(30.0)
    assert benchmark.latency_reduction_pct(100.0, 130.0) == pytest.approx(-30.0)  # slower => negative


def test_throughput_multiplier_formula():
    assert benchmark.throughput_multiplier(200.0, 400.0) == pytest.approx(2.0)


def test_zero_baseline_is_handled():
    assert benchmark.latency_reduction_pct(0, 5) is None
    assert benchmark.throughput_multiplier(0, 5) is None


def test_parse_cache_counters():
    text = "# HELP cache_hits_total x\ncache_hits_total 12.0\ncache_misses_total 3.0\ncache_hits_created 1.7e9\n"
    assert benchmark.parse_cache_counters(text) == (12.0, 3.0)


def test_parse_locust_csv(tmp_path):
    csv_file = tmp_path / "x_stats.csv"
    csv_file.write_text(
        "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,Min Response Time,"
        "Max Response Time,Average Content Size,Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%\n"
        "POST,/predict,10,1,5,6,1,9,100,2.5,0.25,5,6,7,7,8,9,9,9,9,9,9\n"
        ",Aggregated,10,1,5,6,1,9,100,2.5,0.25,5,6,7,7,8,9,9,9,9,9,9\n"
    )
    row = benchmark.parse_locust_stats(csv_file)
    assert row["requests"] == 10 and row["failures"] == 1
    assert row["error_rate_pct"] == pytest.approx(10.0)
    assert row["p95_ms"] == 9 and row["rps"] == 2.5
