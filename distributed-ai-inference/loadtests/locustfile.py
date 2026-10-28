"""Locust workload for the inference platform.

Configured with environment variables so scripts/benchmark.py (or you) can pick the scenario:

  SCENARIO          baseline | cold | warm | varied | baseline_varied      (default: warm)
  HOT_KEYS          size of the repeated-input set                          (default: 20)
  HOT_SEED          seed for the repeated-input set                         (default: 1234)
  VARIED_SEED       seed for the unique-input stream (random per run if unset)
  INPUT_DIM         feature vector length                                   (default: 16)
  THINK_TIME_SECONDS  pause between a user's requests                       (default: 0)
  MAX_REQUESTS      stop after ~this many requests (0 = only --run-time)    (default: 0)

Scenario meaning (endpoint / inputs):
  baseline         /predict/baseline  repeated inputs   (no cache involved)
  cold             /predict           repeated inputs   (run after flushing the cache; hits build up as it fills)
  warm             /predict           repeated inputs   (run after pre-filling the cache)
  varied           /predict           unique inputs     (cache cannot help)
  baseline_varied  /predict/baseline  unique inputs     (fair comparison partner for 'varied')

A "user" here is a closed-loop client: it sends one request, waits for the answer, (optionally thinks),
then sends the next. So simultaneous in-flight requests <= number of users.

Example (500 users, spawn 50/s, 60 s):
  locust -f loadtests/locustfile.py --headless --host http://localhost:8080 -u 500 -r 50 -t 60s
"""
import itertools
import os
import random
import time

import gevent
from locust import HttpUser, constant, events, task

from payloads import hot_set, varied_vector

SCENARIOS = {
    "baseline": ("/predict/baseline", "hot"),
    "cold": ("/predict", "hot"),
    "warm": ("/predict", "hot"),
    "varied": ("/predict", "varied"),
    "baseline_varied": ("/predict/baseline", "varied"),
}
SCENARIO = os.environ.get("SCENARIO", "warm").lower()
if SCENARIO not in SCENARIOS:
    raise SystemExit(f"Unknown SCENARIO '{SCENARIO}'. Choose from: {', '.join(SCENARIOS)}")

ENDPOINT, INPUT_KIND = SCENARIOS[SCENARIO]
INPUT_DIM = int(os.environ.get("INPUT_DIM", "16"))
HOT_KEYS = int(os.environ.get("HOT_KEYS", "20"))
HOT_SEED = int(os.environ.get("HOT_SEED", "1234"))
VARIED_SEED = os.environ.get("VARIED_SEED", str(time.time_ns()))
THINK_TIME = float(os.environ.get("THINK_TIME_SECONDS", "0"))
MAX_REQUESTS = int(os.environ.get("MAX_REQUESTS", "0"))

HOT = hot_set(HOT_KEYS, INPUT_DIM, HOT_SEED)
_varied_counter = itertools.count()


class InferenceUser(HttpUser):
    wait_time = constant(THINK_TIME)

    def on_start(self):
        self._rng = random.Random()

    def _next_features(self):
        if INPUT_KIND == "hot":
            return self._rng.choice(HOT)
        return varied_vector(next(_varied_counter), INPUT_DIM, VARIED_SEED)

    @task
    def predict(self):
        with self.client.post(ENDPOINT, json={"features": self._next_features()}, name=ENDPOINT, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")


# ---- optional request-count limit -------------------------------------------------------------
_state = {"env": None, "seen": 0, "stopping": False}


@events.init.add_listener
def _remember_environment(environment, **_kwargs):
    _state["env"] = environment


@events.request.add_listener
def _stop_after_max_requests(**_kwargs):
    if not MAX_REQUESTS:
        return
    _state["seen"] += 1
    if _state["seen"] >= MAX_REQUESTS and not _state["stopping"] and _state["env"].runner:
        _state["stopping"] = True
        gevent.spawn(_state["env"].runner.quit)
