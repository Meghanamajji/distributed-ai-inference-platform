import asyncio
import time

import pytest

from app.errors import InferenceTimeoutError, OverloadedError
from app.inference import InferenceService
from tests.helpers import FEATURES


async def test_predict_returns_valid_distribution(inference):
    out = await inference.predict(FEATURES)
    assert out["label"] == list(out["probabilities"])[out["class_index"]]
    assert pytest.approx(sum(out["probabilities"].values()), abs=1e-4) == 1.0


async def test_concurrent_predictions_are_consistent(inference):
    results = await asyncio.gather(*[inference.predict(FEATURES) for _ in range(40)])
    assert all(r == results[0] for r in results)


class SlowService(InferenceService):
    def _predict_sync(self, features, record_metrics=True):
        time.sleep(0.3)
        return super()._predict_sync(features, record_metrics)


async def test_timeout_raises(loaded_model):
    svc = SlowService(loaded_model, threads=1, timeout_seconds=0.05, max_pending=10)
    try:
        with pytest.raises(InferenceTimeoutError):
            await svc.predict(FEATURES)
    finally:
        svc.shutdown()


async def test_overload_is_rejected(loaded_model):
    svc = SlowService(loaded_model, threads=1, timeout_seconds=5, max_pending=1)
    try:
        first = asyncio.create_task(svc.predict(FEATURES))
        await asyncio.sleep(0.01)
        with pytest.raises(OverloadedError):
            await svc.predict(FEATURES)
        await first
    finally:
        svc.shutdown()


async def test_event_loop_is_not_blocked_by_inference(loaded_model):
    """While 4 slow (0.3s) inferences run, a ticker coroutine must keep ticking every ~10ms."""
    svc = SlowService(loaded_model, threads=4, timeout_seconds=5, max_pending=10)
    gaps, stop = [], False

    async def ticker():
        last = time.perf_counter()
        while not stop:
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    t = asyncio.create_task(ticker())
    try:
        await asyncio.gather(*[svc.predict(FEATURES) for _ in range(4)])
    finally:
        stop = True
        await t
        svc.shutdown()
    assert len(gaps) > 10
    assert max(gaps) < 0.15, f"event loop stalled for {max(gaps):.3f}s"
