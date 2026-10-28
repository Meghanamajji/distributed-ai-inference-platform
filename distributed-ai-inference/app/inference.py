"""Runs the model without blocking the asyncio event loop.

Why a thread pool?  Model inference is CPU-bound Python/C++ code. If we called it directly inside an
`async def` endpoint it would freeze the event loop (no other request could even be accepted) for the
whole compute time. `run_in_executor` moves it to a worker thread. PyTorch releases the GIL inside its
C++ kernels, so threads genuinely run in parallel for the heavy part.

Safety: the model is read-only (eval mode, frozen params) and each call builds its own tensors, so
there is no shared mutable prediction state. The only mutable field here, `_pending`, is touched only
from the event-loop thread.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import torch

from app.errors import InferenceTimeoutError, OverloadedError
from app.metrics import INFERENCE_DURATION, PREDICTION_FAILURES
from app.model import LoadedModel


class InferenceService:
    def __init__(
        self,
        model: LoadedModel,
        *,
        threads: int,
        timeout_seconds: float,
        max_pending: int,
        extra_forward_passes: int = 0,
    ) -> None:
        self._model = model
        self._executor = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="inference")
        self._timeout = timeout_seconds
        self._max_pending = max_pending
        self._extra_passes = extra_forward_passes
        self._pending = 0

    # Runs in a worker thread.
    def _predict_sync(self, features: Sequence[float], record_metrics: bool = True) -> dict:
        start = time.perf_counter()
        with torch.inference_mode():
            x = torch.tensor([list(features)], dtype=torch.float32)
            logits = self._model.module(x)
            for _ in range(self._extra_passes):  # benchmark knob: emulate a heavier model
                logits = self._model.module(x)
            probs = torch.softmax(logits, dim=1)[0].tolist()
        if record_metrics:
            INFERENCE_DURATION.observe(time.perf_counter() - start)
        best = max(range(len(probs)), key=probs.__getitem__)
        names = self._model.class_names
        return {
            "class_index": best,
            "label": names[best],
            "probabilities": {name: round(p, 6) for name, p in zip(names, probs)},
        }

    async def predict(self, features: Sequence[float]) -> dict:
        if self._pending >= self._max_pending:
            PREDICTION_FAILURES.labels(reason="overloaded").inc()
            raise OverloadedError()
        self._pending += 1
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._predict_sync, list(features)),
                timeout=self._timeout,
            )
        except TimeoutError:
            # Note: a thread that already started cannot be killed; its result is simply discarded.
            PREDICTION_FAILURES.labels(reason="timeout").inc()
            raise InferenceTimeoutError() from None
        except Exception:
            PREDICTION_FAILURES.labels(reason="error").inc()
            raise
        finally:
            self._pending -= 1

    def warmup(self) -> None:
        """One dummy forward pass at startup so the first real request is not unusually slow."""
        self._predict_sync([0.0] * self._model.input_dim, record_metrics=False)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
