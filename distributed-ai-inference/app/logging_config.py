"""Structured (JSON lines) logging with a per-request correlation id."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

# The middleware sets this for each request; every log line emitted while handling that request carries it.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_EXTRA_FIELDS = ("method", "path", "status", "duration_ms", "cache", "endpoint")


class JsonFormatter(logging.Formatter):
    def __init__(self, instance: str) -> None:
        super().__init__()
        self._instance = instance

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "instance": self._instance,
        }
        for name in _EXTRA_FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        if record.exc_info:
            # Stack traces go to the server log only, never into API responses.
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str, instance: str) -> None:
    root = logging.getLogger()
    # Replace only handlers we installed earlier, so test tooling keeps its own handlers.
    root.handlers[:] = [h for h in root.handlers if not getattr(h, "_inference_json", False)]
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(instance))
    handler._inference_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level.upper())
