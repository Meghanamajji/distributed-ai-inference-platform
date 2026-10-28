"""FastAPI application factory."""
from __future__ import annotations

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.cache import PredictionCache
from app.config import Settings, get_settings
from app.errors import AppError
from app.inference import InferenceService
from app.logging_config import configure_logging, request_id_var
from app.metrics import HTTP_LATENCY, HTTP_REQUESTS, MODEL_INFO
from app.model import LoadedModel, load_model
from app.routes import routers

logger = logging.getLogger("app")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def create_app(
    settings: Settings | None = None,
    *,
    cache: PredictionCache | None = None,
    inference: InferenceService | None = None,
    model: LoadedModel | None = None,
) -> FastAPI:
    """Build the app. Pass cache/inference/model to inject test doubles; otherwise startup builds them."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.instance_name)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        created: list[str] = []
        if app.state.model is None:
            app.state.model = load_model(settings.model_path)  # once, at startup - never per request
            created.append("model")
        if app.state.inference is None:
            torch.set_num_threads(settings.torch_num_threads)
            app.state.inference = InferenceService(
                app.state.model,
                threads=settings.inference_threads,
                timeout_seconds=settings.inference_timeout_seconds,
                max_pending=settings.max_pending_inferences,
                extra_forward_passes=settings.extra_forward_passes,
            )
            app.state.inference.warmup()
            created.append("inference")
        if app.state.cache is None:
            app.state.cache = PredictionCache.from_settings(settings)
            created.append("cache")
        MODEL_INFO.labels(model_version=app.state.model.version).set(1)
        app.state.ready = True
        logger.info("Startup complete: model_version=%s cache_enabled=%s", app.state.model.version, app.state.cache.enabled)
        yield
        app.state.ready = False
        if "inference" in created:
            app.state.inference.shutdown()
        if "cache" in created:
            await app.state.cache.close()

    app = FastAPI(
        title="Distributed AI Model Serving & Inference Platform",
        description="Demo inference service: FastAPI + PyTorch + Redis cache + Prometheus metrics.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model = model
    app.state.inference = inference
    app.state.cache = cache
    app.state.started_at = time.monotonic()
    app.state.ready = model is not None and inference is not None and cache is not None

    @app.middleware("http")
    async def observability(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error while processing request")
            response = JSONResponse(
                status_code=500,
                content={"error": "internal_error", "detail": "Internal server error.", "request_id": request_id},
            )
        duration = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = getattr(route, "path", None) or "unmatched"  # template path => low cardinality
        HTTP_REQUESTS.labels(request.method, endpoint, str(response.status_code)).inc()
        HTTP_LATENCY.labels(request.method, endpoint).observe(duration)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Served-By"] = settings.instance_name
        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 3),
                "cache": response.headers.get("x-cache"),
            },
        )
        request_id_var.reset(token)
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error, "detail": exc.message, "request_id": request.state.request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        # Build a safe summary by hand: pydantic's raw errors can echo large inputs or non-JSON objects.
        details = [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "detail": details, "request_id": request.state.request_id},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "http_error", "detail": str(exc.detail), "request_id": request.state.request_id},
        )

    for router in routers:
        app.include_router(router)
    return app


app = create_app()
