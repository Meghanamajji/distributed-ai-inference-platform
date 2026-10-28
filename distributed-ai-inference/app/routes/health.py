import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.cache import PredictionCache
from app.config import Settings
from app.dependencies import get_app_settings, get_cache
from app.schemas import HealthResponse

router = APIRouter(tags=["operations"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Liveness: the process is up and the event loop answers. Does not check dependencies."""
    return HealthResponse(
        status="ok",
        instance=settings.instance_name,
        uptime_seconds=round(time.monotonic() - request.app.state.started_at, 3),
    )


@router.get("/ready")
async def ready(
    request: Request,
    cache: PredictionCache = Depends(get_cache),
    settings: Settings = Depends(get_app_settings),
) -> JSONResponse:
    """Readiness: can this replica serve inference right now?

    - Model must be loaded (else 503).
    - Redis down + policy "bypass": still ready (predictions work, just uncached) -> reported as degraded.
    - Redis down + policy "fail": not ready (503), so a load balancer / Kubernetes stops sending traffic.
    """
    checks: dict[str, str] = {}
    model_ok = bool(getattr(request.app.state, "ready", False)) and request.app.state.model is not None
    checks["model"] = "ok" if model_ok else "not_loaded"

    cache_ok = True
    if not cache.enabled:
        checks["cache"] = "disabled"
    elif await cache.ping():
        checks["cache"] = "ok"
    elif settings.cache_failure_policy == "bypass":
        checks["cache"] = "unavailable (degraded: requests bypass the cache)"
    else:
        checks["cache"] = "unavailable"
        cache_ok = False

    is_ready = model_ok and cache_ok
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={"status": "ready" if is_ready else "not_ready", "checks": checks},
    )
