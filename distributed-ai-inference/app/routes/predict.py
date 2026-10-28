import time

from fastapi import APIRouter, Depends, Request, Response

from app import cache as cache_module
from app.cache import PredictionCache
from app.config import Settings
from app.dependencies import get_app_settings, get_cache, get_inference, get_model
from app.errors import InvalidInputError
from app.inference import InferenceService
from app.model import LoadedModel
from app.schemas import Prediction, PredictRequest, PredictResponse

router = APIRouter(tags=["prediction"])

BYPASS = "BYPASS"  # X-Cache value when the cache was intentionally not used


def _check_dimensions(features: list[float], model: LoadedModel) -> None:
    if len(features) != model.input_dim:
        raise InvalidInputError(f"Expected exactly {model.input_dim} features, got {len(features)}.")


def _build_response(
    request: Request, settings: Settings, model: LoadedModel, prediction: dict, cached: bool, started: float
) -> PredictResponse:
    return PredictResponse(
        request_id=request.state.request_id,
        model_version=model.version,
        prediction=Prediction(**prediction),
        cached=cached,
        instance=settings.instance_name,
        server_latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    request: Request,
    response: Response,
    model: LoadedModel = Depends(get_model),
    settings: Settings = Depends(get_app_settings),
    cache: PredictionCache = Depends(get_cache),
    inference: InferenceService = Depends(get_inference),
) -> PredictResponse:
    """Cache-enabled prediction: Redis lookup -> (on miss) model inference -> store in Redis."""
    started = time.perf_counter()
    _check_dimensions(body.features, model)

    if not cache.enabled:
        prediction = await inference.predict(body.features)
        response.headers["X-Cache"] = BYPASS
        return _build_response(request, settings, model, prediction, False, started)

    status, prediction = await cache.get(body.features, model.version)
    if status == cache_module.HIT:
        response.headers["X-Cache"] = cache_module.HIT
        return _build_response(request, settings, model, prediction, True, started)

    prediction = await inference.predict(body.features)
    if status == cache_module.MISS:  # only try to write if Redis answered the lookup
        await cache.set(body.features, model.version, prediction)
    response.headers["X-Cache"] = status
    return _build_response(request, settings, model, prediction, False, started)


@router.post("/predict/baseline", response_model=PredictResponse)
async def predict_baseline(
    body: PredictRequest,
    request: Request,
    response: Response,
    model: LoadedModel = Depends(get_model),
    settings: Settings = Depends(get_app_settings),
    inference: InferenceService = Depends(get_inference),
) -> PredictResponse:
    """Same validation + same model + same response shape as /predict, but never touches Redis.
    Exists so benchmarks compare cache vs. no cache on an otherwise identical code path."""
    started = time.perf_counter()
    _check_dimensions(body.features, model)
    prediction = await inference.predict(body.features)
    response.headers["X-Cache"] = BYPASS
    return _build_response(request, settings, model, prediction, False, started)
