import torch
from fastapi import APIRouter, Depends

from app.cache import PredictionCache
from app.config import Settings
from app.dependencies import get_app_settings, get_cache, get_model
from app.model import LoadedModel
from app.schemas import InferenceInfo, ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(
    model: LoadedModel = Depends(get_model),
    settings: Settings = Depends(get_app_settings),
    cache: PredictionCache = Depends(get_cache),
) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=model.version,
        architecture="MLP: Linear-ReLU-Linear-ReLU-Linear",
        input_dim=model.input_dim,
        hidden_dim=model.hidden_dim,
        num_classes=len(model.class_names),
        class_names=list(model.class_names),
        parameter_count=model.parameter_count,
        inference=InferenceInfo(
            device="cpu",
            inference_threads=settings.inference_threads,
            torch_intra_op_threads=torch.get_num_threads(),
            timeout_seconds=settings.inference_timeout_seconds,
            extra_forward_passes=settings.extra_forward_passes,
            cache_enabled=cache.enabled,
            cache_ttl_seconds=cache.ttl_seconds,
            cache_failure_policy=cache.failure_policy,
        ),
        note="Demo model trained on synthetic data to exercise the serving infrastructure; "
        "it makes no claim of real-world predictive accuracy.",
    )
