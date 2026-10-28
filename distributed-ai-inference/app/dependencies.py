"""FastAPI dependencies: fetch long-lived objects from app.state (created once at startup)."""
from fastapi import Request

from app.cache import PredictionCache
from app.config import Settings
from app.inference import InferenceService
from app.model import LoadedModel


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_model(request: Request) -> LoadedModel:
    return request.app.state.model


def get_inference(request: Request) -> InferenceService:
    return request.app.state.inference


def get_cache(request: Request) -> PredictionCache:
    return request.app.state.cache
