import pytest
import torch
from fakeredis import FakeAsyncRedis, FakeServer
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.inference import InferenceService
from app.main import create_app
from app.model import MLP, load_model, save_artifact
from tests.helpers import make_cache


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("model") / "model.pt"
    save_artifact(MLP(16, 32, 3), path, base_version="test")
    return path


@pytest.fixture(scope="session")
def loaded_model(model_path):
    return load_model(model_path)


@pytest.fixture
def settings(model_path):
    return Settings(_env_file=None, model_path=str(model_path), instance_name="test-instance", log_level="WARNING")


@pytest.fixture
def fake_redis():
    return FakeAsyncRedis(server=FakeServer(), decode_responses=True)


@pytest.fixture
def cache(fake_redis):
    return make_cache(fake_redis)


@pytest.fixture
def inference(loaded_model):
    svc = InferenceService(loaded_model, threads=2, timeout_seconds=2.0, max_pending=100)
    yield svc
    svc.shutdown()


@pytest.fixture
async def client(settings, loaded_model, cache, inference):
    app = create_app(settings, cache=cache, inference=inference, model=loaded_model)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def make_client(settings, loaded_model, inference):
    """Factory for tests that need a custom cache / inference service / settings."""

    async def _make(cache, inference_override=None, settings_override=None):
        app = create_app(settings_override or settings, cache=cache, inference=inference_override or inference, model=loaded_model)
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    return _make
