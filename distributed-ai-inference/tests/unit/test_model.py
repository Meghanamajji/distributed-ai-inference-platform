import pytest
import torch

from app.model import MLP, load_model, save_artifact


def test_model_loads_in_eval_mode(loaded_model):
    assert loaded_model.module.training is False
    assert loaded_model.input_dim == 16
    assert all(not p.requires_grad for p in loaded_model.module.parameters())
    assert loaded_model.version.startswith("test-")


def test_inference_returns_probabilities(loaded_model):
    with torch.inference_mode():
        probs = torch.softmax(loaded_model.module(torch.zeros(1, 16)), dim=1)[0]
    assert probs.shape == (3,)
    assert pytest.approx(probs.sum().item(), abs=1e-5) == 1.0


def test_inference_is_deterministic(loaded_model):
    x = torch.linspace(-1, 1, 16).unsqueeze(0)
    with torch.inference_mode():
        assert torch.equal(loaded_model.module(x), loaded_model.module(x))


def test_missing_artifact_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.train"):
        load_model(tmp_path / "nope.pt")


def test_version_depends_on_weights(tmp_path):
    torch.manual_seed(1)
    save_artifact(MLP(16, 32, 3), tmp_path / "a.pt", base_version="v")
    torch.manual_seed(2)
    save_artifact(MLP(16, 32, 3), tmp_path / "b.pt", base_version="v")
    assert load_model(tmp_path / "a.pt").version != load_model(tmp_path / "b.pt").version
    assert load_model(tmp_path / "a.pt").version == load_model(tmp_path / "a.pt").version
