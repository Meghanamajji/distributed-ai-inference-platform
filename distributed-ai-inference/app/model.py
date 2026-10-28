"""The PyTorch model, plus saving/loading of its artifact."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

CLASS_NAMES: tuple[str, ...] = ("class_a", "class_b", "class_c")


class MLP(nn.Module):
    """Small multilayer perceptron: fixed-size feature vector -> class logits."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class LoadedModel:
    module: MLP
    input_dim: int
    hidden_dim: int
    class_names: tuple[str, ...]
    version: str
    parameter_count: int
    path: str


def _weights_digest(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash of the weight *values* (not the file bytes), so identical weights => identical version."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode())
        digest.update(json.dumps(state_dict[name].detach().cpu().flatten().tolist()).encode())
    return digest.hexdigest()


def save_artifact(
    module: MLP,
    path: str | Path,
    *,
    base_version: str,
    class_names: Sequence[str] = CLASS_NAMES,
    training: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "state_dict": module.state_dict(),
            "input_dim": module.input_dim,
            "hidden_dim": module.hidden_dim,
            "class_names": list(class_names),
            "base_version": base_version,
            "training": training or {},
        },
        path,
    )


def load_model(path: str | Path) -> LoadedModel:
    """Load the artifact once (at startup), put it in eval mode and freeze it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. Create it first with: python -m model.train"
        )
    # weights_only=True refuses to unpickle arbitrary code - important for safety.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    class_names = tuple(payload["class_names"])
    module = MLP(payload["input_dim"], payload["hidden_dim"], len(class_names))
    module.load_state_dict(payload["state_dict"])
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    version = f"{payload['base_version']}-{_weights_digest(module.state_dict())[:8]}"
    return LoadedModel(
        module=module,
        input_dim=payload["input_dim"],
        hidden_dim=payload["hidden_dim"],
        class_names=class_names,
        version=version,
        parameter_count=sum(p.numel() for p in module.parameters()),
        path=str(path),
    )
