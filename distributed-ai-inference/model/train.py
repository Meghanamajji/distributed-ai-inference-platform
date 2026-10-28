"""Reproducible training on a synthetic dataset.

Run from the project root:   python -m model.train

The data are three Gaussian clusters in 16 dimensions. The point of this model is to give the
serving infrastructure something real to run - NOT to claim real-world predictive accuracy.
"""
from __future__ import annotations

import argparse

import torch
from torch import nn

from app.model import CLASS_NAMES, MLP, save_artifact

INPUT_DIM = 16
HIDDEN_DIM = 128


def make_dataset(seed: int, samples_per_class: int, input_dim: int, num_classes: int):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, input_dim, generator=g) * 1.5
    x = torch.cat(
        [centers[c] + torch.randn(samples_per_class, input_dim, generator=g) for c in range(num_classes)]
    )
    y = torch.arange(num_classes).repeat_interleave(samples_per_class)
    perm = torch.randperm(len(x), generator=g)
    return x[perm], y[perm], g


def train(seed: int, epochs: int, samples_per_class: int, lr: float = 1e-2, batch_size: int = 64):
    torch.set_num_threads(1)  # single thread => identical float summation order => reproducible
    torch.manual_seed(seed)
    num_classes = len(CLASS_NAMES)
    x, y, g = make_dataset(seed, samples_per_class, INPUT_DIM, num_classes)
    split = int(0.8 * len(x))
    x_tr, y_tr, x_va, y_va = x[:split], y[:split], x[split:], y[split:]

    module = MLP(INPUT_DIM, HIDDEN_DIM, num_classes)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        module.train()
        order = torch.randperm(len(x_tr), generator=g)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(module(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()

    module.eval()
    with torch.no_grad():
        val_acc = (module(x_va).argmax(dim=1) == y_va).float().mean().item()
    return module, {"seed": seed, "epochs": epochs, "val_accuracy": round(val_acc, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the demo MLP on synthetic data.")
    parser.add_argument("--output", default="model/artifacts/model.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-class", type=int, default=600)
    args = parser.parse_args()

    module, info = train(args.seed, args.epochs, args.samples_per_class)
    save_artifact(module, args.output, base_version="1.0.0", training=info)
    print(f"Saved model to {args.output}")
    print(f"Synthetic validation accuracy: {info['val_accuracy']:.2%} (demo data only)")


if __name__ == "__main__":
    main()
