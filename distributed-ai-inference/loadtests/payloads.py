"""Deterministic request payloads shared by the Locust file and the warm-up/benchmark scripts.

Keeping this in ONE place guarantees the cache warm-up uses exactly the same inputs Locust will send.
"""
import random


def hot_set(size: int, dim: int = 16, seed: int = 1234) -> list[list[float]]:
    """`size` fixed feature vectors. Sending only these repeatedly = a 'repeated inputs' workload."""
    rng = random.Random(seed)
    return [[round(rng.uniform(-3, 3), 4) for _ in range(dim)] for _ in range(size)]


def varied_vector(index: int, dim: int = 16, seed: str = "0") -> list[float]:
    """The index-th vector of a stream that practically never repeats (for a given seed)."""
    rng = random.Random(f"{seed}:{index}")
    return [round(rng.uniform(-3, 3), 6) for _ in range(dim)]
