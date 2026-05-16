"""seed_everything determinism."""
from __future__ import annotations

import random

import numpy as np

from insta_influencer.core.seed import seed_everything


def test_seed_python() -> None:
    seed_everything(42)
    a = random.random()
    seed_everything(42)
    b = random.random()
    assert a == b


def test_seed_numpy() -> None:
    seed_everything(42)
    a = np.random.rand(4)
    seed_everything(42)
    b = np.random.rand(4)
    assert np.array_equal(a, b)
