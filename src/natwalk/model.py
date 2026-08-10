"""Minimal autoregressive model boundary used by the rewritten core."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from .tree import Distribution


class Cursor(Protocol):
    """One rewindable causal model state.

    ``predict`` returns the complete normalized next-symbol distribution in the
    token index space accepted by ``observe``. An empty distribution is terminal.
    """

    def predict(self) -> Sequence[float]: ...

    def observe(self, token: int) -> None: ...

    def checkpoint(self) -> object: ...

    def restore(self, checkpoint: object) -> None: ...


def rank(probabilities: Sequence[float]) -> Distribution:
    """Convert a model distribution to nat-sorted tree representation."""
    probs = tuple(float(probability) for probability in probabilities)
    order = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
    return Distribution(
        tokens=tuple(order),
        nats=tuple(-math.log(probs[token]) if probs[token] != 0.0 else math.inf for token in order),
    )
