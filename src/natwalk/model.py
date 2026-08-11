"""Minimal autoregressive model boundary used by the rewritten core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .tree import Distribution, RankedDistribution


class Cursor(Protocol):
    """One rewindable causal model state.

    ``predict`` returns either a complete normalized next-symbol distribution in
    token-index space or a backend-native probability-ranked distribution. An
    empty distribution is terminal. Natwalk never normalizes model output.
    """

    def predict(self) -> Sequence[float] | RankedDistribution: ...

    def observe(self, token: int) -> None: ...

    def checkpoint(self) -> object: ...

    def restore(self, checkpoint: object) -> None: ...


def rank(probabilities: Sequence[float] | RankedDistribution) -> RankedDistribution:
    """Return a probability-ranked view without discarding backend-native storage."""
    if isinstance(probabilities, RankedDistribution):
        return probabilities

    probs = tuple(float(probability) for probability in probabilities)
    order = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
    return Distribution(
        tokens=tuple(order),
        probabilities=tuple(probs[token] for token in order),
    )
