"""Exact navigation of autoregressive distributions in fixed-information steps."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Protocol, Sequence


class Cursor(Protocol):
    """Clonable causal model state consumed by :class:`Navigator`.

    ``predict`` must return the *complete* next-symbol distribution in the same
    index space accepted by ``observe``. No top-k/top-p truncation is allowed.
    """

    prefix: tuple[int, ...]
    ended: bool

    def clone(self) -> "Cursor": ...

    def predict(self) -> Sequence[float]: ...

    def observe(self, token: int) -> None: ...


@dataclass
class State:
    """One exact arithmetic-code region plus its causal model state."""

    cursor: Cursor
    lo: float = 0.0
    hi: float = 1.0
    actions: int = 0
    path_surprisal: float = 0.0

    def clone(self) -> "State":
        return State(
            cursor=self.cursor.clone(),
            lo=self.lo,
            hi=self.hi,
            actions=self.actions,
            path_surprisal=self.path_surprisal,
        )


@dataclass(frozen=True)
class Preview:
    """What one bucket selection would imply without mutating live state."""

    bucket: int
    forced: tuple[int, ...]
    representative: tuple[int, ...]


@dataclass(frozen=True)
class RankedDistribution:
    """A complete next-symbol distribution laid out in descending probability."""

    tokens: tuple[int, ...]
    probabilities: tuple[float, ...]
    edges: tuple[float, ...]


class Navigator:
    """K-ary arithmetic navigator over a complete autoregressive distribution.

    Every user action selects one of ``K`` equal-width subintervals of the
    unresolved arithmetic-code interval, so each action contributes exactly
    ``ln(K)`` nats (``log2(K)`` bits) of information.

    Tokens are committed only when *every* code point in the selected interval
    agrees on the same next token. This is the key completeness invariant: the
    navigator never silently approximates a bucket with a representative path.
    """

    def __init__(
        self,
        cursor: Cursor,
        *,
        choices: int = 5,
        preview_tokens: int = 8,
    ) -> None:
        if choices < 2:
            raise ValueError("choices must be >= 2")
        if preview_tokens < 0:
            raise ValueError("preview_tokens must be >= 0")
        self.choices = choices
        self.preview_tokens = preview_tokens
        self.state = State(cursor=cursor)

    @property
    def nats_per_action(self) -> float:
        return math.log(self.choices)

    @property
    def bits_per_action(self) -> float:
        return math.log2(self.choices)

    @property
    def supplied_nats(self) -> float:
        return self.state.actions * self.nats_per_action

    @property
    def supplied_bits(self) -> float:
        return self.state.actions * self.bits_per_action

    @staticmethod
    def rank(cursor: Cursor) -> RankedDistribution:
        raw = cursor.predict()
        n = len(raw)
        if n == 0:
            raise ValueError("predict() returned an empty distribution")

        probs = [float(raw[i]) for i in range(n)]
        if any((not math.isfinite(p)) or p < 0.0 for p in probs):
            raise ValueError("predict() returned negative or non-finite probability")

        total = math.fsum(probs)
        if total <= 0.0:
            raise ValueError("predict() returned zero total probability")
        probs = [p / total for p in probs]

        # Stable sort means equal-probability symbols retain token-id order.
        order = sorted(range(n), key=probs.__getitem__, reverse=True)
        ranked = [probs[token] for token in order]

        edges = [0.0]
        cumulative = 0.0
        for p in ranked:
            cumulative += p
            edges.append(cumulative)
        # Clamp accumulated floating-point drift at the exact endpoint.
        edges[-1] = 1.0

        return RankedDistribution(tuple(order), tuple(ranked), tuple(edges))

    @staticmethod
    def _child_index(edges: Sequence[float], x: float) -> int:
        # Arithmetic intervals are [lo, hi). Keep x in that half-open domain.
        x = min(max(x, 0.0), math.nextafter(1.0, 0.0))
        return bisect.bisect_right(edges, x) - 1

    def _drain_forced(self, state: State) -> tuple[State, tuple[int, ...]]:
        forced: list[int] = []

        while not state.cursor.ended:
            ranked = self.rank(state.cursor)
            right_probe = math.nextafter(state.hi, state.lo)
            left_i = self._child_index(ranked.edges, state.lo)
            right_i = self._child_index(ranked.edges, right_probe)

            # The current exact interval straddles multiple next-token cells.
            if left_i != right_i:
                break

            token = ranked.tokens[left_i]
            p = ranked.probabilities[left_i]
            a = ranked.edges[left_i]
            b = ranked.edges[left_i + 1]
            if p <= 0.0 or b <= a:
                raise RuntimeError("selected a zero-width model interval")

            state.lo = (state.lo - a) / p
            state.hi = (state.hi - a) / p
            state.lo = min(max(state.lo, 0.0), 1.0)
            state.hi = min(max(state.hi, 0.0), 1.0)
            state.path_surprisal -= math.log(p)
            state.cursor.observe(token)
            forced.append(token)

        return state, tuple(forced)

    def _narrow(self, state: State, bucket: int) -> None:
        if not 0 <= bucket < self.choices:
            raise ValueError(f"bucket must be in [0, {self.choices})")
        width = (state.hi - state.lo) / self.choices
        old_lo = state.lo
        state.lo = old_lo + bucket * width
        state.hi = old_lo + (bucket + 1) * width
        state.actions += 1

    def choose(self, bucket: int) -> tuple[int, ...]:
        """Select one exact equal-information bucket and commit forced tokens."""
        if self.state.cursor.ended:
            return ()
        self._narrow(self.state, bucket)
        self.state, forced = self._drain_forced(self.state)
        return forced

    def decode_point(
        self,
        cursor: Cursor,
        u: float,
        *,
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        """Decode one representative arithmetic-code point.

        This is intended for display only. It does not represent the entire
        selected bucket and must never be treated as committed state.
        """
        limit = self.preview_tokens if max_tokens is None else max_tokens
        out: list[int] = []

        for _ in range(limit):
            if cursor.ended:
                break
            ranked = self.rank(cursor)
            i = self._child_index(ranked.edges, u)
            token = ranked.tokens[i]
            p = ranked.probabilities[i]
            a = ranked.edges[i]
            if p <= 0.0:
                break

            out.append(token)
            cursor.observe(token)
            u = (u - a) / p
            u = min(max(u, 0.0), math.nextafter(1.0, 0.0))

        return tuple(out)

    def preview(self, bucket: int) -> Preview:
        """Preview one bucket without mutating the live navigator."""
        state = self.state.clone()
        self._narrow(state, bucket)
        state, forced = self._drain_forced(state)

        if state.cursor.ended or self.preview_tokens == 0:
            representative: tuple[int, ...] = ()
        else:
            midpoint = (state.lo + state.hi) / 2.0
            representative = self.decode_point(
                state.cursor.clone(), midpoint, max_tokens=self.preview_tokens
            )

        return Preview(bucket=bucket, forced=forced, representative=representative)
