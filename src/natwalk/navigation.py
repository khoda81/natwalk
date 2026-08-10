"""Equal-information navigation layered over a committed model session."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .session import Checkpoint, Session
from .tree import Distribution


@dataclass(frozen=True, slots=True)
class State:
    lo: float = 0.0
    hi: float = 1.0
    actions: int = 0


@dataclass(frozen=True, slots=True)
class _HistoryEntry:
    session: Checkpoint
    state: State


class Navigation:
    """Own pending arithmetic information and user-action undo semantics."""

    def __init__(self, session: Session, *, choices: int = 2) -> None:
        if choices < 2:
            raise ValueError("choices must be >= 2")
        self.session = session
        self.choices = choices
        self.state = State()
        self._history: list[_HistoryEntry] = []

    @property
    def undo_depth(self) -> int:
        return len(self._history)

    @property
    def supplied_nats(self) -> float:
        return self.state.actions * math.log(self.choices)

    @property
    def supplied_bits(self) -> float:
        return self.state.actions * math.log2(self.choices)

    @staticmethod
    def _rank_at(distribution: Distribution, point: float) -> tuple[int, float]:
        edge = 0.0
        for rank, probability in enumerate(distribution.probabilities):
            next_edge = edge + probability
            if point < next_edge:
                return rank, edge
            edge = next_edge
        raise RuntimeError("distribution does not cover arithmetic point")

    def _push_history(self) -> None:
        self._history.append(_HistoryEntry(self.session.checkpoint(), self.state))

    def choose(self, bucket: int) -> tuple[int, ...]:
        """Narrow one equal-width bucket and commit every newly forced token."""
        if not 0 <= bucket < self.choices:
            raise ValueError(f"bucket must be in [0, {self.choices})")
        if not self.session.distribution().tokens:
            return ()

        self._push_history()
        width = (self.state.hi - self.state.lo) / self.choices
        lo = self.state.lo + bucket * width
        hi = lo + width
        self.state = State(lo=lo, hi=hi, actions=self.state.actions + 1)

        forced: list[int] = []

        def forced_tokens():
            while True:
                distribution = self.session.distribution()
                if not distribution.tokens:
                    return

                left_rank, left_edge = self._rank_at(distribution, self.state.lo)
                right_probe = math.nextafter(self.state.hi, self.state.lo)
                right_rank, _ = self._rank_at(distribution, right_probe)
                if left_rank != right_rank:
                    return

                probability = distribution.probabilities[left_rank]
                token = distribution.tokens[left_rank]
                self.state = State(
                    lo=(self.state.lo - left_edge) / probability,
                    hi=(self.state.hi - left_edge) / probability,
                    actions=self.state.actions,
                )
                forced.append(token)
                yield token

        self.session.commit(forced_tokens())
        return tuple(forced)

    def accept(self, tokens: tuple[int, ...]) -> None:
        """Commit an explicit path as one undoable action and clear pending bits."""
        if not tokens:
            return
        self._push_history()
        self.session.commit(tokens)
        self.state = State(actions=self.state.actions)

    def undo(self) -> bool:
        if not self._history:
            return False
        previous = self._history.pop()
        self.session.restore(previous.session)
        self.state = previous.state
        return True
