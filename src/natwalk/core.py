"""Exact navigation and information-priority exploration of autoregressive distributions."""

from __future__ import annotations

import bisect
import heapq
import math
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol


class Cursor(Protocol):
    """Causal model state consumed by :class:`Navigator`.

    ``predict`` must return the complete next-symbol distribution in the same
    index space accepted by ``observe``. No top-k/top-p truncation is allowed.
    """

    prefix: tuple[int, ...]
    ended: bool

    def clone(self) -> Cursor: ...

    def predict(self) -> Sequence[float]: ...

    def observe(self, token: int) -> None: ...


class RewindableCursor(Cursor, Protocol):
    """Optional fast branching API for backends with large mutable caches."""

    def checkpoint(self) -> object: ...

    def restore(self, checkpoint: object) -> None: ...


@dataclass
class State:
    """One exact arithmetic-code region plus its causal model state."""

    cursor: Cursor
    lo: float = 0.0
    hi: float = 1.0
    actions: int = 0
    path_surprisal: float = 0.0

    def clone(self) -> State:
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


@dataclass(frozen=True)
class NavigationSnapshot:
    """Immutable public view of the live navigation state."""

    prefix: tuple[int, ...]
    ended: bool
    lo: float
    hi: float
    actions: int
    path_surprisal: float
    undo_depth: int


@dataclass(frozen=True)
class GreedySuggestion:
    """Greedy continuation whose model surprisal does not exceed a budget."""

    tokens: tuple[int, ...]
    nats: float
    bits: float
    next_token_nats: float | None
    complete: bool

    @property
    def next_token_bits(self) -> float | None:
        if self.next_token_nats is None:
            return None
        return self.next_token_nats / math.log(2)


@dataclass
class _HistoryEntry:
    rewindable: bool
    cursor_state: object
    lo: float
    hi: float
    actions: int
    path_surprisal: float


class Navigator:
    """K-ary arithmetic navigator over a complete autoregressive distribution."""

    def __init__(
        self,
        cursor: Cursor,
        *,
        choices: int = 2,
        preview_tokens: int = 8,
    ) -> None:
        if choices < 2:
            raise ValueError("choices must be >= 2")
        if preview_tokens < 0:
            raise ValueError("preview_tokens must be >= 0")
        self.choices = choices
        self.preview_tokens = preview_tokens
        self.state = State(cursor=cursor)
        self._history: list[_HistoryEntry] = []

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

    @property
    def undo_depth(self) -> int:
        return len(self._history)

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

        order = sorted(range(n), key=probs.__getitem__, reverse=True)
        ranked = [probs[token] for token in order]

        edges = [0.0]
        cumulative = 0.0
        for p in ranked:
            cumulative += p
            edges.append(cumulative)
        edges[-1] = 1.0

        return RankedDistribution(tuple(order), tuple(ranked), tuple(edges))

    @staticmethod
    def _child_index(edges: Sequence[float], x: float) -> int:
        x = min(max(x, 0.0), math.nextafter(1.0, 0.0))
        return bisect.bisect_right(edges, x) - 1

    @staticmethod
    def _can_rewind(cursor: Cursor) -> bool:
        return callable(getattr(cursor, "checkpoint", None)) and callable(
            getattr(cursor, "restore", None)
        )

    @contextmanager
    def temporary_cursor(self, cursor: Cursor | None = None) -> Iterator[Cursor]:
        """Yield a branchable cursor and restore the original state afterwards."""
        target = self.state.cursor if cursor is None else cursor
        if self._can_rewind(target):
            checkpoint = target.checkpoint()  # type: ignore[attr-defined]
            try:
                yield target
            finally:
                target.restore(checkpoint)  # type: ignore[attr-defined]
        else:
            yield target.clone()

    @contextmanager
    def _temporary_state(self) -> Iterator[State]:
        live = self.state
        cursor = live.cursor

        if self._can_rewind(cursor):
            checkpoint = cursor.checkpoint()  # type: ignore[attr-defined]
            saved = (live.lo, live.hi, live.actions, live.path_surprisal)
            try:
                yield live
            finally:
                cursor.restore(checkpoint)  # type: ignore[attr-defined]
                live.lo, live.hi, live.actions, live.path_surprisal = saved
        else:
            self.state = live.clone()
            try:
                yield self.state
            finally:
                self.state = live

    def _push_history(self) -> None:
        cursor = self.state.cursor
        if self._can_rewind(cursor):
            rewindable = True
            cursor_state = cursor.checkpoint()  # type: ignore[attr-defined]
        else:
            rewindable = False
            cursor_state = cursor.clone()
        self._history.append(
            _HistoryEntry(
                rewindable=rewindable,
                cursor_state=cursor_state,
                lo=self.state.lo,
                hi=self.state.hi,
                actions=self.state.actions,
                path_surprisal=self.state.path_surprisal,
            )
        )

    def undo(self) -> bool:
        """Undo one user action (binary/K-ary choice or explicit greedy accept)."""
        if not self._history:
            return False
        item = self._history.pop()
        if item.rewindable:
            self.state.cursor.restore(item.cursor_state)  # type: ignore[attr-defined]
        else:
            self.state.cursor = item.cursor_state  # type: ignore[assignment]
        self.state.lo = item.lo
        self.state.hi = item.hi
        self.state.actions = item.actions
        self.state.path_surprisal = item.path_surprisal
        return True

    def snapshot(self) -> NavigationSnapshot:
        state = self.state
        return NavigationSnapshot(
            prefix=tuple(state.cursor.prefix),
            ended=state.cursor.ended,
            lo=state.lo,
            hi=state.hi,
            actions=state.actions,
            path_surprisal=state.path_surprisal,
            undo_depth=len(self._history),
        )

    def _drain_forced(self, state: State) -> tuple[State, tuple[int, ...]]:
        forced: list[int] = []

        while not state.cursor.ended:
            ranked = self.rank(state.cursor)
            right_probe = math.nextafter(state.hi, state.lo)
            left_i = self._child_index(ranked.edges, state.lo)
            right_i = self._child_index(ranked.edges, right_probe)
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
        self._push_history()
        self._narrow(self.state, bucket)
        self.state, forced = self._drain_forced(self.state)
        return forced

    def greedy_suggestion(
        self,
        *,
        max_bits: float,
        max_tokens: int = 256,
    ) -> GreedySuggestion:
        """Return the longest greedy continuation costing at most ``max_bits``.

        The suggestion is model-greedy from the currently committed prefix.
        It intentionally does not reinterpret a partially narrowed arithmetic
        interval. Accepting it is an explicit choice and resets that pending
        interval to ``[0, 1)``.
        """
        if max_bits < 0:
            raise ValueError("max_bits must be >= 0")
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")

        budget_nats = max_bits * math.log(2)
        tokens: list[int] = []
        used = 0.0
        next_cost: float | None = None
        complete = False

        with self.temporary_cursor() as cursor:
            for _ in range(max_tokens):
                if cursor.ended:
                    complete = True
                    break
                ranked = self.rank(cursor)
                token = ranked.tokens[0]
                p = ranked.probabilities[0]
                cost = -math.log(p)
                if used + cost > budget_nats + 1e-12:
                    next_cost = cost
                    complete = True
                    break
                tokens.append(token)
                used += cost
                cursor.observe(token)
            else:
                complete = False

        return GreedySuggestion(
            tokens=tuple(tokens),
            nats=used,
            bits=used / math.log(2),
            next_token_nats=next_cost,
            complete=complete,
        )

    def accept_greedy(
        self,
        *,
        max_bits: float,
        max_tokens: int = 256,
    ) -> GreedySuggestion:
        """Commit the current greedy suggestion as one undoable user action."""
        suggestion = self.greedy_suggestion(max_bits=max_bits, max_tokens=max_tokens)
        if not suggestion.tokens:
            return suggestion

        self._push_history()
        for token in suggestion.tokens:
            probs = self.state.cursor.predict()
            p = float(probs[token])
            if p <= 0.0:
                raise RuntimeError("greedy token became impossible while committing")
            self.state.path_surprisal -= math.log(p)
            self.state.cursor.observe(token)

        self.state.lo = 0.0
        self.state.hi = 1.0
        return suggestion

    def decode_point(
        self,
        cursor: Cursor,
        u: float,
        *,
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        """Decode one representative arithmetic-code point for display only."""
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

    def _preview_current(self, bucket: int) -> Preview:
        self._narrow(self.state, bucket)
        self.state, forced = self._drain_forced(self.state)

        if self.state.cursor.ended or self.preview_tokens == 0:
            representative: tuple[int, ...] = ()
        else:
            midpoint = (self.state.lo + self.state.hi) / 2.0
            with self.temporary_cursor(self.state.cursor) as cursor:
                representative = self.decode_point(
                    cursor,
                    midpoint,
                    max_tokens=self.preview_tokens,
                )

        return Preview(bucket=bucket, forced=forced, representative=representative)

    def preview(self, bucket: int) -> Preview:
        """Preview one bucket without mutating the live navigator."""
        with self._temporary_state():
            return self._preview_current(bucket)


@dataclass
class _TokenNode:
    path: tuple[int, ...]
    token: int | None
    parent: tuple[int, ...] | None
    rank: int
    edge_nats: float
    path_nats: float
    lo: float
    hi: float
    expanded: bool = False
    ended: bool = False
    ranked: RankedDistribution | None = None
    materialized_ranks: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class TreeEntry:
    """One renderable token-tree row."""

    path: tuple[int, ...]
    depth: int
    token: int | None
    edge_nats: float
    path_nats: float
    expanded: bool
    is_ellipsis: bool = False
    hidden_count: int = 0
    hidden_nats: float | None = None


@dataclass(frozen=True)
class ExplorerStats:
    """Observable state of :class:`TokenTreeExplorer`."""

    nodes: int
    expanded: int
    frontier: int
    computing: bool
    generation: int
    saturated: bool


class TokenTreeExplorer:
    """Background uniform-cost search over the model's token-prefix tree.

    Edge cost is token surprisal ``-ln p(token | prefix)``. Therefore the heap
    is Dijkstra / uniform-cost search in information distance, not token depth.

    A model call expands one token prefix and exposes its complete ranked
    distribution. Children are then materialized lazily: only the cheapest
    unseen sibling of an expanded parent is placed on the global heap. Popping
    it exposes the next sibling. This is exactly equivalent to inserting every
    child into the heap while avoiding a vocabulary-sized heap blow-up.

    Search has no depth limit. ``max_nodes`` is only a resource cap.
    """

    _EXPAND = 0
    _CHILD = 1

    def __init__(
        self,
        navigator: Navigator,
        *,
        max_nodes: int = 10_000,
        autostart: bool = True,
    ) -> None:
        if max_nodes == 1:
            raise ValueError("max_nodes must be 0 (unlimited) or >= 2")
        if max_nodes < 0:
            raise ValueError("max_nodes must be >= 0")

        self.navigator = navigator
        self.max_nodes = max_nodes

        self._condition = threading.Condition()
        self._compute_lock = threading.Lock()
        self._heap: list[tuple[float, int, int, int, tuple[int, ...], int]] = []
        self._serial = 0
        self._generation = 0
        self._expanded_count = 0
        self._computing = False
        self._saturated = False
        self._stop = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

        self._root_prefix = tuple(navigator.state.cursor.prefix)
        self._snapshot = navigator.snapshot()
        self._nodes: dict[tuple[int, ...], _TokenNode] = {}
        with self._condition:
            self._reset_tree_locked()

        if autostart:
            self.start()

    def __enter__(self) -> TokenTreeExplorer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def choices(self) -> int:
        return self.navigator.choices

    @property
    def nats_per_action(self) -> float:
        return self.navigator.nats_per_action

    @property
    def bits_per_action(self) -> float:
        return self.navigator.bits_per_action

    @property
    def snapshot(self) -> NavigationSnapshot:
        with self._condition:
            return self._snapshot

    @property
    def supplied_nats(self) -> float:
        return self.navigator.supplied_nats

    @property
    def supplied_bits(self) -> float:
        return self.navigator.supplied_bits

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._stop:
                raise RuntimeError("cannot restart a closed TokenTreeExplorer")
            self._thread = threading.Thread(
                target=self._worker,
                name="natwalk-dijkstra",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _reset_tree_locked(self) -> None:
        self._generation += 1
        self._heap.clear()
        self._nodes.clear()
        self._saturated = False
        self._root_prefix = tuple(self.navigator.state.cursor.prefix)
        root = _TokenNode(
            path=(),
            token=None,
            parent=None,
            rank=-1,
            edge_nats=0.0,
            path_nats=0.0,
            lo=0.0,
            hi=1.0,
        )
        self._nodes[()] = root
        self._push_expand_locked(root)
        self._condition.notify_all()

    def _active_interval_locked(self) -> tuple[float, float]:
        return self.navigator.state.lo, self.navigator.state.hi

    @staticmethod
    def _intersection(
        lo: float,
        hi: float,
        active_lo: float,
        active_hi: float,
    ) -> float:
        return max(0.0, min(hi, active_hi) - max(lo, active_lo))

    def _priority_for_interval_locked(self, lo: float, hi: float) -> float:
        active_lo, active_hi = self._active_interval_locked()
        overlap = self._intersection(lo, hi, active_lo, active_hi)
        if overlap <= 0.0:
            return math.inf
        active_width = active_hi - active_lo
        if active_width <= 0.0:
            return math.inf
        return -math.log(overlap / active_width)

    def _node_relevant_locked(self, node: _TokenNode) -> bool:
        active_lo, active_hi = self._active_interval_locked()
        return self._intersection(node.lo, node.hi, active_lo, active_hi) > 0.0

    def _push_event_locked(
        self,
        priority: float,
        kind: int,
        path: tuple[int, ...],
        rank: int = -1,
    ) -> None:
        if not math.isfinite(priority):
            return
        self._serial += 1
        heapq.heappush(
            self._heap,
            (priority, kind, self._serial, self._generation, path, rank),
        )

    def _push_expand_locked(self, node: _TokenNode) -> None:
        if node.expanded or node.ended or not self._node_relevant_locked(node):
            return
        self._push_event_locked(
            self._priority_for_interval_locked(node.lo, node.hi),
            self._EXPAND,
            node.path,
        )

    def _child_interval(
        self,
        parent: _TokenNode,
        rank: int,
    ) -> tuple[float, float]:
        assert parent.ranked is not None
        width = parent.hi - parent.lo
        return (
            parent.lo + width * parent.ranked.edges[rank],
            parent.lo + width * parent.ranked.edges[rank + 1],
        )

    def _next_unmaterialized_relevant_rank_locked(
        self,
        parent: _TokenNode,
        after_rank: int = -1,
    ) -> int | None:
        ranked = parent.ranked
        if ranked is None:
            return None
        for rank in range(after_rank + 1, len(ranked.tokens)):
            if rank in parent.materialized_ranks:
                continue
            lo, hi = self._child_interval(parent, rank)
            if self._priority_for_interval_locked(lo, hi) < math.inf:
                return rank
        return None

    def _push_next_child_locked(
        self,
        parent: _TokenNode,
        after_rank: int = -1,
    ) -> None:
        if parent.ranked is None:
            return
        rank = self._next_unmaterialized_relevant_rank_locked(parent, after_rank)
        if rank is None:
            return
        lo, hi = self._child_interval(parent, rank)
        self._push_event_locked(
            self._priority_for_interval_locked(lo, hi),
            self._CHILD,
            parent.path,
            rank,
        )

    def _rebuild_frontier_locked(self) -> None:
        self._generation += 1
        self._heap.clear()
        self._saturated = self.max_nodes > 0 and len(self._nodes) >= self.max_nodes

        for node in self._nodes.values():
            if not self._node_relevant_locked(node):
                continue
            if not node.expanded and not node.ended:
                self._push_expand_locked(node)
            elif node.expanded:
                self._push_next_child_locked(node)
        self._condition.notify_all()

    def _predict_path(self, path: tuple[int, ...]) -> tuple[bool, RankedDistribution | None]:
        with self.navigator.temporary_cursor() as cursor:
            for token in path:
                if cursor.ended:
                    return True, None
                cursor.observe(token)
            if cursor.ended:
                return True, None
            return False, self.navigator.rank(cursor)

    def _materialize_child_locked(
        self,
        parent: _TokenNode,
        rank: int,
    ) -> _TokenNode | None:
        if parent.ranked is None or rank in parent.materialized_ranks:
            return None
        if self.max_nodes > 0 and len(self._nodes) >= self.max_nodes:
            self._saturated = True
            return None

        token = parent.ranked.tokens[rank]
        p = parent.ranked.probabilities[rank]
        if p <= 0.0:
            parent.materialized_ranks.add(rank)
            return None

        lo, hi = self._child_interval(parent, rank)
        path = (*parent.path, token)
        node = self._nodes.get(path)
        if node is None:
            node = _TokenNode(
                path=path,
                token=token,
                parent=parent.path,
                rank=rank,
                edge_nats=-math.log(p),
                path_nats=parent.path_nats - math.log(p),
                lo=lo,
                hi=hi,
            )
            self._nodes[path] = node
        parent.materialized_ranks.add(rank)
        return node

    def _work_once(self) -> bool:
        with self._condition:
            self._raise_worker_error_locked()
            while self._heap:
                priority, kind, _serial, generation, path, rank = heapq.heappop(self._heap)
                if generation != self._generation:
                    continue
                if not math.isfinite(priority):
                    continue
                break
            else:
                return False

            if kind == self._EXPAND:
                node = self._nodes.get(path)
                if node is None or node.expanded or node.ended or not self._node_relevant_locked(node):
                    return True
                self._computing = True
            else:
                parent = self._nodes.get(path)
                if parent is None or parent.ranked is None:
                    return True
                node = None

        if kind == self._CHILD:
            with self._condition:
                parent = self._nodes.get(path)
                if parent is None or parent.ranked is None:
                    return True
                child = self._materialize_child_locked(parent, rank)
                if child is not None:
                    self._push_expand_locked(child)
                self._push_next_child_locked(parent, after_rank=rank)
                self._condition.notify_all()
            return True

        error: BaseException | None = None
        ended = False
        ranked: RankedDistribution | None = None
        try:
            with self._compute_lock:
                with self._condition:
                    stale = generation != self._generation or self._stop
                if not stale:
                    ended, ranked = self._predict_path(path)
        except BaseException as exc:
            error = exc

        with self._condition:
            self._computing = False
            node = self._nodes.get(path)
            if (
                error is None
                and node is not None
                and generation == self._generation
                and not self._stop
            ):
                node.ended = ended
                node.ranked = ranked
                node.expanded = True
                self._expanded_count += 1
                if ranked is not None:
                    self._push_next_child_locked(node)
            self._condition.notify_all()

            if error is not None:
                self._error = error
                self._stop = True
                self._condition.notify_all()
                return False
        return True

    def step(self) -> bool:
        """Synchronously process one Dijkstra frontier event."""
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("cannot step while background worker is running")
        return self._work_once()

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._heap and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
            if not self._work_once():
                with self._condition:
                    if self._stop:
                        return
                    self._condition.wait(0.05)

    def _raise_worker_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("natwalk background worker failed") from self._error

    def choose(self, bucket: int) -> tuple[int, ...]:
        """Commit one exact arithmetic bucket and retarget background search."""
        with self._compute_lock:
            with self._condition:
                self._raise_worker_error_locked()
                old_prefix = tuple(self.navigator.state.cursor.prefix)
            forced = self.navigator.choose(bucket)
            with self._condition:
                self._snapshot = self.navigator.snapshot()
                new_prefix = tuple(self.navigator.state.cursor.prefix)
                if new_prefix != old_prefix:
                    self._reset_tree_locked()
                else:
                    self._rebuild_frontier_locked()
        return forced

    def accept_greedy(
        self,
        *,
        max_bits: float,
        max_tokens: int = 256,
    ) -> GreedySuggestion:
        """Accept a budgeted greedy continuation and re-root the search tree."""
        with self._compute_lock:
            with self._condition:
                self._raise_worker_error_locked()
            suggestion = self.navigator.accept_greedy(
                max_bits=max_bits,
                max_tokens=max_tokens,
            )
            if suggestion.tokens:
                with self._condition:
                    self._snapshot = self.navigator.snapshot()
                    self._reset_tree_locked()
        return suggestion

    def undo(self) -> bool:
        """Undo one navigator action and re-root the search tree."""
        with self._compute_lock:
            with self._condition:
                self._raise_worker_error_locked()
            changed = self.navigator.undo()
            if changed:
                with self._condition:
                    self._snapshot = self.navigator.snapshot()
                    self._reset_tree_locked()
        return changed

    def exact_greedy_suggestion(
        self,
        *,
        max_bits: float,
        max_tokens: int = 256,
    ) -> GreedySuggestion:
        """Compute a complete budgeted greedy suggestion synchronously."""
        with self._compute_lock:
            return self.navigator.greedy_suggestion(
                max_bits=max_bits,
                max_tokens=max_tokens,
            )

    def cached_greedy_suggestion(
        self,
        *,
        max_bits: float,
        max_tokens: int = 256,
    ) -> GreedySuggestion:
        """Return as much of the greedy path as the Dijkstra cache already knows."""
        if max_bits < 0:
            raise ValueError("max_bits must be >= 0")
        budget_nats = max_bits * math.log(2)
        used = 0.0
        path: tuple[int, ...] = ()
        tokens: list[int] = []
        next_cost: float | None = None
        complete = False

        with self._condition:
            self._raise_worker_error_locked()
            for _ in range(max_tokens):
                node = self._nodes.get(path)
                if node is None or not node.expanded:
                    break
                if node.ended:
                    complete = True
                    break
                ranked = node.ranked
                if ranked is None or not ranked.tokens:
                    complete = True
                    break
                token = ranked.tokens[0]
                p = ranked.probabilities[0]
                cost = -math.log(p)
                if used + cost > budget_nats + 1e-12:
                    next_cost = cost
                    complete = True
                    break
                tokens.append(token)
                used += cost
                path = (*path, token)
                if path not in self._nodes:
                    break
            else:
                complete = False

        return GreedySuggestion(
            tokens=tuple(tokens),
            nats=used,
            bits=used / math.log(2),
            next_token_nats=next_cost,
            complete=complete,
        )

    def _hidden_summary_locked(self, node: _TokenNode) -> tuple[int, float]:
        ranked = node.ranked
        if ranked is None:
            return 0, 0.0
        active_lo, active_hi = self._active_interval_locked()
        hidden_count = 0
        hidden_mass = 0.0
        width = node.hi - node.lo
        if width <= 0:
            return 0, 0.0

        for rank in range(len(ranked.tokens)):
            if rank in node.materialized_ranks:
                continue
            lo, hi = self._child_interval(node, rank)
            overlap = self._intersection(lo, hi, active_lo, active_hi)
            if overlap <= 0:
                continue
            hidden_count += 1
            hidden_mass += overlap / width
        return hidden_count, hidden_mass

    def tree_entries(self) -> tuple[TreeEntry, ...]:
        """Return a DFS render view of currently relevant materialized nodes."""
        with self._condition:
            self._raise_worker_error_locked()
            active_lo, active_hi = self._active_interval_locked()
            children: dict[tuple[int, ...], list[_TokenNode]] = {}
            for node in self._nodes.values():
                if node.parent is None:
                    continue
                if self._intersection(node.lo, node.hi, active_lo, active_hi) <= 0:
                    continue
                children.setdefault(node.parent, []).append(node)
            for siblings in children.values():
                siblings.sort(key=lambda node: node.rank)

            entries: list[TreeEntry] = []

            def visit(parent_path: tuple[int, ...], depth: int) -> None:
                parent = self._nodes.get(parent_path)
                for child in children.get(parent_path, []):
                    entries.append(
                        TreeEntry(
                            path=child.path,
                            depth=depth,
                            token=child.token,
                            edge_nats=child.edge_nats,
                            path_nats=child.path_nats,
                            expanded=child.expanded,
                        )
                    )
                    visit(child.path, depth + 1)

                if parent is not None and parent.expanded:
                    hidden_count, hidden_mass = self._hidden_summary_locked(parent)
                    if hidden_count and hidden_mass > 0:
                        entries.append(
                            TreeEntry(
                                path=parent_path,
                                depth=depth,
                                token=None,
                                edge_nats=0.0,
                                path_nats=parent.path_nats,
                                expanded=False,
                                is_ellipsis=True,
                                hidden_count=hidden_count,
                                hidden_nats=-math.log(hidden_mass),
                            )
                        )

            visit((), 0)
            return tuple(entries)

    def stats(self) -> ExplorerStats:
        with self._condition:
            self._raise_worker_error_locked()
            return ExplorerStats(
                nodes=len(self._nodes),
                expanded=self._expanded_count,
                frontier=len(self._heap),
                computing=self._computing,
                generation=self._generation,
                saturated=self._saturated,
            )


TreeExplorer = TokenTreeExplorer
