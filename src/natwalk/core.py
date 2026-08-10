"""Exact navigation of autoregressive distributions in fixed-information steps."""

from __future__ import annotations

import bisect
import heapq
import itertools
import math
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol


class Cursor(Protocol):
    """Clonable causal model state consumed by :class:`Navigator`.

    ``predict`` must return the *complete* next-symbol distribution in the same
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


@dataclass(frozen=True)
class ExplorerStats:
    """Observable state of :class:`TreeExplorer`'s speculative cache."""

    cached: int
    queued: int
    computing: bool
    expanded: int
    generation: int


class Navigator:
    """K-ary arithmetic navigator over a complete autoregressive distribution."""

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
    def _temporary_cursor(self, cursor: Cursor) -> Iterator[Cursor]:
        if self._can_rewind(cursor):
            checkpoint = cursor.checkpoint()  # type: ignore[attr-defined]
            try:
                yield cursor
            finally:
                cursor.restore(checkpoint)  # type: ignore[attr-defined]
        else:
            yield cursor.clone()

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
            with self._temporary_cursor(self.state.cursor) as cursor:
                representative = self.decode_point(
                    cursor, midpoint, max_tokens=self.preview_tokens
                )

        return Preview(bucket=bucket, forced=forced, representative=representative)

    def preview(self, bucket: int) -> Preview:
        """Preview one bucket without mutating the live navigator."""
        with self._temporary_state():
            return self._preview_current(bucket)

    def preview_path(self, buckets: Sequence[int]) -> Preview:
        """Preview the final action in a speculative multi-action path."""
        path = tuple(buckets)
        if not path:
            raise ValueError("preview path must contain at least one bucket")

        with self._temporary_state():
            for bucket in path[:-1]:
                if self.state.cursor.ended:
                    return Preview(bucket=path[-1], forced=(), representative=())
                self._narrow(self.state, bucket)
                self.state, _ = self._drain_forced(self.state)
            if self.state.cursor.ended:
                return Preview(bucket=path[-1], forced=(), representative=())
            return self._preview_current(path[-1])


class TreeExplorer:
    """Background probability-mass-first speculative tree for a Navigator.

    Every depth-d action region has exact mass ``choices**-d``. The worker
    therefore expands shallow paths first; equal-mass paths are tie-broken
    lexicographically so modeward buckets are explored before residual ones.

    Only paths ending in visible buckets are cached. Residual buckets are still
    traversed as ancestors, so selecting ``…`` can immediately reuse any
    already-computed descendants.
    """

    def __init__(
        self,
        navigator: Navigator,
        *,
        prefetch_depth: int = 2,
        max_cached: int = 128,
        autostart: bool = True,
    ) -> None:
        if prefetch_depth < 1:
            raise ValueError("prefetch_depth must be >= 1")
        if max_cached < navigator.choices - 1:
            raise ValueError("max_cached must fit all visible current choices")

        self.navigator = navigator
        self.prefetch_depth = prefetch_depth
        self.max_cached = max_cached

        self._condition = threading.Condition()
        self._compute_lock = threading.Lock()
        self._queue: list[tuple[int, tuple[int, ...], int]] = []
        self._pending: set[tuple[int, ...]] = set()
        self._cache: dict[tuple[int, ...], Preview] = {}
        self._generation = 0
        self._expanded = 0
        self._computing = False
        self._stop = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._snapshot = self._snapshot_live()

        with self._condition:
            self._schedule_locked()
        if autostart:
            self.start()

    def __enter__(self) -> TreeExplorer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def choices(self) -> int:
        return self.navigator.choices

    def _snapshot_live(self) -> NavigationSnapshot:
        state = self.navigator.state
        return NavigationSnapshot(
            prefix=tuple(state.cursor.prefix),
            ended=state.cursor.ended,
            lo=state.lo,
            hi=state.hi,
            actions=state.actions,
            path_surprisal=state.path_surprisal,
        )

    @property
    def snapshot(self) -> NavigationSnapshot:
        with self._condition:
            return self._snapshot

    @property
    def supplied_nats(self) -> float:
        return self.snapshot.actions * self.nats_per_action

    @property
    def supplied_bits(self) -> float:
        return self.snapshot.actions * self.bits_per_action

    @property
    def nats_per_action(self) -> float:
        return self.navigator.nats_per_action

    @property
    def bits_per_action(self) -> float:
        return self.navigator.bits_per_action

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._stop:
                raise RuntimeError("cannot restart a closed TreeExplorer")
            self._thread = threading.Thread(
                target=self._worker,
                name="natwalk-prefetch",
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

    def _candidate_paths(self) -> Iterator[tuple[int, ...]]:
        visible = range(self.choices - 1)
        all_buckets = range(self.choices)
        for depth in range(1, self.prefetch_depth + 1):
            if depth == 1:
                for bucket in visible:
                    yield (bucket,)
                continue
            for prefix in itertools.product(all_buckets, repeat=depth - 1):
                for bucket in visible:
                    yield (*prefix, bucket)

    def _schedule_locked(self) -> None:
        if self._snapshot.ended:
            return

        for path in self._candidate_paths():
            if path in self._cache or path in self._pending:
                continue
            if len(self._cache) + len(self._pending) >= self.max_cached:
                break
            heapq.heappush(self._queue, (len(path), path, self._generation))
            self._pending.add(path)
        self._condition.notify_all()

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                _depth, path, generation = heapq.heappop(self._queue)
                self._pending.discard(path)
                if generation != self._generation or path in self._cache:
                    continue
                self._computing = True

            preview: Preview | None = None
            error: BaseException | None = None
            try:
                with self._compute_lock:
                    with self._condition:
                        stale = generation != self._generation or self._stop
                    if not stale:
                        preview = self.navigator.preview_path(path)
            except BaseException as exc:
                error = exc

            with self._condition:
                self._computing = False
                if preview is not None and generation == self._generation and not self._stop:
                    self._cache[path] = preview
                    self._expanded += 1
                self._condition.notify_all()

                if error is not None:
                    self._error = error
                    self._stop = True
                    self._condition.notify_all()
                    return

    def _raise_worker_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("natwalk background worker failed") from self._error

    def preview(self, bucket: int) -> Preview | None:
        """Return a cached current preview, or None while it is computing."""
        if not 0 <= bucket < self.choices - 1:
            raise ValueError(f"visible bucket must be in [0, {self.choices - 1})")
        with self._condition:
            self._raise_worker_error_locked()
            return self._cache.get((bucket,))

    def current_previews(self) -> tuple[Preview | None, ...]:
        with self._condition:
            self._raise_worker_error_locked()
            return tuple(self._cache.get((bucket,)) for bucket in range(self.choices - 1))

    def wait_current(self, timeout: float | None = None) -> bool:
        """Wait until every visible current preview is cached."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                self._raise_worker_error_locked()
                if self._snapshot.ended:
                    return True
                if all((bucket,) in self._cache for bucket in range(self.choices - 1)):
                    return True
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(remaining)

    def choose(self, bucket: int) -> tuple[int, ...]:
        """Commit a choice, prune non-descendants, and rebase cached descendants."""
        if not 0 <= bucket < self.choices:
            raise ValueError(f"bucket must be in [0, {self.choices})")

        with self._compute_lock:
            with self._condition:
                self._raise_worker_error_locked()

            forced = self.navigator.choose(bucket)

            with self._condition:
                self._snapshot = self._snapshot_live()
                self._generation += 1
                self._cache = {
                    path[1:]: preview
                    for path, preview in self._cache.items()
                    if len(path) > 1 and path[0] == bucket
                }
                self._queue.clear()
                self._pending.clear()
                self._schedule_locked()
                self._condition.notify_all()

        return forced

    def stats(self) -> ExplorerStats:
        with self._condition:
            self._raise_worker_error_locked()
            return ExplorerStats(
                cached=len(self._cache),
                queued=len(self._queue),
                computing=self._computing,
                expanded=self._expanded,
                generation=self._generation,
            )
