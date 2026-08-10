"""Budget-sphere completion views over a :class:`TokenTreeExplorer` cache.

This module deliberately treats the explorer's probability-ranked sibling order
as the lexicographic order. The all-rank-zero (model-greedy) continuation is
therefore the first completion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .core import GreedySuggestion, TokenTreeExplorer, TreeEntry

_EPS = 1e-12


def cached_budget_completions(
    explorer: TokenTreeExplorer,
    *,
    max_nats: float,
    max_tokens: int = 256,
    limit: int = 64,
) -> tuple[GreedySuggestion, ...]:
    """Return cached maximal paths inside an information budget.

    Paths are returned in the same probability-ranked DFS order used by the
    tree renderer. A path is a cached completion when it fits inside
    ``max_nats`` and has no currently materialized child that also fits. The
    first result is pinned to the cached model-greedy suggestion when one is
    available, so rank-zero descent remains lexicographically first even while
    the background tree is still filling.

    This is intentionally a view of the *known* tree. Under a tight node cap an
    ellipsis may still hide another admissible completion; as Dijkstra discovers
    more nodes, subsequent calls expose it without changing search semantics.
    """
    if max_nats < 0:
        raise ValueError("max_nats must be >= 0")
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    if limit <= 0:
        return ()

    entries = explorer.tree_entries()
    visible = [
        entry
        for entry in entries
        if not entry.is_ellipsis
        and len(entry.path) <= max_tokens
        and entry.path_nats <= max_nats + _EPS
    ]

    parents_with_fitting_children = {entry.path[:-1] for entry in visible}
    leaves = [entry for entry in visible if entry.path not in parents_with_fitting_children]

    out: list[GreedySuggestion] = []
    seen: set[tuple[int, ...]] = set()

    greedy = explorer.cached_greedy_suggestion(
        max_bits=max_nats / math.log(2),
        max_tokens=max_tokens,
    )
    if greedy.tokens:
        out.append(greedy)
        seen.add(greedy.tokens)

    for entry in leaves:
        if len(out) >= limit:
            break
        if entry.path in seen:
            continue
        out.append(_entry_suggestion(entry))
        seen.add(entry.path)

    return tuple(out)


def accept_completion(
    explorer: TokenTreeExplorer,
    tokens: Sequence[int],
) -> GreedySuggestion:
    """Commit an arbitrary cached completion as one undoable explicit action.

    ``Navigator`` does not yet expose arbitrary-path acceptance publicly, so
    this package-level helper performs the same operation under the explorer's
    compute lock and then re-roots Dijkstra. It intentionally lives beside the
    explorer rather than in the MuScriptor example so the interaction remains
    model-agnostic.
    """
    path = tuple(tokens)
    if not path:
        return GreedySuggestion((), 0.0, 0.0, None, True)

    with explorer._compute_lock:  # noqa: SLF001 - package-level companion API
        with explorer._condition:  # noqa: SLF001
            explorer._raise_worker_error_locked()  # noqa: SLF001

        navigator = explorer.navigator

        # Validate the cached path and measure its exact current surprisal
        # before mutating the live cursor.
        used = 0.0
        with navigator.temporary_cursor() as cursor:
            for token in path:
                if cursor.ended:
                    raise ValueError("completion continues after EOS")
                probs = cursor.predict()
                p = float(probs[token])
                if p <= 0.0:
                    raise ValueError(f"completion token {token} has zero probability")
                used -= math.log(p)
                cursor.observe(token)

        navigator._push_history()  # noqa: SLF001 - preserve normal undo semantics
        for token in path:
            probs = navigator.state.cursor.predict()
            p = float(probs[token])
            navigator.state.path_surprisal -= math.log(p)
            navigator.state.cursor.observe(token)

        navigator.state.lo = 0.0
        navigator.state.hi = 1.0

        with explorer._condition:  # noqa: SLF001
            explorer._snapshot = navigator.snapshot()  # noqa: SLF001
            explorer._reset_tree_locked()  # noqa: SLF001

    return GreedySuggestion(
        tokens=path,
        nats=used,
        bits=used / math.log(2),
        next_token_nats=None,
        complete=True,
    )


def _entry_suggestion(entry: TreeEntry) -> GreedySuggestion:
    return GreedySuggestion(
        tokens=entry.path,
        nats=entry.path_nats,
        bits=entry.path_nats / math.log(2),
        next_token_nats=None,
        complete=entry.expanded,
    )
