"""Read-only completion queries over discovered probability-tree knowledge."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .tree import NodeId, Tree


@dataclass(frozen=True, slots=True)
class Suggestion:
    tokens: tuple[int, ...]
    nats: float
    complete: bool
    next_nats: float | None = None

    @property
    def bits(self) -> float:
        return self.nats / math.log(2)


def greedy(
    tree: Tree,
    root: NodeId,
    *,
    max_nats: float,
    max_tokens: int = 256,
) -> Suggestion:
    """Return the known rank-zero continuation inside ``max_nats``."""
    node_id = root
    tokens: list[int] = []
    used = 0.0

    for _ in range(max_tokens):
        distribution = tree[node_id].distribution
        if len(distribution) == 0:
            return Suggestion(tuple(tokens), used, True)
        if distribution.revealed == 0:
            return Suggestion(tuple(tokens), used, False)

        cost = distribution.nats(0)
        if used + cost > max_nats:
            return Suggestion(tuple(tokens), used, True, cost)

        tokens.append(distribution.token(0))
        used += cost
        child_id = tree.child(node_id, 0)
        if child_id is None:
            return Suggestion(tuple(tokens), used, False)
        node_id = child_id

    return Suggestion(tuple(tokens), used, False)


def completions(
    tree: Tree,
    root: NodeId,
    *,
    max_nats: float,
    max_tokens: int = 256,
    limit: int = 64,
) -> tuple[Suggestion, ...]:
    """Return maximal known paths in probability-ranked DFS order."""
    if limit <= 0 or max_tokens <= 0:
        return ()

    out: list[Suggestion] = []

    def visit(node_id: NodeId, path: tuple[int, ...], used: float) -> None:
        if len(out) >= limit:
            return

        distribution = tree[node_id].distribution
        if len(distribution) == 0:
            if path:
                out.append(Suggestion(path, used, True))
            return

        extended = False
        for rank in range(distribution.revealed):
            token = distribution.token(rank)
            cost = distribution.nats(rank)
            next_used = used + cost
            if next_used > max_nats:
                break
            extended = True
            next_path = (*path, token)
            child_id = tree.child(node_id, rank)
            if len(next_path) >= max_tokens:
                out.append(Suggestion(next_path, next_used, False))
            elif child_id is None:
                out.append(Suggestion(next_path, next_used, False))
            else:
                before = len(out)
                visit(child_id, next_path, next_used)
                if len(out) == before:
                    out.append(Suggestion(next_path, next_used, True))
            if len(out) >= limit:
                return

        if path and not extended:
            next_nats = distribution.nats(0) if distribution.revealed else None
            out.append(Suggestion(path, used, True, next_nats))

    visit(root, (), 0.0)
    return tuple(out[:limit])
