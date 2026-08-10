"""Synchronous lazy-sibling Dijkstra over a :mod:`natwalk.tree` trie."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass

from .tree import Distribution, NodeId, Tree

Evaluator = Callable[[Tree, NodeId], Distribution]


@dataclass(order=True, frozen=True, slots=True)
class Candidate:
    """One virtual child waiting on the global Dijkstra frontier."""

    path_nats: float
    parent: NodeId
    rank: int


class Search:
    """Uniform-cost search using one queued sibling per expanded parent.

    For each expanded parent, child costs are nondecreasing by rank. Therefore
    its children form a sorted stream. The heap stores only the head of every
    active stream; popping ``(parent, rank)`` advances that stream to
    ``rank + 1`` and starts the popped child's stream at rank zero.
    """

    def __init__(self, tree: Tree, evaluate: Evaluator, *, root: NodeId = 0) -> None:
        self.tree = tree
        self.evaluate = evaluate
        self.root = root
        self.frontier: list[Candidate] = []
        self._expand(root)

    def _push(self, parent_id: NodeId, rank: int) -> None:
        parent = self.tree[parent_id]
        distribution = parent.distribution
        if distribution is None:
            raise RuntimeError("frontier parent is unexpectedly unexpanded")
        if rank >= len(distribution):
            return
        path_nats = parent.path_nats + distribution.nats[rank]
        if not math.isfinite(path_nats):
            return
        heapq.heappush(
            self.frontier,
            Candidate(
                path_nats=path_nats,
                parent=parent_id,
                rank=rank,
            ),
        )

    def _expand(self, node_id: NodeId) -> None:
        node = self.tree[node_id]
        if node.distribution is not None:
            return
        node.distribution = self.evaluate(self.tree, node_id)
        self._push(node_id, 0)

    def step(self) -> NodeId | None:
        """Pop and expand the next lowest-cost concrete node."""
        if not self.frontier:
            return None

        candidate = heapq.heappop(self.frontier)

        # Advance the parent's sorted sibling stream.
        self._push(candidate.parent, candidate.rank + 1)

        # Materialize and expand the popped child. ``Tree.child`` is idempotent,
        # so other read-only consumers may have already allocated the node.
        child = self.tree.child(candidate.parent, candidate.rank)
        self._expand(child)
        return child
