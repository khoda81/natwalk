"""Synchronous lazy-sibling Dijkstra over a :mod:`natwalk.tree` trie."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass

from .tree import Distribution, NodeId, Tree

type Evaluator = Callable[[Tree, NodeId], Distribution]


@dataclass(order=True, frozen=True, slots=True)
class Candidate:
    """One virtual child waiting on the global Dijkstra frontier."""

    path_nats: float
    parent: NodeId
    rank: int


class Search:
    """Uniform-cost search using one queued sibling per expanded parent."""

    def __init__(self, tree: Tree, evaluate: Evaluator, *, root: NodeId = 0) -> None:
        self.tree = tree
        self.evaluate = evaluate
        self.frontier: list[Candidate] = []
        self.reset(root)

    def _push(self, parent_id: NodeId, rank: int, parent_nats: float) -> None:
        distribution = self.tree[parent_id].distribution
        if distribution is None:
            raise RuntimeError("frontier parent is unexpectedly unexpanded")
        if rank >= len(distribution):
            return
        path_nats = parent_nats + distribution.nats(rank)
        if math.isfinite(path_nats):
            heapq.heappush(self.frontier, Candidate(path_nats, parent_id, rank))

    def _expand(self, node_id: NodeId) -> None:
        node = self.tree[node_id]
        if node.distribution is None:
            node.distribution = self.evaluate(self.tree, node_id)

    def reset(self, root: NodeId) -> None:
        """Restart search from ``root`` without discarding discovered tree knowledge."""
        self.frontier.clear()
        self._expand(root)
        self._push(root, 0, 0.0)

    def step(self) -> NodeId | None:
        """Pop and expand the next lowest-cost concrete node."""
        if not self.frontier:
            return None

        candidate = heapq.heappop(self.frontier)
        distribution = self.tree[candidate.parent].distribution
        assert distribution is not None
        parent_nats = candidate.path_nats - distribution.nats(candidate.rank)
        self._push(candidate.parent, candidate.rank + 1, parent_nats)

        child = self.tree.child(candidate.parent, candidate.rank)
        self._expand(child)
        self._push(child, 0, candidate.path_nats)
        return child
