"""Synchronous lazy-sibling Dijkstra over a :mod:`natwalk.tree` trie."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass

from .tree import NodeId, RankedDistribution, Tree

type Evaluator = Callable[[Tree, NodeId, int], RankedDistribution]


@dataclass(order=True, frozen=True, slots=True)
class Candidate:
    """One virtual child waiting on the global Dijkstra frontier."""

    path_nats: float
    parent: NodeId
    rank: int


class Search:
    """Uniform-cost search using one queued sibling per discovered parent."""

    def __init__(self, tree: Tree, evaluate: Evaluator, *, root: NodeId = 0) -> None:
        self.tree = tree
        self.evaluate = evaluate
        self.frontier: list[Candidate] = []
        self.reset(root)

    def _push(self, parent_id: NodeId, rank: int, parent_nats: float) -> None:
        distribution = self.tree[parent_id].distribution
        if rank >= len(distribution):
            return
        path_nats = parent_nats + distribution.nats(rank)
        if math.isfinite(path_nats):
            heapq.heappush(self.frontier, Candidate(path_nats, parent_id, rank))

    def reset(self, root: NodeId) -> None:
        """Restart search from ``root`` without discarding discovered tree knowledge."""
        self.frontier.clear()
        self._push(root, 0, 0.0)

    def _advance(self) -> tuple[NodeId, bool] | None:
        """Advance one Dijkstra candidate and report whether it discovered a node."""
        if not self.frontier:
            return None

        candidate = heapq.heappop(self.frontier)
        distribution = self.tree[candidate.parent].distribution
        parent_nats = candidate.path_nats - distribution.nats(candidate.rank)
        self._push(candidate.parent, candidate.rank + 1, parent_nats)

        child = self.tree.child(candidate.parent, candidate.rank)
        discovered = child is None
        if child is None:
            child_distribution = self.evaluate(
                self.tree,
                candidate.parent,
                candidate.rank,
            )
            child = self.tree.put_child(
                candidate.parent,
                candidate.rank,
                child_distribution,
            )

        self._push(child, 0, candidate.path_nats)
        return child, discovered

    def step(self) -> NodeId | None:
        """Advance exactly one lowest-cost frontier candidate."""
        result = self._advance()
        return None if result is None else result[0]

    def discover(self) -> NodeId | None:
        """Fast-forward known edges and return the next newly discovered node."""
        while (result := self._advance()) is not None:
            child, discovered = result
            if discovered:
                return child
        return None
