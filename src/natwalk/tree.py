"""Persistent discovered probability tree."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

NodeId = int


@dataclass(frozen=True, slots=True)
class Distribution:
    """A complete next-symbol distribution in descending probability order."""

    tokens: tuple[int, ...]
    probabilities: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.tokens)

    def nats(self, rank: int) -> float:
        probability = self.probabilities[rank]
        return -math.log(probability) if probability != 0.0 else math.inf


@dataclass(slots=True)
class Node:
    """One concrete node in the discovered tree."""

    parent: NodeId | None
    rank: int
    distribution: Distribution | None = None
    children: dict[int, NodeId] = field(default_factory=dict)


class Tree:
    """Arena-backed probability trie."""

    def __init__(self) -> None:
        self.nodes: list[Node] = [Node(parent=None, rank=-1)]

    @property
    def root(self) -> NodeId:
        return 0

    def __getitem__(self, node: NodeId) -> Node:
        return self.nodes[node]

    def child(self, parent_id: NodeId, rank: int) -> NodeId:
        """Return the concrete child at ``rank``, allocating it if necessary."""
        parent = self.nodes[parent_id]
        distribution = parent.distribution
        if distribution is None:
            raise ValueError("cannot materialize a child of an unexpanded node")
        if not 0 <= rank < len(distribution):
            raise IndexError(rank)

        existing = parent.children.get(rank)
        if existing is not None:
            return existing

        child_id = len(self.nodes)
        self.nodes.append(Node(parent=parent_id, rank=rank))
        parent.children[rank] = child_id
        return child_id

    def token(self, node_id: NodeId) -> int:
        node = self.nodes[node_id]
        if node.parent is None:
            raise ValueError("root has no token")
        parent = self.nodes[node.parent]
        if parent.distribution is None:
            raise RuntimeError("node parent is unexpectedly unexpanded")
        return parent.distribution.tokens[node.rank]

    def edge_nats(self, node_id: NodeId) -> float:
        node = self.nodes[node_id]
        if node.parent is None:
            return 0.0
        parent = self.nodes[node.parent]
        if parent.distribution is None:
            raise RuntimeError("node parent is unexpectedly unexpanded")
        return parent.distribution.nats(node.rank)

    def path(self, node_id: NodeId) -> tuple[int, ...]:
        return self.path_from(self.root, node_id)

    def path_from(self, ancestor: NodeId, node_id: NodeId) -> tuple[int, ...]:
        """Return tokens from ``ancestor`` (exclusive) to ``node_id`` (inclusive)."""
        tokens: list[int] = []
        current = node_id
        while current != ancestor:
            node = self.nodes[current]
            if node.parent is None:
                raise ValueError("node is not a descendant of ancestor")
            tokens.append(self.token(current))
            current = node.parent
        tokens.reverse()
        return tuple(tokens)

    def path_nats(self, node_id: NodeId, *, ancestor: NodeId = 0) -> float:
        """Derive cumulative surprisal from ``ancestor`` to ``node_id``."""
        total = 0.0
        current = node_id
        while current != ancestor:
            node = self.nodes[current]
            if node.parent is None:
                raise ValueError("node is not a descendant of ancestor")
            total += self.edge_nats(current)
            current = node.parent
        return total
