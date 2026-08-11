"""Persistent append-only probability tree."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

type NodeId = int


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


@dataclass(frozen=True, slots=True)
class Node:
    """One complete immutable node in the discovered tree."""

    parent: NodeId | None
    rank: int
    distribution: Distribution
    _children: dict[int, NodeId] = field(default_factory=dict, compare=False, repr=False)


class Tree:
    """Arena-backed append-only probability trie.

    A node is published only after its complete distribution is known. Child
    insertion is idempotent: repeating the same write returns the existing node;
    attempting to give an existing edge a different distribution is an invariant
    violation.
    """

    def __init__(self, root_distribution: Distribution) -> None:
        self.nodes: list[Node] = [
            Node(parent=None, rank=-1, distribution=root_distribution)
        ]

    @property
    def root(self) -> NodeId:
        return 0

    def __getitem__(self, node: NodeId) -> Node:
        return self.nodes[node]

    def child(self, parent_id: NodeId, rank: int) -> NodeId | None:
        """Return the discovered child at ``rank`` without mutating the tree."""
        return self.nodes[parent_id]._children.get(rank)

    def put_child(
        self,
        parent_id: NodeId,
        rank: int,
        distribution: Distribution,
    ) -> NodeId:
        """Publish one complete child, idempotently."""
        parent = self.nodes[parent_id]
        if not 0 <= rank < len(parent.distribution):
            raise IndexError(rank)

        existing = parent._children.get(rank)
        if existing is not None:
            node = self.nodes[existing]
            if node.distribution != distribution:
                raise ValueError(
                    f"conflicting distribution for child ({parent_id}, {rank})"
                )
            return existing

        child_id = len(self.nodes)
        child = Node(parent=parent_id, rank=rank, distribution=distribution)
        self.nodes.append(child)
        parent._children[rank] = child_id
        return child_id

    def token(self, node_id: NodeId) -> int:
        node = self.nodes[node_id]
        if node.parent is None:
            raise ValueError("root has no token")
        return self.nodes[node.parent].distribution.tokens[node.rank]

    def edge_nats(self, node_id: NodeId) -> float:
        node = self.nodes[node_id]
        if node.parent is None:
            return 0.0
        return self.nodes[node.parent].distribution.nats(node.rank)

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
