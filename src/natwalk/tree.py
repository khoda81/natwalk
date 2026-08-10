"""Persistent discovered probability tree.

The tree stores model knowledge, not view state or search scheduling state.
An expanded node owns its complete ranked next-symbol distribution. Child
nodes are allocated only when some consumer needs a concrete subtree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NodeId = int


@dataclass(frozen=True, slots=True)
class Distribution:
    """A complete next-symbol distribution in descending probability order.

    ``nats[i]`` is ``-log p(tokens[i])``. Empty distributions are terminal.
    """

    tokens: tuple[int, ...]
    nats: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(slots=True)
class Node:
    """One concrete node in the discovered tree."""

    parent: NodeId | None
    rank: int
    path_nats: float
    distribution: Distribution | None = None
    children: dict[int, NodeId] = field(default_factory=dict)


class Tree:
    """Arena-backed probability trie.

    Child identity is ``(parent, rank)``. Token IDs, edge costs, paths, and
    depths are derived from that relation instead of being stored redundantly.
    """

    def __init__(self) -> None:
        self.nodes: list[Node] = [Node(parent=None, rank=-1, path_nats=0.0)]

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
        self.nodes.append(
            Node(
                parent=parent_id,
                rank=rank,
                path_nats=parent.path_nats + distribution.nats[rank],
            )
        )
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
        return node.path_nats - parent.path_nats

    def path(self, node_id: NodeId) -> tuple[int, ...]:
        tokens: list[int] = []
        current = node_id
        while current != self.root:
            tokens.append(self.token(current))
            parent = self.nodes[current].parent
            assert parent is not None
            current = parent
        tokens.reverse()
        return tuple(tokens)
