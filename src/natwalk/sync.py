"""Idempotent synchronization for append-only discovered trees."""

from __future__ import annotations

from dataclasses import dataclass

from .tree import Distribution, NodeId, Tree


@dataclass(frozen=True, slots=True)
class NodeUpdate:
    """Absolute contents of one node in the append-only tree log."""

    node: NodeId
    parent: NodeId | None
    rank: int
    distribution: Distribution


def updates(tree: Tree, *, start: NodeId = 0) -> tuple[NodeUpdate, ...]:
    """Return the append-only suffix beginning at node id ``start``."""
    if not 0 <= start <= len(tree.nodes):
        raise IndexError(start)
    return tuple(
        NodeUpdate(
            node=node_id,
            parent=node.parent,
            rank=node.rank,
            distribution=node.distribution,
        )
        for node_id, node in enumerate(tree.nodes[start:], start=start)
    )


class TreeReplica:
    """A client-side tree reconstructed from idempotent absolute node updates."""

    def __init__(self) -> None:
        self.tree: Tree | None = None

    @property
    def next_node(self) -> NodeId:
        """First node id not yet present in this replica."""
        return 0 if self.tree is None else len(self.tree.nodes)

    def apply(self, update: NodeUpdate) -> None:
        """Apply one ordered update; duplicates are verified no-ops."""
        if update.node == 0:
            self._apply_root(update)
            return

        tree = self.tree
        if tree is None:
            raise ValueError("tree sync must start with root node 0")

        if update.node < len(tree.nodes):
            self._verify_existing(update)
            return
        if update.node > len(tree.nodes):
            raise ValueError(
                f"missing tree updates before node {update.node}; expected {len(tree.nodes)}"
            )
        if update.parent is None:
            raise ValueError("non-root tree update has no parent")

        child = tree.put_child(update.parent, update.rank, update.distribution)
        if child != update.node:
            raise ValueError(f"tree update {update.node} conflicts with existing child id {child}")

    def apply_many(self, batch: tuple[NodeUpdate, ...]) -> None:
        for update in batch:
            self.apply(update)

    def _apply_root(self, update: NodeUpdate) -> None:
        if update.parent is not None or update.rank != -1:
            raise ValueError("root update must have parent=None and rank=-1")
        if self.tree is None:
            self.tree = Tree(update.distribution)
            return
        if self.tree[0].distribution != update.distribution:
            raise ValueError("conflicting root distribution")

    def _verify_existing(self, update: NodeUpdate) -> None:
        assert self.tree is not None
        node = self.tree[update.node]
        if (
            node.parent != update.parent
            or node.rank != update.rank
            or node.distribution != update.distribution
        ):
            raise ValueError(f"conflicting contents for tree node {update.node}")
        if update.parent is not None and self.tree.child(update.parent, update.rank) != update.node:
            raise ValueError(f"conflicting edge for tree node {update.node}")
