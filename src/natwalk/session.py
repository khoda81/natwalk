"""Single mutation boundary tying a model cursor to tree and search state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .model import Cursor, rank
from .search import Search
from .tree import NodeId, RankedDistribution, Tree


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Committed execution state; discovered tree knowledge is intentionally absent."""

    root: NodeId
    cursor: object


class Session:
    """Own committed model state, the discovered tree, and its search frontier."""

    def __init__(self, cursor: Cursor) -> None:
        self.cursor = cursor
        root_distribution = rank(cursor.predict())
        self.tree = Tree(root_distribution)
        self.root = self.tree.root
        self._root_checkpoint = cursor.checkpoint()
        self.search = Search(self.tree, self._evaluate_child, root=self.root)

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(self.root, self._root_checkpoint)

    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore committed execution state without discarding discovered knowledge."""
        self.cursor.restore(checkpoint.cursor)
        self.root = checkpoint.root
        self._root_checkpoint = checkpoint.cursor
        self.search.reset(self.root)

    def distribution(self) -> RankedDistribution:
        """Return the complete authoritative distribution at the committed root."""
        return self.tree[self.root].distribution

    def inspect(self, node_id: NodeId) -> RankedDistribution:
        """Return one already-discovered authoritative node distribution."""
        return self.tree[node_id].distribution

    def inspect_child(self, parent_id: NodeId, rank_index: int) -> NodeId:
        """Evaluate and publish one ranked child if it is not already discovered."""
        distribution = self.tree[parent_id].distribution
        if not 0 <= rank_index < len(distribution):
            raise IndexError(rank_index)

        child = self.tree.child(parent_id, rank_index)
        if child is not None:
            return child

        child_distribution = self._evaluate_child(self.tree, parent_id, rank_index)
        return self.tree.put_child(parent_id, rank_index, child_distribution)

    def _evaluate_child(
        self,
        tree: Tree,
        parent_id: NodeId,
        rank_index: int,
    ) -> RankedDistribution:
        parent_distribution = tree[parent_id].distribution
        token = parent_distribution.token(rank_index)

        self.cursor.restore(self._root_checkpoint)
        try:
            for prefix_token in tree.path_from(self.root, parent_id):
                self.cursor.observe(prefix_token)
            self.cursor.observe(token)
            return rank(self.cursor.predict())
        finally:
            self.cursor.restore(self._root_checkpoint)

    def commit(self, tokens: Iterable[int]) -> NodeId:
        """Commit a token stream and make its endpoint the search root."""
        old_root = self.root

        for token in tokens:
            parent = self.root
            distribution = self.tree[parent].distribution
            rank_index = distribution.rank(token)
            child = self.tree.child(parent, rank_index)

            self.cursor.observe(token)
            if child is None:
                child_distribution = rank(self.cursor.predict())
                child = self.tree.put_child(parent, rank_index, child_distribution)
            self.root = child

        if self.root != old_root:
            self._root_checkpoint = self.cursor.checkpoint()
            self.search.reset(self.root)
        return self.root
