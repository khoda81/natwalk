"""Single mutation boundary tying a model cursor to tree and search state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .model import Cursor, rank
from .search import Search
from .tree import Distribution, NodeId, Tree


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Committed execution state; discovered tree knowledge is intentionally absent."""

    root: NodeId
    cursor: object


class Session:
    """Own committed model state, the discovered tree, and its search frontier."""

    def __init__(self, cursor: Cursor) -> None:
        self.cursor = cursor
        self.tree = Tree()
        self.root = self.tree.root
        self._root_checkpoint = cursor.checkpoint()
        self.search = Search(self.tree, self._evaluate, root=self.root)

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(self.root, self._root_checkpoint)

    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore committed execution state without discarding discovered knowledge."""
        self.cursor.restore(checkpoint.cursor)
        self.root = checkpoint.root
        self._root_checkpoint = checkpoint.cursor
        self.search.reset(self.root)

    def distribution(self) -> Distribution:
        """Return the complete distribution at the current committed root."""
        node = self.tree[self.root]
        if node.distribution is None:
            node.distribution = rank(self.cursor.predict())
        return node.distribution

    def _evaluate(self, tree: Tree, node_id: NodeId) -> Distribution:
        self.cursor.restore(self._root_checkpoint)
        try:
            for token in tree.path_from(self.root, node_id):
                self.cursor.observe(token)
            return rank(self.cursor.predict())
        finally:
            self.cursor.restore(self._root_checkpoint)

    def commit(self, tokens: Iterable[int]) -> NodeId:
        """Commit a token stream and make its endpoint the search root."""
        old_root = self.root

        for token in tokens:
            distribution = self.distribution()
            rank_index = distribution.tokens.index(token)
            self.root = self.tree.child(self.root, rank_index)
            self.cursor.observe(token)

        if self.root != old_root:
            self._root_checkpoint = self.cursor.checkpoint()
            self.search.reset(self.root)
        return self.root
