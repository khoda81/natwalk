"""Single mutation boundary tying a model cursor to tree and search state."""

from __future__ import annotations

from dataclasses import dataclass

from .model import Cursor, rank
from .search import Search
from .tree import NodeId, Tree


@dataclass(slots=True)
class _HistoryEntry:
    root: NodeId
    checkpoint: object


class Session:
    """Own the committed model state, discovered tree, search, and undo history."""

    def __init__(self, cursor: Cursor) -> None:
        self.cursor = cursor
        self.tree = Tree()
        self.root = self.tree.root
        self._root_checkpoint = cursor.checkpoint()
        self._history: list[_HistoryEntry] = []
        self.search = Search(self.tree, self._evaluate, root=self.root)

    @property
    def undo_depth(self) -> int:
        return len(self._history)

    def _evaluate(self, tree: Tree, node_id: NodeId):
        self.cursor.restore(self._root_checkpoint)
        try:
            for token in tree.path_from(self.root, node_id):
                self.cursor.observe(token)
            return rank(self.cursor.predict())
        finally:
            self.cursor.restore(self._root_checkpoint)

    def accept(self, tokens: tuple[int, ...]) -> NodeId:
        """Commit one path as one undoable action and make it the search root."""
        if not tokens:
            return self.root

        self._history.append(_HistoryEntry(self.root, self._root_checkpoint))
        current = self.root

        for token in tokens:
            node = self.tree[current]
            if node.distribution is None:
                node.distribution = rank(self.cursor.predict())
            rank_index = node.distribution.tokens.index(token)
            current = self.tree.child(current, rank_index)
            self.cursor.observe(token)

        self.root = current
        self._root_checkpoint = self.cursor.checkpoint()
        self.search.reset(self.root)
        return self.root

    def undo(self) -> bool:
        """Restore the previous committed root without discarding tree knowledge."""
        if not self._history:
            return False
        previous = self._history.pop()
        self.cursor.restore(previous.checkpoint)
        self.root = previous.root
        self._root_checkpoint = previous.checkpoint
        self.search.reset(self.root)
        return True
