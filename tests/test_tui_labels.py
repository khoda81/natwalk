from __future__ import annotations

import unittest

from natwalk.tree import Distribution, Tree
from natwalk.tui import _TreeRenderer
from natwalk.view import View, partition_rows


class TreeLabelTests(unittest.TestCase):
    def test_forest_preview_replaces_semantic_ellipsis_but_keeps_incoming_edge(self) -> None:
        tree = Tree(
            Distribution(
                tokens=(10, 11, 12),
                probabilities=(0.6, 0.3, 0.1),
            )
        )
        rows = partition_rows(tree, View(), row_limit=2)
        renderer = _TreeRenderer(
            tree,
            tree.root,
            View(),
            rows,
            str,
            selected_rank=-1,
            suggestion=None,
            max_preview_tokens=64,
        )

        line = renderer.render(columns=100, color=False)[1]

        self.assertIn("└─ · 11 · …", line)
        self.assertNotIn("└─…", line)

    def test_forest_without_preview_keeps_semantic_ellipsis(self) -> None:
        tree = Tree(
            Distribution(
                tokens=(10, 11, 12),
                probabilities=(0.6, 0.3, 0.1),
            )
        )
        rows = partition_rows(tree, View(), row_limit=2)
        renderer = _TreeRenderer(
            tree,
            tree.root,
            View(),
            rows,
            str,
            selected_rank=-1,
            suggestion=None,
            max_preview_tokens=0,
        )

        line = renderer.render(columns=100, color=False)[1]

        self.assertIn("└─…", line)


if __name__ == "__main__":
    unittest.main()
