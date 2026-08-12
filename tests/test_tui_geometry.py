from __future__ import annotations

import unittest

from natwalk.tree import Distribution, Tree
from natwalk.tui import _TreeRenderer, _cell_width, _path_branch_column
from natwalk.view import BranchRole, View, partition_rows


def distribution(*probabilities: float) -> Distribution:
    return Distribution(
        tokens=tuple(range(len(probabilities))),
        probabilities=probabilities,
    )


def render_rows(
    tree: Tree,
    rows,
    *,
    view: View = View(),
    describe=str,
    columns: int = 100,
) -> tuple[str, ...]:
    renderer = _TreeRenderer(
        tree,
        tree.root,
        view,
        rows,
        describe,
        selected_rank=-1,
        suggestion=None,
        max_preview_tokens=0,
    )
    return renderer.render(columns=columns, color=False)


class TreeGeometryTests(unittest.TestCase):
    def test_root_continuation_and_siblings_have_distinct_semantic_roles(self) -> None:
        tree = Tree(distribution(0.6, 0.3, 0.1))
        rows = partition_rows(tree, View(), row_limit=3)

        self.assertIs(rows[0].branch_role, BranchRole.CONTINUATION)
        self.assertTrue(all(row.branch_role is BranchRole.SIBLING for row in rows[1:]))

        lines = render_rows(tree, rows)
        self.assertTrue(lines[0].startswith("  ┬ 0"))
        self.assertTrue(lines[1].startswith("  ├─1"))
        self.assertTrue(lines[2].startswith("  └─2"))

    def test_scrolling_does_not_promote_first_visible_sibling_to_continuation(self) -> None:
        tree = Tree(distribution(0.6, 0.3, 0.1))
        view = View()
        rows = partition_rows(tree, view, row_limit=2, first_rank=1)

        self.assertTrue(all(row.branch_role is BranchRole.SIBLING for row in rows))

        lines = render_rows(tree, rows, view=view)
        self.assertTrue(lines[0].startswith("  ├─1"))
        self.assertTrue(lines[1].startswith("  └─2"))
        self.assertNotIn("┬ ", lines[0][:4])

    def test_nested_split_uses_inline_junction_then_ordinary_sibling_branch(self) -> None:
        tree = Tree(distribution(0.6, 0.4))
        first = tree.put_child(tree.root, 0, distribution(0.7, 0.3))
        self.assertIsNotNone(first)
        rows = partition_rows(tree, View(), row_limit=3)

        lines = render_rows(tree, rows)
        self.assertTrue(lines[0].startswith("  ┬ 0 ┬ 0"))
        self.assertIn("└─1", lines[1])
        self.assertTrue(lines[2].startswith("  └─1"))

    def test_branch_column_uses_unicode_terminal_cells(self) -> None:
        labels = {1: "界", 2: "e\u0301"}

        # 2 connector cells + 2 wide-character cells + 3 separator cells
        # + 1 combining-character label cell + 1 cell to the junction.
        self.assertEqual(_path_branch_column((1, 2), labels.__getitem__), 9)
        self.assertEqual(_cell_width(labels[1]), 2)
        self.assertEqual(_cell_width(labels[2]), 1)

    def test_offscreen_branch_is_reported_without_remapping_geometry(self) -> None:
        tree = Tree(distribution(1.0))
        tree.put_child(tree.root, 0, distribution(0.6, 0.4))
        rows = partition_rows(tree, View(), row_limit=2)

        lines = render_rows(
            tree,
            rows,
            describe=lambda _token: "x" * 100,
            columns=40,
        )

        self.assertIn("branch off-screen", lines[1])
        self.assertNotIn("├─", lines[1])
        self.assertNotIn("└─", lines[1])
        self.assertLessEqual(_cell_width(lines[1]), 40)

    def test_degenerate_terminal_width_still_fits(self) -> None:
        tree = Tree(distribution(1.0))
        rows = partition_rows(tree, View(), row_limit=1)

        line = render_rows(tree, rows, columns=8)[0]
        self.assertLessEqual(_cell_width(line), 8)


if __name__ == "__main__":
    unittest.main()
