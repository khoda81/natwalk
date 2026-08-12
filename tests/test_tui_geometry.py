from __future__ import annotations

import unittest

from natwalk.tui import _cell_width, _format_tree_row, _path_branch_column
from natwalk.view import AncestorConnector, CompactRow


class TreeGeometryTests(unittest.TestCase):
    def test_root_branch_uses_same_tight_connector_as_nested_branches(self) -> None:
        row = CompactRow(
            parent=0,
            rank=0,
            depth=0,
            ancestors=(),
            is_last=False,
            edges=(),
            edge_nats=1.0,
            path_nats=1.0,
            child=None,
        )

        line = _format_tree_row(
            row,
            lambda _token: "token",
            selected=False,
            columns=80,
            color=False,
            tokens=(1,),
        )

        self.assertTrue(line.startswith("  ├─token"))
        self.assertFalse(line.startswith("  ├─ token"))

    def test_branch_column_is_measured_from_rendered_terminal_cells(self) -> None:
        labels = {1: "界", 2: "e\u0301"}
        rendered_prefix = "├─界 · e\u0301 "

        self.assertEqual(
            _path_branch_column((1, 2), labels.__getitem__),
            _cell_width(rendered_prefix),
        )

    def test_offscreen_branch_is_reported_instead_of_remapped(self) -> None:
        row = CompactRow(
            parent=0,
            rank=0,
            depth=2,
            ancestors=(
                AncestorConnector(is_last=False, nats=1.0),
                AncestorConnector(is_last=False, nats=1.0),
            ),
            is_last=False,
            edges=(),
            edge_nats=1.0,
            path_nats=4.015,
            child=None,
        )

        line = _format_tree_row(
            row,
            str,
            selected=False,
            columns=80,
            color=False,
            tokens=(1,),
            ancestor_columns=(12, 40),
            branch_column=100,
        )

        self.assertIn("branch off-screen", line)
        self.assertNotIn("├─", line)
        self.assertLessEqual(_cell_width(line), 80)
        self.assertTrue(line.endswith("    4.015 nat"))

    def test_degenerate_offscreen_fallback_still_fits_terminal(self) -> None:
        row = CompactRow(
            parent=0,
            rank=0,
            depth=1,
            ancestors=(AncestorConnector(is_last=False, nats=1.0),),
            is_last=False,
            edges=(),
            edge_nats=1.0,
            path_nats=4.015,
            child=None,
        )

        line = _format_tree_row(
            row,
            str,
            selected=False,
            columns=8,
            color=False,
            tokens=(1,),
            ancestor_columns=(4,),
            branch_column=100,
        )

        self.assertLessEqual(_cell_width(line), 8)
        self.assertNotIn("├─", line)


if __name__ == "__main__":
    unittest.main()
