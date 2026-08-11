from __future__ import annotations

import unittest

from natwalk.tui import (
    _cell_width,
    _clip,
    _context_text,
    _fit,
    _format_forest_summary,
    _format_tree_row,
    _wrap_spans,
)
from natwalk.view import CompactRow


class TerminalWidthTests(unittest.TestCase):
    def test_clip_respects_terminal_cells(self) -> None:
        self.assertEqual(_clip("abcdef", 4), "abc…")
        self.assertLessEqual(_cell_width(_clip("界界界", 4)), 4)
        self.assertLessEqual(_cell_width(_clip("e\u0301e\u0301e\u0301", 2)), 2)

    def test_fit_pads_by_display_width(self) -> None:
        fitted = _fit("界", 4)
        self.assertEqual(_cell_width(fitted), 4)

    def test_tree_row_never_exceeds_terminal_width(self) -> None:
        row = CompactRow(
            parent=0,
            rank=0,
            depth=2,
            ancestor_last=(False, True),
            is_last=False,
            tokens=(1, 2, 3),
            edge_nats=1.25,
            path_nats=13.506,
            child=4,
            open_ended=True,
        )
        labels = {
            1: "program_119",
            2: "C♯7",
            3: "界界界界界界界界界界",
        }
        for columns in (24, 40, 80, 180):
            with self.subTest(columns=columns):
                line = _format_tree_row(
                    row,
                    labels.__getitem__,
                    selected=True,
                    columns=columns,
                    color=False,
                )
                self.assertLessEqual(_cell_width(line), columns)

    def test_forest_summary_never_exceeds_terminal_width(self) -> None:
        for columns in (24, 40, 80, 180):
            line = _format_forest_summary(
                "↓",
                151_908,
                2.75,
                columns=columns,
                color=False,
            )
            self.assertLessEqual(_cell_width(line), columns)

    def test_context_wrap_keeps_suggestion_inline(self) -> None:
        lines = _wrap_spans(
            (("The interesting thing is", ""), (" that it", "7")),
            24,
            color=False,
        )

        self.assertEqual(lines, ("The interesting thing is ", "that it"))

    def test_context_can_render_exact_backend_text(self) -> None:
        text = _context_text(
            "The answer is",
            lambda token: str(token),
            (1, 2),
            lambda tokens: " definitely" if tokens == (1, 2) else "",
        )

        self.assertEqual(text, "The answer is definitely")


if __name__ == "__main__":
    unittest.main()
