from __future__ import annotations

import time
import unittest
from collections.abc import Sequence

from natwalk.tui import (
    App,
    _cell_width,
    _clip,
    _context_spans,
    _context_text,
    _fit,
    _format_forest_summary,
    _format_tree_row,
    _wrap_spans,
)
from natwalk.view import CompactRow


class SlowChildCursor:
    def __init__(self) -> None:
        self.path: tuple[int, ...] = ()

    def predict(self) -> Sequence[float]:
        if self.path:
            time.sleep(10)
        return (0.6, 0.4) if not self.path else ()

    def observe(self, token: int) -> None:
        self.path = (*self.path, token)

    def checkpoint(self) -> object:
        return self.path

    def restore(self, checkpoint: object) -> None:
        self.path = checkpoint


def slow_child_cursor() -> SlowChildCursor:
    return SlowChildCursor()


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

    def test_context_wrap_keeps_suggestion_span_intact(self) -> None:
        lines = _wrap_spans(
            (("The interesting thing is", ""), (" that it", "7")),
            24,
            color=False,
        )

        self.assertEqual(lines, ("The interesting thing is", " that it"))

    def test_context_can_render_exact_backend_text(self) -> None:
        text = _context_text(
            "The answer is",
            lambda token: str(token),
            (1, 2),
            lambda tokens: " definitely" if tokens == (1, 2) else "",
        )

        self.assertEqual(text, "The answer is definitely")

    def test_semantic_suggestion_gets_sequence_separator(self) -> None:
        spans = _context_spans("tie · t=0.21s", "acoustic_guitar", None)

        self.assertEqual(
            spans,
            (
                ("tie · t=0.21s", ""),
                (" · ", ""),
                ("acoustic_guitar", "1"),
            ),
        )

    def test_exact_text_suggestion_keeps_backend_spacing(self) -> None:
        decode = lambda tokens: " that it"
        spans = _context_spans("information theory is", " that it", decode)

        self.assertEqual(
            spans,
            (
                ("information theory is", ""),
                (" that it", "1"),
            ),
        )


class InteractiveAppTests(unittest.TestCase):
    def test_model_call_never_blocks_local_navigation(self) -> None:
        app = App(
            slow_child_cursor,
            str,
            title="test",
            context="",
            decode_tokens=None,
            max_tokens=8,
            budget_nats=1.0,
            budget_step=0.25,
            lines=8,
        )
        try:
            started = time.monotonic()
            self.assertFalse(app.handle_key("RIGHT"))
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertIsNotNone(app.pending)

            started = time.monotonic()
            self.assertTrue(app.handle_key("DOWN"))
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertEqual(app.view.selected_rank, 1)
        finally:
            app.engine.terminate()


if __name__ == "__main__":
    unittest.main()
