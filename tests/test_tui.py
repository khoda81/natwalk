from __future__ import annotations

import math
import re
import time
import unittest
from collections.abc import Sequence

from natwalk.tree import Distribution, Tree
from natwalk.tui import (
    _PREDICTION_STYLE,
    _SUGGESTION_STYLE,
    App,
    _cell_width,
    _clip,
    _context_spans,
    _context_text,
    _fit,
    _format_forest_summary,
    _format_tree_row,
    _relative_probability,
    _row_display_nats,
    _row_preview,
    _row_separator_nats,
    _tree_viewport,
    _wrap_spans,
)
from natwalk.view import CompactRow, View, partition_rows

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


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


class NavigationCursor:
    def __init__(self) -> None:
        self.path: tuple[int, ...] = ()

    def predict(self) -> Sequence[float]:
        return {
            (): (1.0,),
            (0,): (1.0,),
            (0, 0): (1.0,),
        }.get(self.path, ())

    def observe(self, token: int) -> None:
        self.path = (*self.path, token)

    def checkpoint(self) -> object:
        return self.path

    def restore(self, checkpoint: object) -> None:
        self.path = checkpoint


def slow_child_cursor() -> SlowChildCursor:
    return SlowChildCursor()


def navigation_cursor() -> NavigationCursor:
    return NavigationCursor()


def wait_for_app(app: App, predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.poll()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("app condition did not become true before timeout")


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

    def test_colored_tree_structure_does_not_move_nat_column(self) -> None:
        row = CompactRow(
            parent=0,
            rank=0,
            depth=2,
            ancestor_last=(False, False),
            is_last=False,
            tokens=(1, 2),
            edge_nats=1.25,
            path_nats=3.5,
            child=4,
            open_ended=True,
        )
        kwargs = {
            "selected": False,
            "columns": 100,
            "display_nats": 3.5,
            "nat_reference": 1.0,
            "branch_nats": 1.25,
            "branch_reference": 1.0,
            "ancestor_branch_nats": (1.1, 1.2),
            "ancestor_columns": (4, 12),
            "branch_column": 20,
        }

        plain = _format_tree_row(row, str, color=False, **kwargs)
        colored = _format_tree_row(row, str, color=True, **kwargs)

        self.assertEqual(_ANSI.sub("", colored), plain)
        self.assertTrue(plain.endswith("    3.500 nat"))

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
            (("The interesting thing is", ""), (" that it", _SUGGESTION_STYLE)),
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
                ("acoustic_guitar", _SUGGESTION_STYLE),
            ),
        )

    def test_exact_text_suggestion_keeps_backend_spacing(self) -> None:
        def decode(_tokens: tuple[int, ...]) -> str:
            return " that it"

        spans = _context_spans("information theory is", " that it", decode)

        self.assertEqual(
            spans,
            (
                ("information theory is", ""),
                (" that it", _SUGGESTION_STYLE),
            ),
        )

    def test_relative_probability_is_probability_ratio(self) -> None:
        self.assertEqual(_relative_probability(2.0, 2.0), 1.0)
        self.assertAlmostEqual(_relative_probability(3.0, 2.0), math.exp(-1.0))
        self.assertEqual(_relative_probability(math.inf, 2.0), 0.0)

    def test_row_number_stays_relative_to_committed_root(self) -> None:
        root = Distribution(tokens=(10,), probabilities=(0.5,))
        tree = Tree(root)
        child = tree.put_child(0, 0, Distribution(tokens=(20,), probabilities=(0.8,)))
        view = View(node=child)
        row = CompactRow(
            parent=child,
            rank=0,
            depth=0,
            ancestor_last=(),
            is_last=True,
            tokens=(20,),
            edge_nats=-math.log(0.8),
            path_nats=-math.log(0.8),
            child=None,
            open_ended=True,
        )

        self.assertAlmostEqual(
            _row_display_nats(tree, tree.root, view, row),
            -math.log(0.5) - math.log(0.8),
        )

    def test_separator_nats_use_known_radix_ranks_without_token_search(self) -> None:
        class NoIndexTuple(tuple):
            def index(self, *_args, **_kwargs):
                raise AssertionError("renderer searched the vocabulary")

        tree = Tree(
            Distribution(
                tokens=NoIndexTuple((10, 11)),
                probabilities=(0.6, 0.4),
            )
        )
        child = tree.put_child(
            0,
            0,
            Distribution(
                tokens=NoIndexTuple((20, 21)),
                probabilities=(0.7, 0.3),
            ),
        )
        row = CompactRow(
            parent=0,
            rank=0,
            depth=0,
            ancestor_last=(),
            is_last=False,
            tokens=(10, 20),
            edge_nats=-math.log(0.6),
            path_nats=-math.log(0.6 * 0.7),
            child=child,
            ranks=(0, 0),
        )

        self.assertEqual(
            _row_separator_nats(tree, View(), row),
            (-math.log(0.6) - math.log(0.7),),
        )

    def test_row_preview_walks_only_known_greedy_children(self) -> None:
        tree = Tree(Distribution(tokens=(10,), probabilities=(1.0,)))
        first = tree.put_child(
            tree.root,
            0,
            Distribution(tokens=(20, 21), probabilities=(0.75, 0.25)),
        )
        second = tree.put_child(
            first,
            0,
            Distribution(tokens=(30,), probabilities=(1.0,)),
        )
        row = CompactRow(
            parent=tree.root,
            rank=0,
            depth=0,
            ancestor_last=(),
            is_last=True,
            tokens=(10,),
            edge_nats=0.0,
            path_nats=0.0,
            child=first,
            open_ended=True,
            ranks=(0,),
        )
        before = len(tree.nodes)

        tokens, separator_nats, complete = _row_preview(tree, View(), row)

        self.assertEqual(tokens, (20, 30))
        self.assertEqual(
            separator_nats,
            (-math.log(0.75), -math.log(0.75)),
        )
        self.assertFalse(complete)
        self.assertEqual(len(tree.nodes), before)
        self.assertIsNone(tree.child(second, 0))

    def test_forest_preview_uses_its_most_probable_concrete_member(self) -> None:
        tree = Tree(
            Distribution(
                tokens=(10, 11, 12),
                probabilities=(0.6, 0.3, 0.1),
            )
        )
        forest = next(row for row in partition_rows(tree, View(), row_limit=2) if row.forest)

        tokens, separator_nats, complete = _row_preview(tree, View(), forest)

        self.assertEqual(tokens, (11,))
        self.assertEqual(separator_nats, (-math.log(0.3),))
        self.assertFalse(complete)

    def test_preview_replaces_open_ended_ellipsis_and_dims_only_nodes(self) -> None:
        row = CompactRow(
            parent=0,
            rank=0,
            depth=0,
            ancestor_last=(),
            is_last=True,
            tokens=(1,),
            edge_nats=1.0,
            path_nats=1.0,
            child=1,
            open_ended=True,
        )
        kwargs = {
            "selected": False,
            "columns": 100,
            "display_nats": 1.0,
            "nat_reference": 1.0,
            "branch_nats": 1.0,
            "branch_reference": 1.0,
            "preview_tokens": (2, 3),
            "preview_separator_nats": (1.5, 2.0),
            "preview_complete": False,
        }

        plain = _format_tree_row(row, str, color=False, **kwargs)
        colored = _format_tree_row(row, str, color=True, **kwargs)

        self.assertIn("1 · 2 · 3 · …", plain)
        self.assertNotIn("1 · …", plain)
        self.assertIn(f"\033[{_PREDICTION_STYLE}m2\033[0m", colored)
        self.assertNotIn(f"\033[{_PREDICTION_STYLE}m · \033[0m", colored)

    def test_forest_keeps_its_semantic_ellipsis_before_preview(self) -> None:
        row = CompactRow(
            parent=0,
            rank=-1,
            depth=0,
            ancestor_last=(),
            is_last=True,
            tokens=(),
            edge_nats=1.0,
            path_nats=1.0,
            child=None,
            forest_count=2,
            forest_start=1,
        )

        line = _format_tree_row(
            row,
            str,
            selected=False,
            columns=100,
            color=False,
            preview_tokens=(11,),
            preview_separator_nats=(2.0,),
            preview_complete=False,
        )

        self.assertIn("… · 11 · …", line)

    def test_selected_sibling_stays_inside_tree_viewport(self) -> None:
        tree = Tree(
            Distribution(
                tokens=(0, 1, 2),
                probabilities=(0.99, 0.009, 0.001),
            )
        )
        tree.put_child(
            tree.root,
            0,
            Distribution(
                tokens=tuple(range(100, 150)),
                probabilities=(0.02,) * 50,
            ),
        )
        view = View(selected_rank=1)

        start, above, visible = _tree_viewport(
            tree,
            view,
            selected=1,
            tree_lines=8,
        )

        self.assertEqual(start, 1)
        self.assertEqual(above, 1)
        self.assertTrue(
            any(row.parent == tree.root and row.rank == 1 and not row.forest for row in visible)
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
            self.assertEqual(len(app.pending), 1)

            started = time.monotonic()
            self.assertTrue(app.handle_key("DOWN"))
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertEqual(app.view.selected_rank, 1)
        finally:
            app.engine.terminate()

    def test_right_and_left_move_model_search_and_view_root_together(self) -> None:
        app = App(
            navigation_cursor,
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
            initial_root = app.root
            app.handle_key("RIGHT")
            wait_for_app(app, lambda: not app.pending and app.root != initial_root)

            child = app.root
            self.assertEqual(app.view.node, child)
            self.assertEqual(app.tree.path(child), (0,))
            self.assertEqual(app.engine.rewind_depth, 1)

            app.handle_key("LEFT")
            wait_for_app(app, lambda: not app.pending and app.root == initial_root)

            self.assertEqual(app.view.node, initial_root)
            self.assertEqual(app.engine.rewind_depth, 0)
            self.assertIn(child, range(len(app.tree.nodes)))
        finally:
            app.engine.terminate()

    def test_known_navigation_queues_ahead_of_engine_without_snapping_back(self) -> None:
        app = App(
            navigation_cursor,
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
            wait_for_app(app, lambda: len(app.tree.nodes) >= 4)
            confirmed_root = app.root

            self.assertTrue(app.handle_key("RIGHT"))
            self.assertTrue(app.handle_key("RIGHT"))
            self.assertTrue(app.handle_key("RIGHT"))

            self.assertEqual(app.tree.path(app.view.node), (0, 0, 0))
            self.assertEqual(app.root, confirmed_root)
            self.assertEqual(len(app.pending), 3)

            self.assertTrue(app.handle_key("LEFT"))
            self.assertEqual(app.tree.path(app.view.node), (0, 0))
            self.assertEqual(len(app.pending), 4)

            wait_for_app(app, lambda: not app.pending)
            self.assertEqual(app.tree.path(app.root), (0, 0))
            self.assertEqual(app.view.node, app.root)
        finally:
            app.engine.terminate()


if __name__ == "__main__":
    unittest.main()
