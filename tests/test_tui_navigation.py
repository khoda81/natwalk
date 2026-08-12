from __future__ import annotations

import time
import unittest
from collections.abc import Sequence

from natwalk.tree import Distribution, Tree
from natwalk.tui import App, _decode_escape, _TreeRenderer
from natwalk.view import View, partition_rows


class NavigationCursor:
    def __init__(self) -> None:
        self.path: tuple[int, ...] = ()

    def predict(self) -> Sequence[float]:
        return {
            (): (0.4, 0.3, 0.2, 0.1),
            (0,): (1.0,),
            (1,): (1.0,),
            (2,): (1.0,),
            (3,): (1.0,),
        }.get(self.path, ())

    def observe(self, token: int) -> None:
        self.path = (*self.path, token)

    def checkpoint(self) -> object:
        return self.path

    def restore(self, checkpoint: object) -> None:
        self.path = checkpoint


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


class TreeNavigationTests(unittest.TestCase):
    def test_escape_decoder_supports_navigation_and_wheel(self) -> None:
        self.assertEqual(_decode_escape(b"\x1b[H"), "HOME")
        self.assertEqual(_decode_escape(b"\x1b[F"), "END")
        self.assertEqual(_decode_escape(b"\x1b[5~"), "PAGE_UP")
        self.assertEqual(_decode_escape(b"\x1b[6~"), "PAGE_DOWN")
        self.assertEqual(_decode_escape(b"\x1b[<64;10;20M"), "UP")
        self.assertEqual(_decode_escape(b"\x1b[<65;10;20M"), "DOWN")
        self.assertEqual(_decode_escape(b"\x1b[<68;10;20M"), "UP")

    def test_scrolled_root_uses_branch_connector_without_selection_marker(self) -> None:
        tree = Tree(Distribution(tokens=(0, 1, 2), probabilities=(0.6, 0.3, 0.1)))
        view = View(first_rank=1)
        rows = partition_rows(tree, view, row_limit=2)
        renderer = _TreeRenderer(
            tree,
            tree.root,
            view,
            rows,
            str,
            suggestion=None,
            max_preview_tokens=0,
        )

        lines = renderer.render(columns=100, color=False)

        self.assertTrue(lines[0].startswith("  ├─1"))
        self.assertTrue(all("❯" not in line for line in lines))

    def test_scroll_right_back_and_home_share_one_viewport_rank(self) -> None:
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
            app.engine.reveal(app.view.node, 4)
            wait_for_app(
                app,
                lambda: app.tree[app.view.node].distribution.revealed >= 4,
            )
            original_root = app.root

            self.assertTrue(app.handle_key("DOWN"))
            self.assertEqual(app.view.first_rank, 1)

            app.handle_key("RIGHT")
            wait_for_app(app, lambda: not app.pending and app.root != original_root)
            self.assertEqual(app.tree.path(app.root), (1,))

            app.handle_key("LEFT")
            wait_for_app(app, lambda: not app.pending and app.root == original_root)
            self.assertEqual(app.view.first_rank, 1)

            self.assertTrue(app.handle_key("HOME"))
            self.assertEqual(app.view.first_rank, 0)

            self.assertTrue(app.handle_key("END"))
            self.assertEqual(app.view.first_rank, 3)
        finally:
            app.engine.terminate()


if __name__ == "__main__":
    unittest.main()
