from __future__ import annotations

import time
import unittest
from collections.abc import Sequence

from natwalk.tui import App
from natwalk.view import View


class WideCursor:
    def __init__(self) -> None:
        self.path: tuple[int, ...] = ()

    def predict(self) -> Sequence[float]:
        if self.path:
            return ()
        weights = tuple(range(256, 0, -1))
        total = sum(weights)
        return tuple(weight / total for weight in weights)

    def observe(self, token: int) -> None:
        self.path = (*self.path, token)

    def checkpoint(self) -> object:
        return self.path

    def restore(self, checkpoint: object) -> None:
        self.path = checkpoint


def wide_cursor() -> WideCursor:
    return WideCursor()


def wait_for(app: App, predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.poll()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("app condition did not become true before timeout")


class ProgressiveTuiTests(unittest.TestCase):
    def test_tall_viewport_prefetches_next_page_before_visible_boundary(self) -> None:
        app = App(
            wide_cursor,
            str,
            title="test",
            context="",
            decode_tokens=None,
            max_tokens=8,
            budget_nats=1.0,
            budget_step=0.25,
            lines=64,
        )
        try:
            distribution = app.tree[app.root].distribution
            self.assertEqual(len(distribution), 256)
            self.assertEqual(distribution.revealed, 128)

            # The selected rank itself is nowhere near the revealed boundary.
            # A tall viewport nevertheless needs enough read-ahead to keep
            # that boundary out of the ordinary scrolling window.
            app.view = View(node=app.root, selected_rank=31)
            self.assertTrue(app.handle_key("DOWN"))
            self.assertEqual(app.view.selected_rank, 32)
            self.assertEqual(distribution.revealed, 128)

            wait_for(app, lambda: distribution.revealed == 256)

            app.view = View(node=app.root, selected_rank=127)
            self.assertTrue(app.handle_key("DOWN"))
            self.assertEqual(app.view.selected_rank, 128)
        finally:
            app.engine.terminate()


if __name__ == "__main__":
    unittest.main()
