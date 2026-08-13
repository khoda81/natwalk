from __future__ import annotations

import time
import unittest
from collections.abc import Sequence
from unittest.mock import patch

from natwalk.tui import App, _tree_viewport
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
    def test_viewport_prefetches_before_old_fixed_band(self) -> None:
        app = App(
            wide_cursor,
            str,
            title="test",
            context="",
            decode_tokens=None,
            max_tokens=8,
            budget_nats=1.0,
            budget_step=0.25,
            lines=50,
        )
        try:
            distribution = app.tree[app.root].distribution
            self.assertEqual(len(distribution), 256)
            self.assertEqual(distribution.revealed, 128)

            app.view = View(node=app.root, first_rank=70)
            self.assertLess(app.view.first_rank, distribution.revealed - 32)
            visible, reveal_demands = _tree_viewport(
                app.tree,
                app.view,
                tree_lines=50,
            )

            self.assertEqual(reveal_demands, ((app.root, 256),))
            self.assertFalse(
                any(
                    row.forest
                    and row.parent == app.root
                    and row.forest_start >= distribution.revealed
                    for row in visible
                )
            )

            with (
                patch("natwalk.tui._dimensions", return_value=(120, 100)),
                patch("natwalk.tui._write_frame"),
            ):
                app.render()
            wait_for(app, lambda: distribution.revealed == 256)
        finally:
            app.engine.terminate()

    def test_down_at_revealed_end_still_requests_next_page(self) -> None:
        app = App(
            wide_cursor,
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
            distribution = app.tree[app.root].distribution
            self.assertEqual(distribution.revealed, 128)
            app.view = View(node=app.root, first_rank=127)

            self.assertFalse(app.handle_key("DOWN"))
            wait_for(app, lambda: distribution.revealed == 256)

            self.assertTrue(app.handle_key("DOWN"))
            self.assertEqual(app.view.first_rank, 128)
        finally:
            app.engine.terminate()


if __name__ == "__main__":
    unittest.main()
