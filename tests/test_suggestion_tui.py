from __future__ import annotations

import time
import unittest
from collections.abc import Sequence

from natwalk.query import suggestion_tokens
from natwalk.tui import App


class TerminalAfterOneTokenCursor:
    def __init__(self) -> None:
        self.path: tuple[int, ...] = ()

    def predict(self) -> Sequence[float]:
        return (1.0,) if not self.path else ()

    def observe(self, token: int) -> None:
        self.path = (*self.path, token)

    def checkpoint(self) -> object:
        return self.path

    def restore(self, checkpoint: object) -> None:
        self.path = checkpoint


def terminal_after_one_token() -> TerminalAfterOneTokenCursor:
    return TerminalAfterOneTokenCursor()


def wait_for(app: App, predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.poll()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("app condition did not become true before timeout")


class SuggestionTuiTests(unittest.TestCase):
    def test_space_after_completed_accept_uses_new_view_without_render(self) -> None:
        app = App(
            terminal_after_one_token,
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
            self.assertEqual(
                suggestion_tokens(app.tree, app.view.node, app.suggestion),
                (0,),
            )
            self.assertFalse(app.handle_key(" "))
            self.assertEqual(len(app.pending), 1)

            wait_for(app, lambda: not app.pending)

            self.assertEqual(len(app.tree[app.view.node].distribution), 0)
            self.assertIsNone(app.suggestion)

            # This is the old crash window: command completion moved the view,
            # but no render has occurred before another buffered Space arrives.
            self.assertFalse(app.handle_key(" "))
            self.assertFalse(app.pending)
            self.assertFalse(app.poll())
        finally:
            app.engine.terminate()


if __name__ == "__main__":
    unittest.main()
