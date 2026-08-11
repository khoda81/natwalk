from __future__ import annotations

import time
import unittest
from collections.abc import Sequence

from natwalk.engine import Commit, EngineClient


class ToyCursor:
    def __init__(self, *, slow_child: bool = False) -> None:
        self.path: tuple[int, ...] = ()
        self.slow_child = slow_child

    def predict(self) -> Sequence[float]:
        if self.slow_child and self.path:
            time.sleep(10)
        table = {
            (): (0.6, 0.3, 0.1),
            (0,): (0.8, 0.2),
            (1,): (0.5, 0.5),
        }
        return table.get(self.path, ())

    def observe(self, token: int) -> None:
        self.path = (*self.path, token)

    def checkpoint(self) -> object:
        return self.path

    def restore(self, checkpoint: object) -> None:
        self.path = checkpoint


def toy_cursor() -> ToyCursor:
    return ToyCursor()


def slow_cursor() -> ToyCursor:
    return ToyCursor(slow_child=True)


def wait_until(client: EngineClient, predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        client.poll()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("engine condition did not become true before timeout")


class EngineTests(unittest.TestCase):
    def test_process_publishes_append_only_tree_replica(self) -> None:
        client = EngineClient(toy_cursor)
        client.start()
        try:
            wait_until(client, lambda: client.replica.next_node >= 2)
            self.assertEqual(client.tree[0].distribution.tokens, (0, 1, 2))
            self.assertGreaterEqual(len(client.tree.nodes), 2)
        finally:
            client.terminate()

    def test_duplicate_write_command_executes_once(self) -> None:
        client = EngineClient(toy_cursor)
        client.start()
        try:
            wait_until(client, lambda: client.root == 0)
            command = Commit(41, (0,))
            client.send(command)
            wait_until(client, lambda: client.done(41) is not None)
            self.assertEqual(client.root, 1)
            self.assertEqual(client.history_depth, 1)

            client.send(command)
            wait_until(client, lambda: client.done(41) is not None)
            self.assertEqual(client.root, 1)
            self.assertEqual(client.history_depth, 1)
        finally:
            client.terminate()

    def test_undo_is_owned_by_engine_process(self) -> None:
        client = EngineClient(toy_cursor)
        client.start()
        try:
            wait_until(client, lambda: client.root == 0)
            commit = client.commit((0,))
            wait_until(client, lambda: client.done(commit) is not None and client.root != 0)
            self.assertEqual(client.history_depth, 1)

            undo = client.undo()
            wait_until(client, lambda: client.done(undo) is not None and client.root == 0)
            self.assertEqual(client.history_depth, 0)
        finally:
            client.terminate()

    def test_terminate_kills_current_model_call(self) -> None:
        client = EngineClient(slow_cursor)
        client.start()
        wait_until(client, lambda: client.root == 0)
        client.inspect(0, 0)
        time.sleep(0.05)

        started = time.monotonic()
        client.terminate()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1.0)
        self.assertFalse(client.alive)


if __name__ == "__main__":
    unittest.main()
