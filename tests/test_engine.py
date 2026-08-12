from __future__ import annotations

import time
import unittest
from collections.abc import Sequence

from natwalk.engine import Advance, EngineClient


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
            distribution = client.tree[0].distribution
            self.assertEqual(
                tuple(distribution.token(rank) for rank in range(distribution.revealed)),
                (0, 1, 2),
            )
            self.assertGreaterEqual(len(client.tree.nodes), 2)
        finally:
            client.terminate()

    def test_tree_memory_limit_pauses_background_search_but_not_commands(self) -> None:
        # Root authoritative payload is 3 entries * 12 packed bytes = 36 bytes.
        client = EngineClient(toy_cursor, max_tree_bytes=36)
        client.start()
        try:
            wait_until(client, lambda: client.root == 0)
            time.sleep(0.05)
            client.poll()
            self.assertEqual(len(client.tree.nodes), 1)

            command = client.advance((0,))
            wait_until(client, lambda: client.done(command) is not None)
            self.assertEqual(client.tree.path(client.root), (0,))
            self.assertEqual(len(client.tree.nodes), 2)
        finally:
            client.terminate()

    def test_duplicate_navigation_command_executes_once(self) -> None:
        client = EngineClient(toy_cursor)
        client.start()
        try:
            wait_until(client, lambda: client.root == 0)
            command = Advance(41, (0,))
            client.send(command)
            wait_until(client, lambda: client.done(41) is not None)
            self.assertEqual(client.tree.path(client.root), (0,))
            self.assertEqual(client.rewind_depth, 1)

            client.send(command)
            wait_until(client, lambda: client.done(41) is not None)
            self.assertEqual(client.tree.path(client.root), (0,))
            self.assertEqual(client.rewind_depth, 1)
        finally:
            client.terminate()

    def test_advance_records_one_rewind_point_per_token(self) -> None:
        client = EngineClient(toy_cursor)
        client.start()
        try:
            wait_until(client, lambda: client.root == 0)
            advance = client.advance((0, 1))
            wait_until(client, lambda: client.done(advance) is not None)

            endpoint = client.root
            self.assertEqual(client.tree.path(endpoint), (0, 1))
            self.assertEqual(client.rewind_depth, 2)

            rewind = client.rewind()
            wait_until(client, lambda: client.done(rewind) is not None)
            self.assertEqual(client.tree.path(client.root), (0,))
            self.assertEqual(client.rewind_depth, 1)
            self.assertIn(endpoint, range(len(client.tree.nodes)))

            rewind = client.rewind()
            wait_until(client, lambda: client.done(rewind) is not None)
            self.assertEqual(client.root, client.tree.root)
            self.assertEqual(client.rewind_depth, 0)
        finally:
            client.terminate()

    def test_terminate_kills_current_model_call(self) -> None:
        client = EngineClient(slow_cursor)
        client.start()
        wait_until(client, lambda: client.root == 0)
        client.advance((0,))
        time.sleep(0.05)

        started = time.monotonic()
        client.terminate()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1.0)
        self.assertFalse(client.alive)


if __name__ == "__main__":
    unittest.main()
