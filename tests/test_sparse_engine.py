from __future__ import annotations

import time
import unittest
from collections.abc import Sequence

from natwalk.engine import EngineClient


class WideCursor:
    def __init__(self) -> None:
        self.path: tuple[int, ...] = ()

    def predict(self) -> Sequence[float]:
        if self.path:
            return ()
        return (1.0 / 200,) * 200

    def observe(self, token: int) -> None:
        self.path = (*self.path, token)

    def checkpoint(self) -> object:
        return self.path

    def restore(self, checkpoint: object) -> None:
        self.path = checkpoint


def wide_cursor() -> WideCursor:
    return WideCursor()


def wait_for(client: EngineClient, predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        client.poll()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("engine condition did not become true before timeout")


class SparseEngineTests(unittest.TestCase):
    def test_high_rank_advance_keeps_replica_path_derivable_without_revealing_gap(self) -> None:
        # Root storage is 200 entries * 12 packed bytes. Hitting the cap at the
        # root keeps background search out of the way while explicit navigation
        # remains authoritative.
        client = EngineClient(wide_cursor, max_tree_bytes=2400)
        client.start()
        try:
            wait_for(client, lambda: client.root == 0)
            root_distribution = client.tree[0].distribution
            self.assertEqual(root_distribution.revealed, 128)

            command = client.advance((145,))
            wait_for(client, lambda: client.done(command) is not None)

            self.assertEqual(client.tree.path(client.root), (145,))
            self.assertEqual(root_distribution.revealed, 128)
            self.assertEqual(root_distribution.token(145), 145)
            with self.assertRaisesRegex(IndexError, "revealed or pinned"):
                root_distribution.token(144)
        finally:
            client.terminate()


if __name__ == "__main__":
    unittest.main()
