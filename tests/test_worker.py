from __future__ import annotations

import threading
import unittest

from natwalk.search import Search
from natwalk.tree import Distribution, Tree
from natwalk.worker import SearchWorker


def distribution(*probabilities: float) -> Distribution:
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    return Distribution(
        tokens=tuple(token for token, _ in ranked),
        probabilities=tuple(probability for _, probability in ranked),
    )


class WorkerTests(unittest.TestCase):
    def test_worker_preserves_synchronous_dijkstra_order(self) -> None:
        table = {
            (): distribution(0.6, 0.4),
            (0,): Distribution((), ()),
            (1,): Distribution((), ()),
        }
        visited: list[tuple[int, ...]] = []
        done = threading.Event()

        def evaluate(tree: Tree, node: int) -> Distribution:
            path = tree.path(node)
            visited.append(path)
            if len(visited) == len(table):
                done.set()
            return table[path]

        tree = Tree()
        search = Search(tree, evaluate)
        with SearchWorker(search):
            self.assertTrue(done.wait(1.0))

        self.assertEqual(visited, [(), (0,), (1,)])
        self.assertEqual(search.frontier, [])

    def test_foreground_reset_wakes_idle_worker(self) -> None:
        expanded = threading.Event()

        def evaluate(tree: Tree, node: int) -> Distribution:
            if node != tree.root:
                expanded.set()
            return Distribution((), ())

        tree = Tree()
        search = Search(tree, evaluate)
        worker = SearchWorker(search)

        with worker:
            with worker.access():
                tree[tree.root].distribution = distribution(1.0)
                search.reset(tree.root)
            self.assertTrue(expanded.wait(1.0))

    def test_worker_failure_is_raised_in_foreground(self) -> None:
        failed = threading.Event()

        def evaluate(tree: Tree, node: int) -> Distribution:
            if node == tree.root:
                return distribution(1.0)
            failed.set()
            raise ValueError("model exploded")

        tree = Tree()
        search = Search(tree, evaluate)
        with SearchWorker(search) as worker:
            self.assertTrue(failed.wait(1.0))
            with self.assertRaisesRegex(ValueError, "model exploded"):
                with worker.access():
                    pass


if __name__ == "__main__":
    unittest.main()
