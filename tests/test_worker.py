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

    def test_waiting_foreground_runs_before_next_search_step(self) -> None:
        first_step_started = threading.Event()
        release_first_step = threading.Event()
        foreground_acquired = threading.Event()
        release_foreground = threading.Event()
        second_step_started = threading.Event()
        evaluations = 0

        def evaluate(tree: Tree, node: int) -> Distribution:
            nonlocal evaluations
            evaluations += 1
            if evaluations == 1:
                first_step_started.set()
                self.assertTrue(release_first_step.wait(1.0))
            elif evaluations == 2:
                second_step_started.set()
            return Distribution((), ())

        tree = Tree()
        tree[tree.root].distribution = distribution(0.6, 0.4)
        search = Search(tree, evaluate)

        with SearchWorker(search) as worker:
            self.assertTrue(first_step_started.wait(1.0))

            def foreground() -> None:
                with worker.access():
                    foreground_acquired.set()
                    self.assertTrue(release_foreground.wait(1.0))

            thread = threading.Thread(target=foreground)
            thread.start()
            release_first_step.set()

            self.assertTrue(foreground_acquired.wait(1.0))
            self.assertFalse(second_step_started.is_set())
            release_foreground.set()
            thread.join(1.0)
            self.assertFalse(thread.is_alive())

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
