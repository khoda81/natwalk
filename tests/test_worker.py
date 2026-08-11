from __future__ import annotations

import threading
import time
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

        def evaluate(tree: Tree, parent: int, rank: int) -> Distribution:
            token = tree[parent].distribution.tokens[rank]
            path = (*tree.path(parent), token)
            visited.append(path)
            if len(visited) == 2:
                done.set()
            return table[path]

        tree = Tree(table[()])
        search = Search(tree, evaluate)
        with SearchWorker(search):
            self.assertTrue(done.wait(1.0))

        self.assertEqual(visited, [(0,), (1,)])
        self.assertEqual(search.frontier, [])

    def test_foreground_reset_wakes_idle_worker(self) -> None:
        first_step = threading.Event()
        second_step = threading.Event()

        class CountingSearch(Search):
            def __init__(self, tree, evaluate):
                self.steps = 0
                super().__init__(tree, evaluate)

            def step(self):
                result = super().step()
                self.steps += 1
                if self.steps == 1:
                    first_step.set()
                elif self.steps == 2:
                    second_step.set()
                return result

        tree = Tree(distribution(1.0))
        tree.put_child(0, 0, Distribution((), ()))
        search = CountingSearch(tree, lambda *_: Distribution((), ()))

        with SearchWorker(search) as worker:
            self.assertTrue(first_step.wait(1.0))
            with worker.access():
                search.reset(tree.root)
            self.assertTrue(second_step.wait(1.0))

    def test_waiting_foreground_runs_before_next_search_step(self) -> None:
        first_step_started = threading.Event()
        release_first_step = threading.Event()
        foreground_acquired = threading.Event()
        release_foreground = threading.Event()
        second_step_started = threading.Event()
        evaluations = 0

        def evaluate(tree: Tree, parent: int, rank: int) -> Distribution:
            nonlocal evaluations
            evaluations += 1
            if evaluations == 1:
                first_step_started.set()
                self.assertTrue(release_first_step.wait(1.0))
            elif evaluations == 2:
                second_step_started.set()
            return Distribution((), ())

        tree = Tree(distribution(0.6, 0.4))
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

    def test_request_stop_does_not_wait_for_current_search_step(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def evaluate(tree: Tree, parent: int, rank: int) -> Distribution:
            started.set()
            release.wait(1.0)
            return Distribution((), ())

        worker = SearchWorker(Search(Tree(distribution(1.0)), evaluate))
        worker.start()
        self.assertTrue(started.wait(1.0))

        before = time.monotonic()
        worker.request_stop()
        elapsed = time.monotonic() - before

        self.assertLess(elapsed, 0.1)
        release.set()
        worker.join()

    def test_worker_failure_is_raised_in_foreground(self) -> None:
        failed = threading.Event()

        def evaluate(tree: Tree, parent: int, rank: int) -> Distribution:
            failed.set()
            raise ValueError("model exploded")

        tree = Tree(distribution(1.0))
        search = Search(tree, evaluate)
        with SearchWorker(search) as worker:
            self.assertTrue(failed.wait(1.0))
            with self.assertRaisesRegex(ValueError, "model exploded"):
                with worker.access():
                    pass


if __name__ == "__main__":
    unittest.main()
