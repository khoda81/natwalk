from __future__ import annotations

import heapq
import math
import random
import unittest
from collections.abc import Callable

from natwalk.search import Search
from natwalk.tree import Distribution, Tree

Table = dict[tuple[int, ...], Distribution]
EMPTY = Distribution((), ())


def distribution(probabilities: list[float]) -> Distribution:
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    return Distribution(
        tokens=tuple(token for token, _ in ranked),
        probabilities=tuple(probability for _, probability in ranked),
    )


def table_evaluator(table: Table) -> Callable[[Tree, int, int], Distribution]:
    def evaluate(tree: Tree, parent: int, rank: int) -> Distribution:
        token = tree[parent].distribution.tokens[rank]
        return table.get((*tree.path(parent), token), EMPTY)

    return evaluate


def full_frontier_order(table: Table) -> list[tuple[tuple[int, ...], float]]:
    """Reference Dijkstra that eagerly enqueues every child."""
    heap: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
    out: list[tuple[tuple[int, ...], float]] = []

    while heap:
        path_nats, path = heapq.heappop(heap)
        if path:
            out.append((path, path_nats))
        dist = table.get(path)
        if dist is None:
            continue
        for rank, token in enumerate(dist.tokens):
            heapq.heappush(heap, (path_nats + dist.nats(rank), (*path, token)))

    return out


class TreeTests(unittest.TestCase):
    def test_tree_derives_token_path_and_edge_cost(self) -> None:
        root_distribution = distribution([0.7, 0.2, 0.1])
        tree = Tree(root_distribution)
        child = tree.put_child(tree.root, 1, EMPTY)

        self.assertEqual(tree.token(child), 1)
        self.assertEqual(tree.path(child), (1,))
        self.assertAlmostEqual(tree.edge_nats(child), -math.log(0.2))

    def test_child_publication_is_idempotent(self) -> None:
        tree = Tree(distribution([0.6, 0.4]))
        child_distribution = distribution([0.8, 0.2])

        first = tree.put_child(0, 0, child_distribution)
        second = tree.put_child(0, 0, child_distribution)

        self.assertEqual(first, second)
        self.assertEqual(tree.child(0, 0), first)
        self.assertEqual(len(tree.nodes), 2)

    def test_conflicting_child_publication_explodes(self) -> None:
        tree = Tree(distribution([1.0]))
        tree.put_child(0, 0, distribution([0.8, 0.2]))

        with self.assertRaisesRegex(ValueError, "conflicting distribution"):
            tree.put_child(0, 0, distribution([0.7, 0.3]))


class SearchTests(unittest.TestCase):
    def test_complete_root_seeds_frontier(self) -> None:
        table = {(): distribution([0.7, 0.3])}
        tree = Tree(table[()])

        search = Search(tree, table_evaluator(table))

        node = search.step()
        assert node is not None
        self.assertEqual(tree.path(node), (0,))

    def test_reset_reuses_known_distribution_at_new_root(self) -> None:
        table = {
            (): distribution([0.7, 0.3]),
            (0,): distribution([0.6, 0.4]),
        }
        tree = Tree(table[()])
        search = Search(tree, table_evaluator(table))
        child = search.step()
        assert child is not None

        search.reset(child)

        node = search.step()
        assert node is not None
        self.assertEqual(tree.path(node), (0, 0))

    def test_zero_probability_tail_never_enters_frontier(self) -> None:
        table = {(): distribution([0.75, 0.25, 0.0])}
        tree = Tree(table[()])
        search = Search(tree, table_evaluator(table))

        paths = []
        while (node := search.step()) is not None:
            paths.append(tree.path(node))

        self.assertEqual(paths, [(0,), (1,)])

    def test_known_example_expands_lowest_path_cost_first(self) -> None:
        table = {
            (): distribution([0.8, 0.2]),
            (0,): distribution([0.9, 0.1]),
            (1,): distribution([0.5, 0.5]),
            (0, 0): distribution([0.6, 0.4]),
        }
        tree = Tree(table[()])
        search = Search(tree, table_evaluator(table))

        paths = []
        while (node := search.step()) is not None:
            paths.append(tree.path(node))

        expected = [path for path, _ in full_frontier_order(table)]
        self.assertEqual(paths, expected)

    def test_prediscovery_does_not_change_search_order(self) -> None:
        table = {
            (): distribution([0.6, 0.3, 0.1]),
            (0,): distribution([0.8, 0.2]),
            (1,): distribution([0.7, 0.3]),
        }
        expected = [path for path, _ in full_frontier_order(table)]

        tree = Tree(table[()])
        search = Search(tree, table_evaluator(table))
        tree.put_child(tree.root, 2, table.get((2,), EMPTY))

        actual = []
        while (node := search.step()) is not None:
            actual.append(tree.path(node))

        self.assertEqual(actual, expected)

    def test_random_trees_match_full_frontier_dijkstra(self) -> None:
        rng = random.Random(0)

        for case in range(250):
            max_depth = rng.randint(1, 5)
            table: Table = {}

            def grow(path: tuple[int, ...], depth: int) -> None:
                if depth >= max_depth or (path and rng.random() < 0.25):
                    return
                width = rng.randint(1, 5)
                weights = [0.05 + rng.random() for _ in range(width)]
                total = sum(weights)
                dist = distribution([weight / total for weight in weights])
                table[path] = dist
                for token in dist.tokens:
                    grow((*path, token), depth + 1)

            grow((), 0)

            reference = full_frontier_order(table)
            tree = Tree(table[()])
            search = Search(tree, table_evaluator(table))
            actual = []
            while (node := search.step()) is not None:
                actual.append((tree.path(node), tree.path_nats(node)))

            self.assertEqual(
                [path for path, _ in actual],
                [path for path, _ in reference],
                msg=f"case {case}",
            )
            self.assertEqual(len(actual), len(reference))
            for (_, actual_cost), (_, reference_cost) in zip(actual, reference, strict=True):
                self.assertAlmostEqual(actual_cost, reference_cost, places=12, msg=f"case {case}")


if __name__ == "__main__":
    unittest.main()
