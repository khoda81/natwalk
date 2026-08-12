from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from natwalk.search import Search
from natwalk.tree import Distribution, Tree
from natwalk.view import View, enter, forest_nats, move, parent, partition_rows, row_tokens, rows


def distribution(*probabilities: float) -> Distribution:
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    return Distribution(
        tokens=tuple(token for token, _ in ranked),
        probabilities=tuple(probability for _, probability in ranked),
    )


class ViewTests(unittest.TestCase):
    def test_rank_tail_is_view_state_not_tree_state(self) -> None:
        tree = Tree(distribution(0.5, 0.3, 0.15, 0.05))
        before = len(tree.nodes)

        visible = rows(tree, View(node=0, first_rank=2, selected_rank=2), limit=20)

        self.assertEqual([row.rank for row in visible], [2, 3])
        self.assertEqual([row.token for row in visible], [2, 3])
        self.assertEqual(len(tree.nodes), before)

    def test_renderer_walks_discovered_descendants_but_keeps_virtual_siblings(self) -> None:
        tree = Tree(distribution(0.6, 0.3, 0.1))
        first = tree.put_child(0, 0, distribution(0.8, 0.2))

        visible = rows(tree, View(), limit=20)

        self.assertEqual(
            [(row.depth, row.parent, row.rank) for row in visible],
            [(0, 0, 0), (1, first, 0), (1, first, 1), (0, 0, 1), (0, 0, 2)],
        )
        self.assertIsNone(visible[-1].child)

    def test_frame_limit_does_not_materialize_offscreen_nodes(self) -> None:
        tree = Tree(distribution(*([0.01] * 100)))

        visible = rows(tree, View(), limit=7)

        self.assertEqual(len(visible), 7)
        self.assertEqual(len(tree.nodes), 1)

    def test_forest_surprisal_uses_total_probability_mass(self) -> None:
        dist = distribution(0.6, 0.3, 0.1)

        self.assertAlmostEqual(forest_nats(dist, 1), -math.log(0.4))
        self.assertAlmostEqual(
            forest_nats(dist, 1, parent_nats=2.0),
            2.0 - math.log(0.4),
        )

    def test_partition_one_row_is_the_whole_visible_event(self) -> None:
        tree = Tree(distribution(0.6, 0.3, 0.1))

        visible = partition_rows(tree, View(), row_limit=1)

        self.assertEqual(len(visible), 1)
        self.assertTrue(visible[0].forest)
        self.assertEqual(visible[0].forest_count, 3)
        self.assertEqual(row_tokens(tree, visible[0]), ())
        self.assertAlmostEqual(visible[0].path_nats, 0.0)

    def test_partition_refinement_conserves_probability_mass(self) -> None:
        tree = Tree(distribution(0.6, 0.3, 0.1))
        first = tree.put_child(0, 0, distribution(0.8, 0.2))
        tree.put_child(first, 0, Distribution((), ()))

        for row_limit in range(1, 5):
            with self.subTest(row_limit=row_limit):
                visible = partition_rows(tree, View(), row_limit=row_limit)
                mass = math.fsum(math.exp(-row.path_nats) for row in visible)
                self.assertAlmostEqual(mass, 1.0)
                self.assertLessEqual(len(visible), row_limit)

    def test_partition_refinement_does_not_rescan_large_sibling_forests(self) -> None:
        size = 20_000
        tree = Tree(
            Distribution(
                tokens=tuple(range(size)),
                probabilities=(1.0 / size,) * size,
            )
        )

        with patch("natwalk.view.forest_nats", wraps=forest_nats) as aggregate:
            visible = partition_rows(tree, View(), row_limit=64)

        self.assertEqual(len(visible), 64)
        self.assertEqual(aggregate.call_count, 0)

    def test_partition_buys_probable_siblings_before_tiny_deep_deviation(self) -> None:
        tree = Tree(distribution(0.6, 0.25, 0.15))
        first = tree.put_child(0, 0, distribution(0.99, 0.01))

        three = partition_rows(tree, View(), row_limit=3)
        four = partition_rows(tree, View(), row_limit=4)

        def paths(visible):
            return [
                (*tree.path_from(tree.root, row.parent), *row_tokens(tree, row)) for row in visible
            ]

        self.assertEqual(paths(three), [(0,), (1,), (2,)])
        self.assertEqual(paths(four), [(0, 0), (0, 1), (1,), (2,)])
        self.assertAlmostEqual(
            max(row.path_nats for row in three),
            -math.log(0.15),
        )
        self.assertAlmostEqual(
            max(row.path_nats for row in four),
            -math.log(0.6 * 0.01),
        )
        self.assertIsNotNone(first)

    def test_partition_layout_factors_shared_prefixes_without_extra_rows(self) -> None:
        tree = Tree(distribution(0.6, 0.4))
        first = tree.put_child(0, 0, distribution(0.8, 0.2))
        second = tree.put_child(first, 0, distribution(0.7, 0.3))

        visible = partition_rows(tree, View(), row_limit=4)

        self.assertEqual(len(visible), 4)
        self.assertEqual(
            [(*tree.path_from(tree.root, row.parent), *row_tokens(tree, row)) for row in visible],
            [(0, 0, 0), (0, 0, 1), (0, 1), (1,)],
        )
        self.assertEqual([row_tokens(tree, row) for row in visible], [(0, 0, 0), (1,), (1,), (1,)])
        self.assertEqual([row.ranks for row in visible], [(0, 0, 0), (1,), (1,), (1,)])
        self.assertEqual([row.depth for row in visible], [0, 2, 1, 0])
        self.assertEqual(visible[1].parent, second)
        self.assertEqual(len(visible[1].ancestor_last), 2)
        self.assertEqual(len(visible[1].ancestor_nats), 2)
        self.assertAlmostEqual(visible[1].ancestor_nats[0], -math.log(0.6))
        self.assertAlmostEqual(visible[1].ancestor_nats[1], -math.log(0.6 * 0.8))

    def test_enter_requires_discovered_child(self) -> None:
        tree = Tree(distribution(0.5, 0.3, 0.2))

        with self.assertRaisesRegex(ValueError, "undiscovered child"):
            enter(tree, View(node=0, selected_rank=2))

        child = tree.put_child(0, 2, Distribution((), ()))
        self.assertEqual(enter(tree, View(node=0, selected_rank=2)).node, child)

    def test_parent_preserves_entered_rank_as_forest_boundary(self) -> None:
        tree = Tree(distribution(0.5, 0.3, 0.2))
        child = tree.put_child(0, 2, Distribution((), ()))

        self.assertEqual(
            parent(tree, View(node=child)),
            View(node=0, first_rank=2, selected_rank=2),
        )

    def test_move_never_crosses_the_tail_boundary(self) -> None:
        tree = Tree(distribution(0.5, 0.3, 0.2))
        view = View(node=0, first_rank=1, selected_rank=1)

        self.assertEqual(move(tree, view, -1).selected_rank, 1)
        self.assertEqual(move(tree, view, 10).selected_rank, 2)

    def test_browsing_does_not_change_dijkstra_frontier(self) -> None:
        table = {
            (): distribution(0.6, 0.3, 0.1),
            (2,): distribution(0.7, 0.3),
        }

        def evaluate(tree: Tree, parent_id: int, rank: int) -> Distribution:
            token = tree[parent_id].distribution.tokens[rank]
            return table.get((*tree.path(parent_id), token), Distribution((), ()))

        tree = Tree(table[()])
        search = Search(tree, evaluate)
        before = tuple(search.frontier)
        child = tree.put_child(0, 2, table[(2,)])

        rows(tree, View(node=child), limit=10)
        parent(tree, View(node=child))

        self.assertEqual(tuple(search.frontier), before)


if __name__ == "__main__":
    unittest.main()
