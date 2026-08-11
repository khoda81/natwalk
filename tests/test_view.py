from __future__ import annotations

import math
import unittest

from natwalk.search import Search
from natwalk.tree import Distribution, Tree
from natwalk.view import View, compact_rows, enter, forest_nats, move, parent, rows


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

    def test_compact_view_collapses_unary_chain_and_keeps_side_forest(self) -> None:
        tree = Tree(distribution(0.6, 0.4))
        first = tree.put_child(0, 0, distribution(0.8, 0.2))
        tree.put_child(first, 0, Distribution((), ()))

        visible = compact_rows(tree, View(), edge_limit=2)

        self.assertEqual(visible[0].tokens, (0, 0))
        self.assertFalse(visible[0].forest)
        self.assertEqual(visible[1].tokens, (0,))
        self.assertTrue(visible[1].forest)
        self.assertEqual(visible[1].forest_count, 1)
        self.assertAlmostEqual(
            visible[1].path_nats,
            -math.log(0.6) - math.log(0.2),
        )

    def test_compact_view_leaves_undiscovered_edge_open_ended(self) -> None:
        tree = Tree(distribution(0.6, 0.4))

        visible = compact_rows(tree, View(), edge_limit=1)

        self.assertEqual(visible[0].tokens, (0,))
        self.assertTrue(visible[0].open_ended)
        self.assertIsNone(visible[0].child)

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
