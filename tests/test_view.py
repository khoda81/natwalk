from __future__ import annotations

import unittest

from natwalk.search import Search
from natwalk.tree import Distribution, Tree
from natwalk.view import View, enter, move, parent, rows


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
