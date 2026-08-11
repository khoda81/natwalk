from __future__ import annotations

import math
import unittest

from natwalk.query import completions, greedy
from natwalk.tree import Distribution, Tree


def distribution(*probabilities: float) -> Distribution:
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    return Distribution(
        tokens=tuple(token for token, _ in ranked),
        probabilities=tuple(probability for _, probability in ranked),
    )


class QueryTests(unittest.TestCase):
    def make_tree(self) -> Tree:
        tree = Tree(distribution(0.6, 0.4))
        zero = tree.put_child(0, 0, distribution(0.55, 0.45))
        one = tree.put_child(0, 1, distribution(0.7, 0.3))
        for parent in (zero, one):
            for rank in (0, 1):
                tree.put_child(parent, rank, Distribution((), ()))
        return tree

    def test_greedy_follows_rank_zero_until_budget_boundary(self) -> None:
        tree = self.make_tree()
        budget = -math.log(0.6 * 0.55) + 1e-12

        suggestion = greedy(tree, 0, max_nats=budget)

        self.assertEqual(suggestion.tokens, (0, 0))
        self.assertAlmostEqual(suggestion.nats, -math.log(0.6 * 0.55))
        self.assertTrue(suggestion.complete)

    def test_greedy_marks_unknown_tail_incomplete(self) -> None:
        tree = Tree(distribution(0.8, 0.2))

        suggestion = greedy(tree, 0, max_nats=10.0)

        self.assertEqual(suggestion.tokens, (0,))
        self.assertFalse(suggestion.complete)
        self.assertEqual(len(tree.nodes), 1)

    def test_completions_are_probability_ranked_dfs_without_special_greedy_logic(self) -> None:
        tree = self.make_tree()

        suggestions = completions(tree, 0, max_nats=3.0)

        self.assertEqual(
            [suggestion.tokens for suggestion in suggestions],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        self.assertEqual(suggestions[0].tokens, greedy(tree, 0, max_nats=3.0).tokens)

    def test_queries_do_not_materialize_virtual_children(self) -> None:
        tree = Tree(distribution(0.5, 0.3, 0.2))
        before = len(tree.nodes)

        suggestions = completions(tree, 0, max_nats=10.0)

        self.assertEqual([s.tokens for s in suggestions], [(0,), (1,), (2,)])
        self.assertEqual(len(tree.nodes), before)


if __name__ == "__main__":
    unittest.main()
