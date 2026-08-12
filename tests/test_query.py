from __future__ import annotations

import math
import unittest

from natwalk.query import (
    Suggestion,
    completions,
    cycle_suggestion,
    greedy,
    normalize_suggestion,
    suggestion_edges,
    suggestion_tokens,
)
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

        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion_tokens(tree, 0, suggestion), (0, 0))
        self.assertEqual(
            suggestion_edges(tree, 0, suggestion),
            (Suggestion(0, 0), Suggestion(1, 0)),
        )

    def test_greedy_keeps_virtual_edge_as_structural_endpoint(self) -> None:
        tree = Tree(distribution(0.8, 0.2))

        suggestion = greedy(tree, 0, max_nats=10.0)

        self.assertEqual(suggestion, Suggestion(0, 0))
        self.assertEqual(suggestion_tokens(tree, 0, suggestion), (0,))
        self.assertEqual(len(tree.nodes), 1)

    def test_completions_are_probability_ranked_dfs_without_special_greedy_logic(self) -> None:
        tree = self.make_tree()

        suggestions = completions(tree, 0, max_nats=3.0)
        tokens = [suggestion_tokens(tree, 0, suggestion) for suggestion in suggestions]

        self.assertEqual(tokens, [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(
            tokens[0],
            suggestion_tokens(tree, 0, greedy(tree, 0, max_nats=3.0)),
        )

    def test_queries_do_not_materialize_virtual_children(self) -> None:
        tree = Tree(distribution(0.5, 0.3, 0.2))
        before = len(tree.nodes)

        suggestions = completions(tree, 0, max_nats=10.0)

        self.assertEqual(
            [suggestion_tokens(tree, 0, suggestion) for suggestion in suggestions],
            [(0,), (1,), (2,)],
        )
        self.assertEqual(len(tree.nodes), before)

    def test_tree_growth_extends_selected_branch_without_changing_its_identity_prefix(self) -> None:
        tree = Tree(distribution(0.8, 0.2))
        selected = Suggestion(0, 0)

        child = tree.put_child(0, 0, distribution(0.9, 0.1))
        tree.put_child(child, 0, Distribution((), ()))

        updated = normalize_suggestion(tree, 0, selected, max_nats=10.0)

        self.assertEqual(updated, Suggestion(child, 0))
        self.assertIn(selected, suggestion_edges(tree, 0, updated))
        self.assertEqual(suggestion_tokens(tree, 0, updated), (0, 0))

    def test_budget_shrink_retreats_to_deepest_valid_endpoint_on_same_branch(self) -> None:
        tree = self.make_tree()
        selected = Suggestion(1, 0)

        updated = normalize_suggestion(
            tree,
            0,
            selected,
            max_nats=-math.log(0.6) + 1e-12,
        )

        self.assertEqual(updated, Suggestion(0, 0))

    def test_tab_cycles_structural_completion_endpoints(self) -> None:
        tree = self.make_tree()
        first = normalize_suggestion(tree, 0, None, max_nats=3.0)

        second = cycle_suggestion(tree, 0, first, 1, max_nats=3.0)
        previous = cycle_suggestion(tree, 0, second, -1, max_nats=3.0)

        self.assertEqual(suggestion_tokens(tree, 0, first), (0, 0))
        self.assertEqual(suggestion_tokens(tree, 0, second), (0, 1))
        self.assertEqual(previous, first)


if __name__ == "__main__":
    unittest.main()
