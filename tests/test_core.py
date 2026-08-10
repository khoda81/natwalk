from __future__ import annotations

import math
import unittest

from natwalk import Navigator


class TableCursor:
    """Tiny exact backend for testing the arithmetic navigation invariants."""

    def __init__(self, table, prefix=(), eos=99):
        self.table = table
        self.prefix = tuple(prefix)
        self.eos = eos
        self.ended = bool(self.prefix and self.prefix[-1] == eos)

    def clone(self):
        return TableCursor(self.table, self.prefix, self.eos)

    def predict(self):
        return self.table[self.prefix]

    def observe(self, token):
        self.prefix = (*self.prefix, token)
        self.ended = token == self.eos


class NavigatorTests(unittest.TestCase):
    def test_each_action_has_exact_information_cost(self):
        table = {
            (): [0.1] * 10,
            (0,): [0.2] * 5,
            (1,): [0.2] * 5,
            (2,): [0.2] * 5,
            (3,): [0.2] * 5,
            (4,): [0.2] * 5,
            (5,): [0.2] * 5,
            (6,): [0.2] * 5,
            (7,): [0.2] * 5,
            (8,): [0.2] * 5,
            (9,): [0.2] * 5,
        }
        nav = Navigator(TableCursor(table), choices=5)
        nav.choose(4)
        nav.choose(0)
        self.assertEqual(nav.state.actions, 2)
        self.assertAlmostEqual(nav.supplied_nats, 2 * math.log(5), places=12)
        self.assertAlmostEqual(nav.supplied_bits, 2 * math.log2(5), places=12)

    def test_token_commits_only_when_whole_interval_agrees(self):
        table = {
            (): [0.9, 0.1],
            (0,): [0.2] * 5,
        }
        nav = Navigator(TableCursor(table), choices=5)
        forced = nav.choose(0)
        self.assertEqual(forced, (0,))
        self.assertEqual(nav.state.cursor.prefix, (0,))
        self.assertAlmostEqual(nav.state.path_surprisal, -math.log(0.9), places=12)
        self.assertAlmostEqual(nav.state.lo, 0.0, places=12)
        self.assertAlmostEqual(nav.state.hi, 2 / 9, places=12)

    def test_residual_bucket_is_not_truncated(self):
        table = {(): [0.1] * 10}
        nav = Navigator(TableCursor(table), choices=5)
        forced = nav.choose(4)
        self.assertEqual(forced, ())
        self.assertEqual(nav.state.cursor.prefix, ())
        self.assertAlmostEqual(nav.state.lo, 0.8, places=12)
        self.assertAlmostEqual(nav.state.hi, 1.0, places=12)

    def test_children_are_ordered_by_probability(self):
        table = {(): [0.05, 0.70, 0.20, 0.05]}
        ranked = Navigator.rank(TableCursor(table))
        self.assertEqual(ranked.tokens, (1, 2, 0, 3))
        self.assertAlmostEqual(ranked.probabilities[0], 0.70)
        self.assertAlmostEqual(ranked.edges[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
