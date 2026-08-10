from __future__ import annotations

import math
import unittest

from natwalk import Navigator, TokenTreeExplorer


class TableCursor:
    """Tiny exact backend for testing arithmetic and tree invariants."""

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


class RewindCursor:
    """Infinite backend whose checkpoint is cheap and clone must never be used."""

    def __init__(self, prefix=()):
        self.prefix = tuple(prefix)
        self.ended = False
        self.clones = 0

    def clone(self):
        self.clones += 1
        raise AssertionError("rewindable preview should not clone")

    def predict(self):
        return [0.55, 0.30, 0.15]

    def observe(self, token):
        self.prefix = (*self.prefix, token)

    def checkpoint(self):
        return self.prefix

    def restore(self, checkpoint):
        self.prefix = checkpoint
        self.ended = False


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
        table = {(): [0.9, 0.1], (0,): [0.2] * 5}
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

    def test_rewindable_preview_does_not_clone_or_mutate(self):
        cursor = RewindCursor()
        nav = Navigator(cursor, choices=5, preview_tokens=4)
        before = (cursor.prefix, nav.state.lo, nav.state.hi, nav.state.actions)
        preview = nav.preview(0)
        after = (cursor.prefix, nav.state.lo, nav.state.hi, nav.state.actions)
        self.assertTrue(preview.forced or preview.representative)
        self.assertEqual(before, after)
        self.assertEqual(cursor.clones, 0)

    def test_greedy_suggestion_is_budgeted_by_surprisal(self):
        table = {
            (): [0.8, 0.2],
            (0,): [0.9, 0.1],
            (0, 0): [0.6, 0.4],
            (0, 0, 0): [0.5, 0.5],
        }
        nav = Navigator(TableCursor(table), choices=2)
        budget = (-math.log2(0.8) - math.log2(0.9)) + 1e-9
        suggestion = nav.greedy_suggestion(max_bits=budget)
        self.assertEqual(suggestion.tokens, (0, 0))
        self.assertAlmostEqual(suggestion.bits, -math.log2(0.8 * 0.9), places=9)
        self.assertAlmostEqual(suggestion.next_token_bits, -math.log2(0.6), places=9)

    def test_accept_greedy_is_one_undoable_action(self):
        table = {
            (): [0.8, 0.2],
            (0,): [0.9, 0.1],
            (0, 0): [0.6, 0.4],
        }
        nav = Navigator(TableCursor(table), choices=2)
        suggestion = nav.accept_greedy(max_bits=1.0)
        self.assertEqual(suggestion.tokens, (0, 0))
        self.assertEqual(nav.state.cursor.prefix, (0, 0))
        self.assertEqual(nav.undo_depth, 1)
        self.assertTrue(nav.undo())
        self.assertEqual(nav.state.cursor.prefix, ())
        self.assertEqual(nav.state.lo, 0.0)
        self.assertEqual(nav.state.hi, 1.0)

    def test_choose_is_undoable(self):
        nav = Navigator(RewindCursor(), choices=2)
        nav.choose(1)
        self.assertEqual(nav.state.actions, 1)
        self.assertTrue(nav.undo())
        self.assertEqual(nav.state.actions, 0)
        self.assertEqual(nav.state.cursor.prefix, ())


class TokenTreeExplorerTests(unittest.TestCase):
    def make_table(self):
        return {
            (): [0.8, 0.2],
            (0,): [0.9, 0.1],
            (1,): [0.5, 0.5],
            (0, 0): [0.9, 0.1],
            (0, 1): [0.5, 0.5],
            (1, 0): [0.5, 0.5],
            (1, 1): [0.5, 0.5],
            (0, 0, 0): [0.5, 0.5],
            (0, 0, 1): [0.5, 0.5],
        }

    def test_dijkstra_expands_lowest_surprisal_prefix_first(self):
        nav = Navigator(TableCursor(self.make_table()), choices=2)
        explorer = TokenTreeExplorer(nav, max_nodes=64, autostart=False)

        self.assertTrue(explorer.step())
        self.assertTrue(explorer.step())
        self.assertTrue(explorer.step())

        entries = explorer.tree_entries()
        node0 = next(e for e in entries if e.path == (0,))
        self.assertTrue(node0.expanded)
        self.assertFalse(any(e.path == (1,) and e.expanded for e in entries))

    def test_ellipsis_cost_is_residual_probability(self):
        nav = Navigator(TableCursor(self.make_table()), choices=2)
        explorer = TokenTreeExplorer(nav, max_nodes=64, autostart=False)
        explorer.step()
        explorer.step()

        root_ellipsis = next(e for e in explorer.tree_entries() if e.is_ellipsis and e.depth == 0)
        self.assertEqual(root_ellipsis.hidden_count, 1)
        self.assertAlmostEqual(root_ellipsis.hidden_nats, -math.log(0.2), places=12)

    def test_no_depth_limit_only_node_cap(self):
        nav = Navigator(RewindCursor(), choices=2)
        explorer = TokenTreeExplorer(nav, max_nodes=12, autostart=False)
        for _ in range(200):
            if not explorer.step():
                break
        stats = explorer.stats()
        self.assertLessEqual(stats.nodes, 12)
        self.assertTrue(stats.saturated or stats.nodes == 12)

    def test_binary_narrowing_retargets_active_tree_without_requiring_token_commit(self):
        table = {
            (): [0.4, 0.35, 0.25],
            (1,): [0.5, 0.5, 0.0],
            (2,): [0.5, 0.5, 0.0],
        }
        nav = Navigator(TableCursor(table), choices=2)
        explorer = TokenTreeExplorer(nav, max_nodes=64, autostart=False)
        explorer.step()
        explorer.step()
        explorer.choose(1)
        self.assertEqual(explorer.snapshot.actions, 1)
        self.assertEqual(explorer.snapshot.prefix, ())
        self.assertAlmostEqual(explorer.snapshot.lo, 0.5)
        self.assertAlmostEqual(explorer.snapshot.hi, 1.0)

    def test_explorer_accept_and_undo_reroot(self):
        nav = Navigator(RewindCursor(), choices=2)
        explorer = TokenTreeExplorer(nav, max_nodes=32, autostart=False)
        suggestion = explorer.accept_greedy(max_bits=2.0, max_tokens=3)
        self.assertTrue(suggestion.tokens)
        self.assertEqual(explorer.snapshot.prefix, suggestion.tokens)
        self.assertTrue(explorer.undo())
        self.assertEqual(explorer.snapshot.prefix, ())


if __name__ == "__main__":
    unittest.main()
