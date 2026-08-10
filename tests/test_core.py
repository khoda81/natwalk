from __future__ import annotations

import math
import unittest

from natwalk import Navigator, TreeExplorer


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

    def test_preview_path_restores_root(self):
        cursor = RewindCursor()
        nav = Navigator(cursor, choices=5, preview_tokens=2)
        preview = nav.preview_path((4, 0, 1))
        self.assertEqual(preview.bucket, 1)
        self.assertEqual(cursor.prefix, ())
        self.assertEqual(nav.state.actions, 0)


class TreeExplorerTests(unittest.TestCase):
    def test_background_worker_populates_and_survives_rebase(self):
        cursor = RewindCursor()
        nav = Navigator(cursor, choices=3, preview_tokens=2)
        with TreeExplorer(nav, prefetch_depth=2, max_cached=16) as explorer:
            self.assertTrue(explorer.wait_current(timeout=2.0))
            previews = explorer.current_previews()
            self.assertEqual(len(previews), 2)
            self.assertTrue(all(preview is not None for preview in previews))

            explorer.choose(2)
            self.assertEqual(explorer.snapshot.actions, 1)
            self.assertTrue(explorer.wait_current(timeout=2.0))
            self.assertTrue(all(p is not None for p in explorer.current_previews()))
            self.assertAlmostEqual(explorer.supplied_nats, math.log(3), places=12)

    def test_worker_errors_surface(self):
        class BrokenCursor(RewindCursor):
            def predict(self):
                raise ValueError("boom")

        nav = Navigator(BrokenCursor(), choices=3)
        with TreeExplorer(nav, prefetch_depth=1) as explorer:
            with self.assertRaises(RuntimeError):
                explorer.wait_current(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
