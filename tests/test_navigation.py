from __future__ import annotations

import unittest

from natwalk.navigation import Navigation, State
from natwalk.session import Session


class TableCursor:
    def __init__(self, table, prefix=()):
        self.table = table
        self.prefix = tuple(prefix)

    def predict(self):
        return self.table.get(self.prefix, ())

    def observe(self, token):
        self.prefix = (*self.prefix, token)

    def checkpoint(self):
        return self.prefix

    def restore(self, checkpoint):
        self.prefix = checkpoint


class NavigationTests(unittest.TestCase):
    def test_unforced_choice_changes_only_arithmetic_state(self) -> None:
        cursor = TableCursor({(): (0.4, 0.35, 0.25)})
        session = Session(cursor)
        navigation = Navigation(session)
        root = session.root
        frontier = tuple(session.search.frontier)
        nodes = len(session.tree.nodes)

        forced = navigation.choose(1)

        self.assertEqual(forced, ())
        self.assertEqual(session.root, root)
        self.assertEqual(cursor.prefix, ())
        self.assertEqual(tuple(session.search.frontier), frontier)
        self.assertEqual(len(session.tree.nodes), nodes)
        self.assertEqual(navigation.state, State(lo=0.5, hi=1.0, actions=1))

    def test_choice_commits_all_tokens_forced_by_same_action(self) -> None:
        cursor = TableCursor(
            {
                (): (0.9, 0.1),
                (0,): (0.95, 0.05),
                (0, 0): (0.2, 0.2, 0.2, 0.2, 0.2),
            }
        )
        session = Session(cursor)
        navigation = Navigation(session, choices=5)

        forced = navigation.choose(0)

        self.assertEqual(forced, (0, 0))
        self.assertEqual(cursor.prefix, (0, 0))
        self.assertEqual(session.tree.path(session.root), (0, 0))
        self.assertEqual(navigation.undo_depth, 1)
        self.assertEqual(navigation.state.actions, 1)

    def test_undo_restores_all_tokens_forced_by_one_choice(self) -> None:
        cursor = TableCursor(
            {
                (): (0.9, 0.1),
                (0,): (0.95, 0.05),
                (0, 0): (0.2, 0.2, 0.2, 0.2, 0.2),
            }
        )
        session = Session(cursor)
        navigation = Navigation(session, choices=5)
        before_frontier = tuple(session.search.frontier)

        navigation.choose(0)
        self.assertTrue(navigation.undo())

        self.assertEqual(cursor.prefix, ())
        self.assertEqual(session.root, session.tree.root)
        self.assertEqual(navigation.state, State())
        self.assertEqual(tuple(session.search.frontier), before_frontier)

    def test_explicit_accept_is_one_action_and_resets_pending_interval(self) -> None:
        cursor = TableCursor({(): (0.4, 0.35, 0.25), (0,): (0.7, 0.3)})
        session = Session(cursor)
        navigation = Navigation(session)
        navigation.choose(1)
        self.assertNotEqual(navigation.state, State())

        navigation.accept((0,))

        self.assertEqual(cursor.prefix, (0,))
        self.assertEqual(navigation.state.lo, 0.0)
        self.assertEqual(navigation.state.hi, 1.0)
        self.assertEqual(navigation.undo_depth, 2)
        self.assertTrue(navigation.undo())
        self.assertEqual(cursor.prefix, ())
        self.assertEqual(navigation.state, State(lo=0.5, hi=1.0, actions=1))


if __name__ == "__main__":
    unittest.main()
