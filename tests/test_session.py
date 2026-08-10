from __future__ import annotations

import unittest

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


class SessionTests(unittest.TestCase):
    def make_table(self):
        return {
            (): (0.7, 0.3),
            (0,): (0.2, 0.8),
            (1,): (0.6, 0.4),
            (0, 1): (1.0,),
        }

    def test_search_evaluation_restores_committed_cursor(self) -> None:
        cursor = TableCursor(self.make_table())
        session = Session(cursor)

        session.search.step()
        session.search.step()

        self.assertEqual(cursor.prefix, ())
        self.assertEqual(session.root, session.tree.root)

    def test_commit_is_the_only_committed_path_mutation(self) -> None:
        cursor = TableCursor(self.make_table())
        session = Session(cursor)

        root = session.commit((0, 1))

        self.assertEqual(cursor.prefix, (0, 1))
        self.assertEqual(session.tree.path(root), (0, 1))

    def test_restore_changes_execution_root_but_keeps_discovered_tree(self) -> None:
        cursor = TableCursor(self.make_table())
        session = Session(cursor)
        checkpoint = session.checkpoint()
        root = session.commit((0, 1))
        known_nodes = len(session.tree.nodes)
        self.assertIsNotNone(session.tree[root].distribution)

        session.restore(checkpoint)

        self.assertEqual(cursor.prefix, ())
        self.assertEqual(session.root, session.tree.root)
        self.assertEqual(len(session.tree.nodes), known_nodes)
        self.assertIsNotNone(session.tree[root].distribution)

    def test_empty_commit_is_noop(self) -> None:
        cursor = TableCursor(self.make_table())
        session = Session(cursor)
        frontier = tuple(session.search.frontier)

        self.assertEqual(session.commit(()), session.root)
        self.assertEqual(cursor.prefix, ())
        self.assertEqual(tuple(session.search.frontier), frontier)

    def test_terminal_is_an_empty_distribution_not_a_cursor_flag(self) -> None:
        cursor = TableCursor({(): ()})
        session = Session(cursor)

        self.assertEqual(session.tree[session.root].distribution.tokens, ())
        self.assertIsNone(session.search.step())


if __name__ == "__main__":
    unittest.main()
