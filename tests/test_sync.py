from __future__ import annotations

import unittest

from natwalk.sync import NodeUpdate, TreeReplica, updates
from natwalk.tree import Distribution, Tree


def distribution(*probabilities: float) -> Distribution:
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    return Distribution(
        tokens=tuple(token for token, _ in ranked),
        probabilities=tuple(probability for _, probability in ranked),
    )


class TreeSyncTests(unittest.TestCase):
    def make_tree(self) -> Tree:
        tree = Tree(distribution(0.6, 0.4))
        zero = tree.put_child(0, 0, distribution(0.8, 0.2))
        tree.put_child(0, 1, Distribution((), ()))
        tree.put_child(zero, 0, Distribution((), ()))
        return tree

    def test_replica_reconstructs_tree_from_append_log(self) -> None:
        source = self.make_tree()
        replica = TreeReplica()

        replica.apply_many(updates(source))

        assert replica.tree is not None
        self.assertEqual(replica.next_node, len(source.nodes))
        self.assertEqual(
            [replica.tree.path(node) for node in range(len(replica.tree.nodes))],
            [source.path(node) for node in range(len(source.nodes))],
        )
        self.assertEqual(
            [node.distribution for node in replica.tree.nodes],
            [node.distribution for node in source.nodes],
        )

    def test_replaying_updates_is_a_verified_noop(self) -> None:
        source = self.make_tree()
        replica = TreeReplica()
        batch = updates(source)
        replica.apply_many(batch)
        before = replica.next_node

        replica.apply_many(batch)

        self.assertEqual(replica.next_node, before)

    def test_suffix_resumes_from_client_node_count(self) -> None:
        source = self.make_tree()
        replica = TreeReplica()
        replica.apply_many(updates(source, start=0)[:2])

        replica.apply_many(updates(source, start=replica.next_node))

        self.assertEqual(replica.next_node, len(source.nodes))

    def test_missing_update_explodes(self) -> None:
        source = self.make_tree()
        replica = TreeReplica()
        batch = updates(source)
        replica.apply(batch[0])

        with self.assertRaisesRegex(ValueError, "missing tree updates"):
            replica.apply(batch[2])

    def test_conflicting_duplicate_explodes(self) -> None:
        source = self.make_tree()
        replica = TreeReplica()
        replica.apply_many(updates(source))
        node = source[1]

        conflicting = NodeUpdate(
            node=1,
            parent=node.parent,
            rank=node.rank,
            distribution=distribution(0.7, 0.3),
        )
        with self.assertRaisesRegex(ValueError, "conflicting contents"):
            replica.apply(conflicting)


if __name__ == "__main__":
    unittest.main()
