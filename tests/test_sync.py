from __future__ import annotations

import unittest

from natwalk.sync import NodeUpdate, TreeReplica, reveal, updates
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
        for source_node, replica_node in zip(
            source.nodes,
            replica.tree.nodes,
            strict=True,
        ):
            source_distribution = source_node.distribution
            replica_distribution = replica_node.distribution
            self.assertEqual(len(replica_distribution), len(source_distribution))
            self.assertEqual(replica_distribution.revealed, len(source_distribution))
            self.assertEqual(
                tuple(
                    replica_distribution.token(rank)
                    for rank in range(replica_distribution.revealed)
                ),
                tuple(source_distribution.token(rank) for rank in range(len(source_distribution))),
            )
            self.assertEqual(
                tuple(
                    replica_distribution.probability(rank)
                    for rank in range(replica_distribution.revealed)
                ),
                tuple(
                    source_distribution.probability(rank)
                    for rank in range(len(source_distribution))
                ),
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
            size=2,
            tokens=(0, 1),
            probabilities=(0.7, 0.3),
            tail_probability=0.0,
        )
        with self.assertRaisesRegex(ValueError, "conflicting contents"):
            replica.apply(conflicting)

    def test_initial_update_keeps_only_ranked_prefix_and_exact_tail_mass(self) -> None:
        source = Tree(distribution(0.4, 0.3, 0.2, 0.1))
        replica = TreeReplica()

        replica.apply_many(updates(source, initial_reveal=2))

        assert replica.tree is not None
        result = replica.tree[0].distribution
        self.assertEqual(len(result), 4)
        self.assertEqual(result.revealed, 2)
        self.assertEqual(result.token(0), 0)
        self.assertEqual(result.token(1), 1)
        self.assertAlmostEqual(result.mass(2, 4), 0.3)
        self.assertAlmostEqual(result.mass(-100, 100), 1.0)
        self.assertEqual(result.mass(3, 2), 0.0)
        with self.assertRaisesRegex(IndexError, "partial unrevealed"):
            result.mass(3, 4)

    def test_reveal_extends_prefix_without_materializing_the_rest(self) -> None:
        source = Tree(distribution(0.4, 0.3, 0.2, 0.1))
        replica = TreeReplica()
        replica.apply_many(updates(source, initial_reveal=2))

        replica.apply(reveal(source, 0, 2, 3))

        assert replica.tree is not None
        result = replica.tree[0].distribution
        self.assertEqual(result.revealed, 3)
        self.assertEqual(result.token(2), 2)
        self.assertAlmostEqual(result.probability(2), 0.2)
        self.assertAlmostEqual(result.mass(3, 4), 0.1)


if __name__ == "__main__":
    unittest.main()
