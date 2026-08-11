from __future__ import annotations

import unittest

from natwalk.sync import RankUpdate, TreeReplica, reveal, updates
from natwalk.tree import Distribution, Tree


class SparseReplicaSyncTests(unittest.TestCase):
    def make_tree(self) -> tuple[Tree, int]:
        size = 200
        root = Distribution(
            tokens=tuple(range(size)),
            probabilities=(1.0 / size,) * size,
        )
        tree = Tree(root)
        child = tree.put_child(0, 145, Distribution((), ()))
        return tree, child

    def test_discovered_out_of_prefix_edge_is_pinned_without_revealing_gap(self) -> None:
        source, child = self.make_tree()
        batch = updates(source)
        replica = TreeReplica()

        self.assertTrue(any(isinstance(update, RankUpdate) for update in batch))
        replica.apply_many(batch)

        assert replica.tree is not None
        distribution = replica.tree[0].distribution
        self.assertEqual(distribution.revealed, 128)
        self.assertEqual(distribution.token(145), 145)
        self.assertAlmostEqual(distribution.probability(145), 1.0 / 200)
        with self.assertRaisesRegex(IndexError, "revealed or pinned"):
            distribution.token(144)
        self.assertEqual(replica.tree.path(child), (145,))
        self.assertAlmostEqual(replica.tree.edge_nats(child), -__import__("math").log(1.0 / 200))

    def test_contiguous_reveal_absorbs_existing_sparse_pin(self) -> None:
        source, child = self.make_tree()
        replica = TreeReplica()
        replica.apply_many(updates(source))

        replica.apply(reveal(source, 0, 128, 160))

        assert replica.tree is not None
        distribution = replica.tree[0].distribution
        self.assertEqual(distribution.revealed, 160)
        self.assertEqual(distribution.token(145), 145)
        self.assertEqual(replica.tree.path(child), (145,))


if __name__ == "__main__":
    unittest.main()
