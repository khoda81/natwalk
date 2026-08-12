from __future__ import annotations

import math
import unittest

from natwalk.tree import Distribution, Tree
from natwalk.view import View, partition_rows, row_tokens


class PartitionNumericsTests(unittest.TestCase):
    def test_tiny_tail_survives_refining_rounded_parent_mass(self) -> None:
        probabilities = (0.9, 0.1, 1e-16)
        tree = Tree(
            Distribution(
                tokens=(0, 1, 2),
                probabilities=probabilities,
            )
        )

        visible = partition_rows(tree, View(), row_limit=3)

        self.assertEqual(len(visible), 3)
        self.assertTrue(all(not row.forest for row in visible))
        self.assertEqual([row_tokens(tree, row) for row in visible], [(0,), (1,), (2,)])
        for row, probability in zip(visible, probabilities, strict=True):
            self.assertAlmostEqual(row.path_nats, -math.log(probability))


if __name__ == "__main__":
    unittest.main()
