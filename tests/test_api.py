from __future__ import annotations

import unittest

import natwalk


class PublicApiTests(unittest.TestCase):
    def test_public_api_is_the_rewritten_state_model(self) -> None:
        expected = {
            "Cursor",
            "Distribution",
            "Navigation",
            "NavigationState",
            "NodeUpdate",
            "Row",
            "Search",
            "SearchWorker",
            "Session",
            "Suggestion",
            "Tree",
            "TreeReplica",
            "View",
            "completions",
            "greedy",
            "rows",
            "updates",
        }
        self.assertEqual(set(natwalk.__all__), expected)
        for name in expected:
            self.assertTrue(hasattr(natwalk, name), name)

    def test_removed_explorer_api_is_not_kept_as_compatibility_state(self) -> None:
        for name in (
            "Navigator",
            "TokenTreeExplorer",
            "TreeEntry",
            "RankedDistribution",
            "RewindableCursor",
            "accept_completion",
            "cached_budget_completions",
        ):
            self.assertFalse(hasattr(natwalk, name), name)


if __name__ == "__main__":
    unittest.main()
