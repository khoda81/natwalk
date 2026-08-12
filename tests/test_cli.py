from __future__ import annotations

import argparse
import unittest

from natwalk.cli import HelpFormatter, add_tui_arguments


class CliTests(unittest.TestCase):
    def test_shared_tui_help_shows_concrete_defaults_and_unlimited_memory(self) -> None:
        parser = argparse.ArgumentParser(formatter_class=HelpFormatter)
        add_tui_arguments(
            parser,
            max_tokens_default=123,
            max_tokens_help="test token limit",
        )

        help_text = parser.format_help()
        self.assertIn("test token limit (default: 123)", help_text)
        args = parser.parse_args([])
        self.assertEqual(args.budget_nats, 1.5)
        self.assertEqual(args.budget_step, 0.25)
        self.assertIn("default unlimited", help_text)
        self.assertNotIn("default: None", help_text)


if __name__ == "__main__":
    unittest.main()
