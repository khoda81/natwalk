from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from natwalk.tui import _write_frame


class _Stdout(io.StringIO):
    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


class FrameOutputTests(unittest.TestCase):
    def test_interactive_frame_does_not_advance_past_last_row(self) -> None:
        stdout = _Stdout(tty=True)
        frame = ["row 1", "row 2", "row 3"]

        with patch("natwalk.tui.sys.stdout", stdout):
            _write_frame(frame)

        prefix = "\033[2J\033[H"
        self.assertEqual(stdout.getvalue(), prefix + "row 1\nrow 2\nrow 3")
        self.assertEqual(stdout.getvalue()[len(prefix) :].count("\n"), len(frame) - 1)
        self.assertFalse(stdout.getvalue().endswith("\n"))

    def test_non_interactive_frame_remains_line_terminated(self) -> None:
        stdout = _Stdout(tty=False)

        with patch("natwalk.tui.sys.stdout", stdout):
            _write_frame(["row 1", "row 2"])

        self.assertEqual(stdout.getvalue(), "row 1\nrow 2\n")


if __name__ == "__main__":
    unittest.main()
