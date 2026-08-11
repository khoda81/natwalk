from __future__ import annotations

import unittest

from natwalk.tui import _cell_width, _clip, _fit, _format_distribution_row


class TerminalWidthTests(unittest.TestCase):
    def test_clip_respects_terminal_cells(self) -> None:
        self.assertEqual(_clip("abcdef", 4), "abc…")
        self.assertLessEqual(_cell_width(_clip("界界界", 4)), 4)
        self.assertLessEqual(_cell_width(_clip("e\u0301e\u0301e\u0301", 2)), 2)

    def test_fit_pads_by_display_width(self) -> None:
        fitted = _fit("界", 4)
        self.assertEqual(_cell_width(fitted), 4)

    def test_distribution_row_never_exceeds_terminal_width(self) -> None:
        labels = (
            "program_119",
            "C♯7",
            "a very long token label that must be clipped rather than wrapped",
            "界界界界界界界界界界",
        )
        for columns in (24, 40, 80, 180):
            for label in labels:
                with self.subTest(columns=columns, label=label):
                    row = _format_distribution_row(
                        1292,
                        label,
                        13.506,
                        selected=True,
                        columns=columns,
                    )
                    self.assertLessEqual(_cell_width(row), columns)


if __name__ == "__main__":
    unittest.main()
