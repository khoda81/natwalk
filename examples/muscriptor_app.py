# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "muscriptor @ git+https://github.com/muscriptor/muscriptor",
# ]
# ///

"""Standalone launcher for the MuScriptor natwalk TUI.

Run this file directly with ``uv run`` from any directory. Its PEP 723 metadata
provides MuScriptor and its dependencies; natwalk itself is imported from the
local checkout containing this script.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path

# PEP 723 scripts run in an isolated environment, so make this checkout's
# package importable without installing it into the script environment.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# The examples directory is already on sys.path when this file is executed.
# Importing the compact module installs its render_screen override on the main
# MuScriptor example, then we invoke that example's normal CLI entry point.
import muscriptor_cli_compact as compact  # noqa: E402


@contextlib.contextmanager
def alternate_screen() -> Iterator[None]:
    """Use the terminal's alternate screen and always restore the normal one."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        yield
        return

    # 1049: save cursor + enter alternate screen. 25: hide cursor while the
    # full-screen app is active. The finally block restores both on normal
    # exit, q, Ctrl-C, or a Python exception.
    sys.stdout.write("\033[?1049h\033[H\033[?25l")
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


def main() -> None:
    with alternate_screen():
        compact.cli.main()


if __name__ == "__main__":
    main()
