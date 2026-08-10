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
import termios
import tty
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


# In the focused tree, an ellipsis is a probability region, not a pagination
# affordance. Its position in the tree plus the exact nat cost already says
# what matters; counts and "deviations along N steps" were visually noisy.
_normal_ellipsis = compact._normal_ellipsis
_deviation_ellipsis = compact._deviation_ellipsis


def _bare_normal_ellipsis(entry: compact.cli.TreeEntry) -> compact._CompactNode:
    node = _normal_ellipsis(entry)
    node.label = "…"
    return node


def _bare_deviation_ellipsis(
    sources: list[tuple[compact.cli.TreeEntry, compact.cli.TreeEntry]],
) -> compact._CompactNode | None:
    node = _deviation_ellipsis(sources)
    if node is not None:
        node.label = "…"
    return node


compact._normal_ellipsis = _bare_normal_ellipsis
compact._deviation_ellipsis = _bare_deviation_ellipsis


@contextlib.contextmanager
def terminal_session() -> Iterator[None]:
    """Use a full-screen raw terminal session and restore it unconditionally."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        yield
        return

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    # 1049: save cursor + enter alternate screen. 25: hide cursor while the
    # full-screen app is active. Keep stdin raw for the entire session so no
    # key can arrive in the tiny canonical/echo window between UI refreshes.
    sys.stdout.write("\033[?1049h\033[H\033[?25l")
    sys.stdout.flush()
    tty.setraw(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


def _default_auto_tree_height() -> None:
    """Let the full-screen app use all available rows unless overridden."""
    if any(arg == "--tree-lines" or arg.startswith("--tree-lines=") for arg in sys.argv[1:]):
        return
    sys.argv.extend(["--tree-lines", "0"])


def main() -> None:
    _default_auto_tree_height()
    with terminal_session():
        compact.cli.main()


if __name__ == "__main__":
    main()
