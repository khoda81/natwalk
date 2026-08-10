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
import os
import select
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
import muscriptor_cli_compact as compact  # noqa: E402, I001


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
    """Use a full-screen cbreak terminal session and restore it unconditionally."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        yield
        return

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    # 1049: save cursor + enter alternate screen. 25: hide cursor while the
    # full-screen app is active. Keep stdin in cbreak mode for the entire
    # session so keys are immediate and never echoed, while preserving output
    # processing (notably NL -> CRLF) so every printed line returns to column 0.
    sys.stdout.write("\033[?1049h\033[H\033[?25l")
    sys.stdout.flush()
    tty.setcbreak(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


# Bytes that belong to later key events must never be consumed while parsing
# the current event. This matters when rendering is slow and several escape
# sequences have accumulated in the terminal input queue.
_pending_input = bytearray()


def _read_byte(fd: int, timeout: float) -> bytes | None:
    if _pending_input:
        return bytes((_pending_input.pop(0),))
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    data = os.read(fd, 1)
    return data or None


def _read_escape(fd: int) -> str:
    """Consume exactly one ANSI escape sequence, never the following key."""
    seq = bytearray(b"\x1b")
    second = _read_byte(fd, 0.05)
    if second is None:
        return "ESC"
    seq.extend(second)

    # A bare Escape immediately followed by an ordinary key is two events for
    # this UI. Put the ordinary byte back rather than accidentally eating it.
    if second not in {b"[", b"O"}:
        _pending_input[:0] = second
        return "ESC"

    # CSI/SS3 sequences end at the first ANSI final byte (0x40..0x7e). Read
    # one byte at a time: a bulk os.read() can contain several queued arrows
    # after a slow render and would collapse all of them into one event.
    for _ in range(30):
        byte = _read_byte(fd, 0.05)
        if byte is None:
            break
        seq.extend(byte)
        if 0x40 <= byte[0] <= 0x7E:
            break

    text = bytes(seq).decode("ascii", errors="ignore")
    return compact._navigation_override(compact.cli._decode_escape_sequence(text))


def read_key(timeout: float = 0.20) -> str | None:
    """Read exactly one queued terminal event without dropping later events."""
    if not sys.stdin.isatty():
        return input("> ").strip()[:1]

    fd = sys.stdin.fileno()
    first = _read_byte(fd, timeout)
    if first is None:
        return None
    if first == b"\x03":
        raise KeyboardInterrupt
    if first == b"\t":
        return "TAB"
    if first == b"\x1b":
        return _read_escape(fd)
    return first.decode("utf-8", errors="ignore")


compact.cli.read_key = read_key


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
