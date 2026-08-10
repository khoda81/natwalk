"""Print raw terminal key sequences without Python text buffering."""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty

_NAMES = {
    b"\x1b[A": "UP (CSI)",
    b"\x1b[B": "DOWN (CSI)",
    b"\x1b[C": "RIGHT (CSI)",
    b"\x1b[D": "LEFT (CSI)",
    b"\x1bOA": "UP (SS3)",
    b"\x1bOB": "DOWN (SS3)",
    b"\x1bOC": "RIGHT (SS3)",
    b"\x1bOD": "LEFT (SS3)",
    b"\x1b[Z": "SHIFT-TAB",
    b"\t": "TAB",
    b"\x7f": "BACKSPACE",
}


def read_sequence(fd: int, *, quiet: float = 0.05) -> bytes:
    """Read one key plus all bytes arriving before a short quiet period."""
    data = bytearray(os.read(fd, 1))
    deadline = time.monotonic() + quiet
    while len(data) < 64:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            break
        chunk = os.read(fd, 64 - len(data))
        if not chunk:
            break
        data.extend(chunk)
        deadline = time.monotonic() + quiet
    return bytes(data)


def main() -> None:
    if not sys.stdin.isatty():
        raise SystemExit("stdin is not a TTY")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print("Press keys; press q by itself to quit.")
    print("Try ↑ ↓ ← →, Tab, Shift-Tab, [, ], d.\n")
    try:
        tty.setraw(fd)
        while True:
            data = read_sequence(fd)
            if data == b"q":
                break
            name = _NAMES.get(data, "unknown")
            hex_bytes = " ".join(f"{byte:02x}" for byte in data)
            sys.stdout.write(
                f"{name:<14} bytes=[{hex_bytes}] repr={data!r}\r\n"
            )
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
