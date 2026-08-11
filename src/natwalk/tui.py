"""Shared terminal UI for natwalk model backends."""

from __future__ import annotations

import math
import os
import select
import shutil
import sys
import termios
import time
import tty
import unicodedata
from collections.abc import Callable
from contextlib import contextmanager

from .model import Cursor
from .query import Suggestion, completions, greedy
from .session import Checkpoint, Session
from .view import View, move, parent
from .worker import SearchWorker

type DescribeToken = Callable[[int], str]

_KEY_POLL_SECONDS = 0.05
_REDRAW_SECONDS = 0.25
_FRAME_OVERHEAD = 13
_ESCAPE_KEYS = {
    b"\x1b[A": "UP",
    b"\x1b[B": "DOWN",
    b"\x1b[C": "RIGHT",
    b"\x1b[D": "LEFT",
    b"\x1bOA": "UP",
    b"\x1bOB": "DOWN",
    b"\x1bOC": "RIGHT",
    b"\x1bOD": "LEFT",
    b"\x1b[Z": "BACKTAB",
}


def _cell_width(text: str) -> int:
    """Return the terminal-cell width of plain Unicode text."""
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        if unicodedata.category(char) in {"Cc", "Cf"}:
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _clip(text: str, width: int) -> str:
    """Clip plain Unicode text to at most ``width`` terminal cells."""
    if width <= 0:
        return ""
    if _cell_width(text) <= width:
        return text
    if width == 1:
        return "…"

    target = width - 1
    out: list[str] = []
    used = 0
    for char in text:
        char_width = _cell_width(char)
        if char_width == 0:
            if out:
                out.append(char)
            continue
        if used + char_width > target:
            break
        out.append(char)
        used += char_width
    return "".join(out) + "…"


def _fit(text: str, width: int) -> str:
    clipped = _clip(text, width)
    return clipped + " " * max(0, width - _cell_width(clipped))


def _dimensions() -> tuple[int, int]:
    size = shutil.get_terminal_size((100, 30))
    columns = size.columns if size.columns > 0 else 100
    rows = size.lines if size.lines > 0 else 30
    return columns, rows


def _distribution_line_count(
    terminal_rows: int,
    requested: int | None,
    *,
    debug: bool,
) -> int:
    available = max(4, terminal_rows - _FRAME_OVERHEAD - int(debug))
    if requested is None:
        return available
    return min(max(4, requested), available)


def _line(text: str, columns: int) -> str:
    return _clip(text, columns)


def _tokens(describe: DescribeToken, tokens: tuple[int, ...], limit: int = 18) -> str:
    if not tokens:
        return "∅"
    shown = tokens[-limit:]
    prefix = "… · " if len(tokens) > limit else ""
    return prefix + " · ".join(describe(token) for token in shown)


def _format_distribution_row(
    rank: int,
    label: str,
    cost: float,
    *,
    selected: bool,
    columns: int,
) -> str:
    marker = "❯" if selected else " "
    prefix = f"{marker} {rank:5d}  "
    suffix = f"  {cost:7.3f} nat"
    room = max(0, columns - _cell_width(prefix) - _cell_width(suffix))
    return _clip(prefix + _fit(label, room) + suffix, columns)


@contextmanager
def _terminal():
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sys.stdout.write("\033[?1049h\033[H\033[?25l")
    sys.stdout.flush()
    tty.setcbreak(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


def _read_key(timeout: float = _KEY_POLL_SECONDS) -> str | None:
    if not sys.stdin.isatty():
        return input("> ").strip()[:1]

    fd = sys.stdin.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    first = os.read(fd, 1)
    if first == b"\x03":
        raise KeyboardInterrupt
    if first == b"\t":
        return "TAB"
    if first != b"\x1b":
        return first.decode("utf-8", errors="ignore")

    sequence = bytearray(first)
    deadline = time.monotonic() + 0.05
    while len(sequence) < 16:
        key = _ESCAPE_KEYS.get(bytes(sequence))
        if key is not None:
            return key

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            break
        chunk = os.read(fd, 1)
        if not chunk:
            break
        sequence.extend(chunk)
        if sequence[-1] == ord("~"):
            break

    return _ESCAPE_KEYS.get(bytes(sequence), "ESC")


def _read_keys(timeout: float = _KEY_POLL_SECONDS) -> tuple[str, ...]:
    """Read one key with ``timeout``, then drain every already-buffered key."""
    first = _read_key(timeout)
    if first is None:
        return ()

    keys = [first]
    while (key := _read_key(0.0)) is not None:
        keys.append(key)
    return tuple(keys)


def _render(
    session: Session,
    describe: DescribeToken,
    view: View,
    *,
    title: str,
    budget_nats: float,
    completion_index: int,
    max_tokens: int,
    lines: int | None,
    debug: bool,
    undo_depth: int,
) -> tuple[tuple[Suggestion, ...], int]:
    """Render one eventually-consistent frame without mutating model/search state."""
    columns, terminal_rows = _dimensions()
    rule = "─" * columns
    tree = session.tree
    suggestions = completions(
        tree,
        session.root,
        max_nats=budget_nats,
        max_tokens=max_tokens,
    )
    if suggestions:
        completion_index %= len(suggestions)
        suggestion = suggestions[completion_index]
    else:
        completion_index = 0
        suggestion = greedy(
            tree,
            session.root,
            max_nats=budget_nats,
            max_tokens=max_tokens,
        )

    frame: list[str] = []
    frame.append(_line(title, columns))
    frame.append(_line(rule, columns))
    frame.append(
        _line(
            f"budget {budget_nats:.2f} nat / {budget_nats / math.log(2):.2f} bit",
            columns,
        )
    )

    known = sum(node.distribution is not None for node in tree.nodes)
    frame.append(
        _line(
            f"search {len(tree.nodes)} nodes  ·  {known} distributions"
            f"  ·  {len(session.search.frontier)} frontier",
            columns,
        )
    )
    if debug:
        frame.append(
            _line(
                f"root {session.root}  ·  view {view.node}:{view.first_rank}/{view.selected_rank}"
                f"  ·  undo {undo_depth}",
                columns,
            )
        )

    frame.append(_line(rule, columns))
    frame.append(_line(f"committed   {_tokens(describe, tree.path(session.root))}", columns))
    if suggestion.tokens:
        suffix = " · …" if not suggestion.complete else ""
        frame.append(
            _line(
                f"suggestion  {_tokens(describe, suggestion.tokens)}{suffix}"
                f"  [{suggestion.nats:.3f}/{budget_nats:.2f} nat]",
                columns,
            )
        )
    elif suggestion.next_nats is not None:
        frame.append(
            _line(
                f"suggestion  — next token costs {suggestion.next_nats:.3f} nat",
                columns,
            )
        )
    else:
        frame.append(_line("suggestion  ⟳ search has not reached the greedy tail yet", columns))

    distribution = tree[view.node].distribution
    focus = tree.path_from(session.root, view.node)
    frame.append(_line(f"focus       {_tokens(describe, focus, limit=10)}", columns))
    frame.append(_line(rule, columns))

    if distribution is None:
        frame.append(_line("  ⟳ distribution not discovered yet", columns))
    elif not distribution.tokens:
        frame.append(_line("  ∅ terminal", columns))
    else:
        selected = min(view.selected_rank, len(distribution) - 1)
        line_count = _distribution_line_count(
            terminal_rows,
            lines,
            debug=debug,
        )
        start = max(view.first_rank, selected - line_count // 2)
        start = min(start, max(view.first_rank, len(distribution) - line_count))
        end = min(len(distribution), start + line_count)
        if start > view.first_rank:
            frame.append(_line(f"  ↑ … {start - view.first_rank} ranks above", columns))
        for rank in range(start, end):
            frame.append(
                _format_distribution_row(
                    rank,
                    describe(distribution.tokens[rank]),
                    distribution.nats(rank),
                    selected=rank == selected,
                    columns=columns,
                )
            )
        if end < len(distribution):
            frame.append(_line(f"  ↓ … {len(distribution) - end} ranks below", columns))

    frame.append(_line(rule, columns))
    frame.append(
        _line(
            "↑↓ rank  ·  ← parent  ·  → child  ·  Space accept  ·  Backspace undo",
            columns,
        )
    )
    frame.append(
        _line(
            "Tab/Shift-Tab suggestion  ·  [ ] budget  ·  d debug  ·  q quit",
            columns,
        )
    )

    prefix = "\033[2J\033[H" if sys.stdout.isatty() else ""
    sys.stdout.write(prefix + "\n".join(frame) + "\n")
    sys.stdout.flush()
    return suggestions, completion_index


def run_tui(
    cursor: Cursor,
    describe: DescribeToken,
    *,
    title: str,
    max_tokens: int = 256,
    budget_nats: float = 1.5,
    budget_step: float = 0.25,
    lines: int | None = None,
) -> None:
    session = Session(cursor)
    history: list[Checkpoint] = []
    view = View(node=session.root)
    completion_index = 0
    debug = False

    try:
        with _terminal():
            suggestions, completion_index = _render(
                session,
                describe,
                view,
                title=title,
                budget_nats=budget_nats,
                completion_index=completion_index,
                max_tokens=max_tokens,
                lines=lines,
                debug=debug,
                undo_depth=len(history),
            )
            root_distribution = session.tree[session.root].distribution
            if root_distribution is not None and not root_distribution.tokens:
                return

            last_render = time.monotonic()
            with SearchWorker(session.search) as worker:
                while True:
                    worker.raise_if_failed()
                    keys = _read_keys()
                    dirty = False
                    quit_requested = False

                    for key in keys:
                        if key.lower() == "q":
                            quit_requested = True
                            break
                        if key.lower() == "d":
                            debug = not debug
                            dirty = True
                        elif key == "[":
                            budget_nats = max(0.0, budget_nats - budget_step)
                            completion_index = 0
                            dirty = True
                        elif key == "]":
                            budget_nats += budget_step
                            completion_index = 0
                            dirty = True
                        elif key == "TAB" and suggestions:
                            completion_index = (completion_index + 1) % len(suggestions)
                            dirty = True
                        elif key == "BACKTAB" and suggestions:
                            completion_index = (completion_index - 1) % len(suggestions)
                            dirty = True
                        elif key == "UP":
                            view = move(session.tree, view, -1)
                            dirty = True
                        elif key == "DOWN":
                            view = move(session.tree, view, 1)
                            dirty = True
                        elif key == "LEFT" and view.node != session.root:
                            view = parent(session.tree, view)
                            dirty = True
                        elif key in ("\x7f", "\b"):
                            if history:
                                with worker.access():
                                    session.restore(history.pop())
                                view = View(node=session.root)
                                completion_index = 0
                                dirty = True
                        elif key in (" ", "\r", "\n"):
                            if suggestions:
                                suggestion = suggestions[completion_index % len(suggestions)]
                            else:
                                suggestion = greedy(
                                    session.tree,
                                    session.root,
                                    max_nats=budget_nats,
                                    max_tokens=max_tokens,
                                )
                            if suggestion.tokens:
                                with worker.access():
                                    history.append(session.checkpoint())
                                    session.commit(suggestion.tokens)
                                view = View(node=session.root)
                                completion_index = 0
                                dirty = True
                        elif key == "RIGHT":
                            distribution = session.tree[view.node].distribution
                            if distribution is not None and distribution.tokens:
                                selected = min(view.selected_rank, len(distribution) - 1)
                                child = session.tree[view.node].children.get(selected)
                                if child is None or session.tree[child].distribution is None:
                                    with worker.access():
                                        child = session.inspect_child(view.node, selected)
                                view = View(node=child)
                                dirty = True

                    if quit_requested:
                        break

                    now = time.monotonic()
                    if dirty or now - last_render >= _REDRAW_SECONDS:
                        suggestions, completion_index = _render(
                            session,
                            describe,
                            view,
                            title=title,
                            budget_nats=budget_nats,
                            completion_index=completion_index,
                            max_tokens=max_tokens,
                            lines=lines,
                            debug=debug,
                            undo_depth=len(history),
                        )
                        last_render = now

                    root_distribution = session.tree[session.root].distribution
                    if root_distribution is not None and not root_distribution.tokens:
                        break
    except KeyboardInterrupt:
        pass
