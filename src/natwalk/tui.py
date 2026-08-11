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
    frame.append(
        _line(
            f"search {len(tree.nodes)} nodes  ·  {len(tree.nodes)} distributions"
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

    if not distribution.tokens:
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


class App:
    """Interactive TUI state and key dispatch around one natwalk session."""

    def __init__(
        self,
        cursor: Cursor,
        describe: DescribeToken,
        *,
        title: str,
        max_tokens: int,
        budget_nats: float,
        budget_step: float,
        lines: int | None,
    ) -> None:
        self.session = Session(cursor)
        self.worker = SearchWorker(self.session.search)
        self.describe = describe
        self.title = title
        self.max_tokens = max_tokens
        self.budget_nats = budget_nats
        self.budget_step = budget_step
        self.lines = lines

        self.history: list[Checkpoint] = []
        self.view = View(node=self.session.root)
        self.view_history: list[View] = []
        self.completion_index = 0
        self.suggestions: tuple[Suggestion, ...] = ()
        self.debug = False
        self.quit_requested = False
        self.last_render = 0.0

    @property
    def terminal(self) -> bool:
        return not self.session.tree[self.session.root].distribution.tokens

    def render(self) -> None:
        self.suggestions, self.completion_index = _render(
            self.session,
            self.describe,
            self.view,
            title=self.title,
            budget_nats=self.budget_nats,
            completion_index=self.completion_index,
            max_tokens=self.max_tokens,
            lines=self.lines,
            debug=self.debug,
            undo_depth=len(self.history),
        )
        self.last_render = time.monotonic()

    def handle_keys(self, keys: tuple[str, ...]) -> bool:
        """Apply one buffered key burst and return whether the frame changed."""
        redraw = False
        for key in keys:
            redraw = self.handle_key(key) or redraw
            if self.quit_requested:
                break
        return redraw

    def handle_key(self, key: str) -> bool:
        """Apply one key. Return whether the visible frame changed."""
        if key.lower() == "q":
            self.quit_requested = True
            return False
        if key.lower() == "d":
            self.debug = not self.debug
            return True
        if key == "[":
            self.budget_nats = max(0.0, self.budget_nats - self.budget_step)
            self.completion_index = 0
            return True
        if key == "]":
            self.budget_nats += self.budget_step
            self.completion_index = 0
            return True
        if key == "TAB" and self.suggestions:
            self.completion_index = (self.completion_index + 1) % len(self.suggestions)
            return True
        if key == "BACKTAB" and self.suggestions:
            self.completion_index = (self.completion_index - 1) % len(self.suggestions)
            return True
        if key == "UP":
            self.view = move(self.session.tree, self.view, -1)
            return True
        if key == "DOWN":
            self.view = move(self.session.tree, self.view, 1)
            return True
        if key == "LEFT" and self.view.node != self.session.root:
            self._leave_child()
            return True
        if key in ("\x7f", "\b"):
            return self._undo()
        if key in (" ", "\r", "\n"):
            return self._accept()
        if key == "RIGHT":
            return self._enter_child()
        return False

    def _leave_child(self) -> None:
        if self.view_history:
            self.view = self.view_history.pop()
        else:
            self.view = parent(self.session.tree, self.view)

    def _undo(self) -> bool:
        if not self.history:
            return False
        with self.worker.access():
            self.session.restore(self.history.pop())
        self._reset_view()
        return True

    def _accept(self) -> bool:
        suggestion = self._selected_suggestion()
        if not suggestion.tokens:
            return False
        with self.worker.access():
            self.history.append(self.session.checkpoint())
            self.session.commit(suggestion.tokens)
        self._reset_view()
        return True

    def _selected_suggestion(self) -> Suggestion:
        if self.suggestions:
            return self.suggestions[self.completion_index % len(self.suggestions)]
        return greedy(
            self.session.tree,
            self.session.root,
            max_nats=self.budget_nats,
            max_tokens=self.max_tokens,
        )

    def _enter_child(self) -> bool:
        distribution = self.session.tree[self.view.node].distribution
        if not distribution.tokens:
            return False

        selected = min(self.view.selected_rank, len(distribution) - 1)
        child = self.session.tree.child(self.view.node, selected)
        if child is None:
            with self.worker.access():
                child = self.session.inspect_child(self.view.node, selected)

        self.view_history.append(self.view)
        self.view = View(node=child)
        return True

    def _reset_view(self) -> None:
        self.view = View(node=self.session.root)
        self.view_history.clear()
        self.completion_index = 0


def run_tui(
    cursor: Cursor,
    describe: DescribeToken,
    *,
    title: str,
    max_tokens: int = 256,
    budget_nats: float = 1.5,
    budget_step: float = 0.25,
    lines: int | None = None,
    exit_on_quit: bool = False,
) -> None:
    """Run the terminal UI.

    ``exit_on_quit`` is intended for standalone CLI applications. It restores
    the terminal and then terminates the process immediately instead of waiting
    for an uninterruptible native model call in the daemon search thread.
    """
    app = App(
        cursor,
        describe,
        title=title,
        max_tokens=max_tokens,
        budget_nats=budget_nats,
        budget_step=budget_step,
        lines=lines,
    )
    exit_code: int | None = None

    try:
        with _terminal():
            app.render()
            if app.terminal:
                return

            app.worker.start()
            while not app.terminal:
                app.worker.raise_if_failed()
                redraw = app.handle_keys(_read_keys())
                if app.quit_requested:
                    exit_code = 0
                    break
                if redraw or time.monotonic() - app.last_render >= _REDRAW_SECONDS:
                    app.render()
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        if exit_code is not None and exit_on_quit:
            app.worker.request_stop()
        else:
            app.worker.close()

    if exit_code is not None and exit_on_quit:
        os._exit(exit_code)
