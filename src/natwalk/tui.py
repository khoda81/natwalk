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
from dataclasses import dataclass

from .engine import CommandId, CursorFactory, EngineClient
from .query import Suggestion, completions, greedy
from .tree import NodeId, Tree
from .view import CompactRow, View, compact_rows, forest_nats, move, parent

type DescribeToken = Callable[[int], str]
type DecodeTokens = Callable[[tuple[int, ...]], str]

_KEY_POLL_SECONDS = 0.05
_REDRAW_SECONDS = 0.25
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


def _line(text: str, columns: int) -> str:
    return _clip(text, columns)


def _paint(text: str, code: str, *, color: bool) -> str:
    if not color or not text:
        return text
    return f"\033[{code}m{text}\033[0m"


def _nat_color(nats: float) -> str:
    """Map surprisal to a readable 256-color accent for structure, not labels."""
    if nats < 1.0:
        return "38;5;82"
    if nats < 2.0:
        return "38;5;45"
    if nats < 4.0:
        return "38;5;39"
    if nats < 8.0:
        return "38;5;75"
    if nats < 12.0:
        return "38;5;141"
    return "38;5;244"


def _wrap_spans(
    spans: tuple[tuple[str, str], ...],
    width: int,
    *,
    color: bool,
) -> tuple[str, ...]:
    """Wrap styled plain-text spans without counting ANSI escapes as cells."""
    if width <= 0:
        return ("",)

    lines: list[list[tuple[str, str]]] = [[]]
    used = 0

    def append(char: str, style: str) -> None:
        nonlocal used
        if lines[-1] and lines[-1][-1][1] == style:
            text, _ = lines[-1][-1]
            lines[-1][-1] = (text + char, style)
        else:
            lines[-1].append((char, style))
        used += _cell_width(char)

    for text, style in spans:
        for char in text:
            if char == "\n":
                lines.append([])
                used = 0
                continue

            char_width = _cell_width(char)
            if used and char_width and used + char_width > width:
                lines.append([])
                used = 0
            append(char, style)

    return tuple(
        "".join(_paint(text, style, color=color) if style else text for text, style in line)
        for line in lines
    )


def _sequence_text(
    describe: DescribeToken,
    tokens: tuple[int, ...],
    decode: DecodeTokens | None,
) -> str:
    if not tokens:
        return ""
    if decode is not None:
        return decode(tokens)
    return " · ".join(describe(token) for token in tokens)


def _context_text(
    context: str,
    describe: DescribeToken,
    tokens: tuple[int, ...],
    decode: DecodeTokens | None,
) -> str:
    suffix = _sequence_text(describe, tokens, decode)
    if not context:
        return suffix
    if not suffix:
        return context
    if decode is not None:
        return context + suffix
    return f"{context} · {suffix}"


def _context_spans(
    context_text: str,
    suggestion_text: str,
    decode: DecodeTokens | None,
) -> tuple[tuple[str, str], ...]:
    """Compose context and highlighted completion without corrupting exact text."""
    spans: list[tuple[str, str]] = []
    if context_text:
        spans.append((context_text, ""))
    elif not suggestion_text:
        spans.append(("∅", ""))

    if context_text and suggestion_text and decode is None:
        spans.append((" · ", ""))
    if suggestion_text:
        spans.append((suggestion_text, "1"))
    return tuple(spans)


def _format_tree_row(
    row: CompactRow,
    describe: DescribeToken,
    *,
    selected: bool,
    columns: int,
    color: bool,
) -> str:
    marker = "❯ " if selected else "  "
    ancestors = "".join("   " if was_last else "│  " for was_last in row.ancestor_last)
    branch = "└─ " if row.is_last else "├─ "

    label = " · ".join(describe(token) for token in row.tokens)
    if row.forest:
        label = f"{label} · …" if label else "…"
    elif row.open_ended:
        label = f"{label} · …"

    suffix = f"  {row.path_nats:7.3f} nat"
    room = max(
        0,
        columns
        - _cell_width(marker)
        - _cell_width(ancestors)
        - _cell_width(branch)
        - _cell_width(suffix),
    )
    label = _fit(label, room)

    return (
        marker
        + _paint(ancestors, "2", color=color)
        + _paint(branch, _nat_color(row.edge_nats), color=color)
        + _paint(label, "2" if row.forest else "", color=color)
        + _paint(suffix, _nat_color(row.path_nats), color=color)
    )


def _format_forest_summary(
    direction: str,
    count: int,
    nats: float,
    *,
    columns: int,
    color: bool,
) -> str:
    prefix = f"  {direction} … {count} ranks {'above' if direction == '↑' else 'below'}"
    suffix = f"  {nats:7.3f} nat"
    room = max(0, columns - _cell_width(suffix))
    return _paint(_fit(prefix, room), "2", color=color) + _paint(
        suffix, _nat_color(nats), color=color
    )


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
    tree: Tree,
    root: NodeId,
    frontier: int,
    describe: DescribeToken,
    view: View,
    *,
    title: str,
    context: str,
    decode_tokens: DecodeTokens | None,
    budget_nats: float,
    completion_index: int,
    max_tokens: int,
    lines: int | None,
    debug: bool,
    undo_depth: int,
) -> tuple[tuple[Suggestion, ...], int]:
    """Render one frame from replicated tree state without contacting the engine."""
    columns, terminal_rows = _dimensions()
    color = sys.stdout.isatty()
    rule = "─" * columns

    suggestions = completions(
        tree,
        view.node,
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
            view.node,
            max_nats=budget_nats,
            max_tokens=max_tokens,
        )

    frame: list[str] = []
    frame.append(_paint(_line(title, columns), "1", color=color))
    frame.append(_line(rule, columns))
    frame.append(
        _line(
            f"budget {budget_nats:.2f} nat / {budget_nats / math.log(2):.2f} bit"
            f"  ·  {len(tree.nodes)} nodes"
            f"  ·  {frontier} frontier",
            columns,
        )
    )
    if debug:
        frame.append(
            _line(
                f"root {root}  ·  view {view.node}:{view.first_rank}/{view.selected_rank}"
                f"  ·  undo {undo_depth}",
                columns,
            )
        )

    frame.append(_line(rule, columns))
    committed = tree.path(root)
    focus = tree.path_from(root, view.node)
    context_text = _context_text(context, describe, (*committed, *focus), decode_tokens)
    suggestion_text = _sequence_text(describe, suggestion.tokens, decode_tokens)
    if suggestion_text and not suggestion.complete:
        suggestion_text += "…"
    context_lines = _wrap_spans(
        _context_spans(context_text, suggestion_text, decode_tokens),
        columns,
        color=color,
    )
    frame.extend(context_lines)
    frame.append(_line(rule, columns))

    footer = (
        _line(rule, columns),
        _line(
            "↑↓ rank  ·  ← parent  ·  → child  ·  Space accept  ·  Backspace undo",
            columns,
        ),
        _line(
            "Tab/Shift-Tab suggestion  ·  [ ] budget  ·  d debug  ·  q quit",
            columns,
        ),
    )
    tree_lines = max(2, terminal_rows - len(frame) - len(footer))
    if lines is not None:
        tree_lines = min(tree_lines, max(2, lines))

    distribution = tree[view.node].distribution
    if not distribution.tokens:
        frame.append(_line("  ∅ terminal", columns))
    else:
        selected = min(view.selected_rank, len(distribution) - 1)
        roots_above = min(max(2, tree_lines // 8), selected - view.first_rank)
        start = max(view.first_rank, selected - roots_above)
        edge_limit = max(64, tree_lines * 6)
        rendered = compact_rows(
            tree,
            view,
            edge_limit=edge_limit,
            first_rank=start,
        )
        if not any(
            row.parent == view.node and row.rank == selected and not row.forest for row in rendered
        ):
            start = selected
            rendered = compact_rows(
                tree,
                view,
                edge_limit=edge_limit,
                first_rank=start,
            )

        above = start - view.first_rank
        reserve_above = int(above > 0)
        reserve_below = int(start < len(distribution) - 1)
        row_budget = max(0, tree_lines - reserve_above - reserve_below)
        visible = rendered[:row_budget]

        if above:
            frame.append(
                _format_forest_summary(
                    "↑",
                    above,
                    forest_nats(distribution, view.first_rank, start),
                    columns=columns,
                    color=color,
                )
            )

        for row in visible:
            frame.append(
                _format_tree_row(
                    row,
                    describe,
                    selected=(not row.forest and row.parent == view.node and row.rank == selected),
                    columns=columns,
                    color=color,
                )
            )

        root_ranks = [
            row.rank
            for row in visible
            if row.depth == 0 and row.parent == view.node and not row.forest
        ]
        last_root = max(root_ranks, default=start - 1)
        tail_start = max(start, last_root + 1)
        if tail_start < len(distribution):
            frame.append(
                _format_forest_summary(
                    "↓",
                    len(distribution) - tail_start,
                    forest_nats(distribution, tail_start),
                    columns=columns,
                    color=color,
                )
            )

    frame.extend(footer)
    prefix = "\033[2J\033[H" if sys.stdout.isatty() else ""
    sys.stdout.write(prefix + "\n".join(frame) + "\n")
    sys.stdout.flush()
    return suggestions, completion_index


@dataclass(frozen=True, slots=True)
class _PendingEnter:
    command_id: CommandId
    view: View


@dataclass(frozen=True, slots=True)
class _PendingCommit:
    command_id: CommandId


@dataclass(frozen=True, slots=True)
class _PendingUndo:
    command_id: CommandId


type _Pending = _PendingEnter | _PendingCommit | _PendingUndo


class App:
    """Interactive client state around one process-isolated natwalk engine."""

    def __init__(
        self,
        factory: CursorFactory,
        describe: DescribeToken,
        *,
        title: str,
        context: str,
        decode_tokens: DecodeTokens | None,
        max_tokens: int,
        budget_nats: float,
        budget_step: float,
        lines: int | None,
    ) -> None:
        self.engine = EngineClient(factory)
        self.engine.start()
        try:
            self.engine.wait_ready()
        except BaseException:
            self.engine.terminate()
            raise

        self.describe = describe
        self.title = title
        self.context = context
        self.decode_tokens = decode_tokens
        self.max_tokens = max_tokens
        self.budget_nats = budget_nats
        self.budget_step = budget_step
        self.lines = lines

        self.view = View(node=self.root)
        self.view_history: list[View] = []
        self.completion_index = 0
        self.suggestions: tuple[Suggestion, ...] = ()
        self.pending: _Pending | None = None
        self.debug = False
        self.quit_requested = False
        self.last_render = 0.0

    @property
    def tree(self) -> Tree:
        return self.engine.tree

    @property
    def root(self) -> NodeId:
        root = self.engine.root
        if root is None:
            raise RuntimeError("engine has no root")
        return root

    @property
    def terminal(self) -> bool:
        return not self.tree[self.root].distribution.tokens

    def poll(self) -> bool:
        """Apply engine events and complete any pending UI intent."""
        changed = self.engine.poll() > 0
        pending = self.pending
        if pending is None:
            return changed

        done = self.engine.take_done(pending.command_id)
        if done is None:
            return changed

        self.pending = None
        if isinstance(pending, _PendingEnter):
            if self.view == pending.view:
                self.view_history.append(pending.view)
                self.view = View(node=done.node)
                self.completion_index = 0
        else:
            self._reset_view()
        return True

    def render(self) -> None:
        self.suggestions, self.completion_index = _render(
            self.tree,
            self.root,
            self.engine.frontier,
            self.describe,
            self.view,
            title=self.title,
            context=self.context,
            decode_tokens=self.decode_tokens,
            budget_nats=self.budget_nats,
            completion_index=self.completion_index,
            max_tokens=self.max_tokens,
            lines=self.lines,
            debug=self.debug,
            undo_depth=self.engine.history_depth,
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
            self.view = move(self.tree, self.view, -1)
            return True
        if key == "DOWN":
            self.view = move(self.tree, self.view, 1)
            return True
        if key == "LEFT" and self.view.node != self.root:
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
            self.view = parent(self.tree, self.view)
        self.completion_index = 0

    def _undo(self) -> bool:
        if self.pending is not None or self.engine.history_depth == 0:
            return False
        self.pending = _PendingUndo(self.engine.undo())
        return False

    def _accept(self) -> bool:
        if self.pending is not None:
            return False
        suggestion = self._selected_suggestion()
        focus = self.tree.path_from(self.root, self.view.node)
        tokens = (*focus, *suggestion.tokens)
        if not tokens:
            return False
        self.pending = _PendingCommit(self.engine.commit(tokens))
        return False

    def _selected_suggestion(self) -> Suggestion:
        if self.suggestions:
            return self.suggestions[self.completion_index % len(self.suggestions)]
        return greedy(
            self.tree,
            self.view.node,
            max_nats=self.budget_nats,
            max_tokens=self.max_tokens,
        )

    def _enter_child(self) -> bool:
        distribution = self.tree[self.view.node].distribution
        if not distribution.tokens:
            return False

        selected = min(self.view.selected_rank, len(distribution) - 1)
        child = self.tree.child(self.view.node, selected)
        if child is None:
            if self.pending is not None:
                return False
            self.pending = _PendingEnter(
                self.engine.inspect(self.view.node, selected),
                self.view,
            )
            return False

        self.view_history.append(self.view)
        self.view = View(node=child)
        self.completion_index = 0
        return True

    def _reset_view(self) -> None:
        self.view = View(node=self.root)
        self.view_history.clear()
        self.completion_index = 0


def run_tui(
    factory: CursorFactory,
    describe: DescribeToken,
    *,
    title: str,
    context: str = "",
    decode_tokens: DecodeTokens | None = None,
    max_tokens: int = 256,
    budget_nats: float = 1.5,
    budget_step: float = 0.25,
    lines: int | None = None,
) -> None:
    """Run a responsive terminal client around a process-isolated model engine."""
    app = App(
        factory,
        describe,
        title=title,
        context=context,
        decode_tokens=decode_tokens,
        max_tokens=max_tokens,
        budget_nats=budget_nats,
        budget_step=budget_step,
        lines=lines,
    )
    hard_stop = False

    try:
        with _terminal():
            app.render()
            while not app.terminal:
                redraw = app.poll()
                redraw = app.handle_keys(_read_keys()) or redraw
                if app.quit_requested:
                    hard_stop = True
                    break
                if redraw or time.monotonic() - app.last_render >= _REDRAW_SECONDS:
                    app.render()
    except KeyboardInterrupt:
        hard_stop = True
    finally:
        if hard_stop:
            app.engine.terminate()
        else:
            app.engine.close()
