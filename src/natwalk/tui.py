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

from .engine import CommandId, CursorFactory, EngineClient
from .query import (
    Suggestion,
    cycle_suggestion,
    normalize_suggestion,
    suggestion_complete,
    suggestion_edges,
    suggestion_tokens,
)
from .tree import NodeId, Tree
from .view import CompactRow, View, forest_nats, move, partition_rows, row_tokens

type DescribeToken = Callable[[int], str]
type DecodeTokens = Callable[[tuple[int, ...]], str]

_KEY_POLL_SECONDS = 0.05
_REDRAW_SECONDS = 0.25
_REVEAL_PAGE = 128
_SUGGESTION_STYLE = "1;38;5;45"
_SELECTED_STYLE = "1;38;5;220"
_FOREST_STYLE = "2;38;5;244"
_PREDICTION_STYLE = "38;5;241"
_VIRIDIS_GAMMA = 0.35
_VIRIDIS_WHITE_MIX = 0.18
_VIRIDIS = (
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
)
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
    if not color or not text or not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def _relative_probability(nats: float, reference_nats: float) -> float:
    """Probability ratio to the best event in the current visible window."""
    if not math.isfinite(nats) or not math.isfinite(reference_nats):
        return 0.0
    return min(1.0, math.exp(min(0.0, reference_nats - nats)))


def _viridis(probability: float) -> str:
    """Return a readable truecolor viridis foreground for a probability ratio."""
    probability = min(1.0, max(0.0, probability))
    value = probability**_VIRIDIS_GAMMA
    position = value * (len(_VIRIDIS) - 1)
    lower = min(int(position), len(_VIRIDIS) - 2)
    fraction = position - lower
    left = _VIRIDIS[lower]
    right = _VIRIDIS[lower + 1]
    rgb = tuple(round(a + (b - a) * fraction) for a, b in zip(left, right, strict=True))
    rgb = tuple(round(channel + (255 - channel) * _VIRIDIS_WHITE_MIX) for channel in rgb)
    return f"38;2;{rgb[0]};{rgb[1]};{rgb[2]}"


def _grayscale(probability: float) -> str:
    """Map relative branch probability to grayscale structural brightness."""
    value = min(1.0, max(0.0, probability))
    level = round(55 + 195 * math.sqrt(value))
    return f"38;2;{level};{level};{level}"


def _minimum_finite(values) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else math.inf


def _ancestor_branch_nats(
    rows: tuple[CompactRow, ...],
) -> tuple[tuple[float, ...], ...]:
    """Return exact radix-ancestor weights without requiring internal rows."""
    return tuple(row.ancestor_nats for row in rows)


def _path_branch_column(tokens: tuple[int, ...], describe: DescribeToken) -> int:
    """Column where a branch after ``tokens`` meets its collapsed token edge."""
    if not tokens:
        return 0
    token_width = sum(_cell_width(describe(token)) for token in tokens)
    return 4 + token_width + 3 * (len(tokens) - 1)


def _branch_prefixes(
    tree: Tree,
    view: View,
    rows: tuple[CompactRow, ...],
) -> set[tuple[int, ...]]:
    """Return concrete token prefixes that own a visible radix branch."""
    return {tree.path_from(view.node, row.parent) for row in rows}


def _row_branch_columns(
    tree: Tree,
    view: View,
    row: CompactRow,
    branch_prefixes: set[tuple[int, ...]],
    describe: DescribeToken,
) -> tuple[tuple[int, ...], int]:
    """Align visible radix connectors with exact token-boundary columns."""
    prefix = tree.path_from(view.node, row.parent)
    ancestor_prefixes = tuple(
        sorted(
            (
                candidate
                for candidate in branch_prefixes
                if len(candidate) < len(prefix) and prefix[: len(candidate)] == candidate
            ),
            key=len,
        )
    )
    if len(ancestor_prefixes) != len(row.ancestor_last):
        raise ValueError("radix ancestor paths must match connector state")
    return (
        tuple(_path_branch_column(candidate, describe) for candidate in ancestor_prefixes),
        _path_branch_column(prefix, describe),
    )


def _row_inline_branches(
    tree: Tree,
    view: View,
    row: CompactRow,
    branch_prefixes: set[tuple[int, ...]],
) -> tuple[int, ...]:
    """Return token-boundary offsets that branch again later in the partition."""
    prefix = tree.path_from(view.node, row.parent)
    tokens = row_tokens(tree, row)
    full_path = (*prefix, *tokens)
    return tuple(
        offset
        for offset in range(1, len(tokens) + 1)
        if full_path[: len(prefix) + offset] in branch_prefixes
    )


def _row_separator_nats(tree: Tree, row: CompactRow) -> tuple[float, ...]:
    """Return local surprisal for each collapsed token boundary."""
    return tuple(tree[edge.parent].distribution.nats(edge.rank) for edge in row.edges[1:])


def _row_preview(
    tree: Tree,
    row: CompactRow,
    *,
    max_tokens: int = 64,
) -> tuple[tuple[int, ...], tuple[float, ...], bool]:
    """Return read-only best-known context beyond one measured row event."""
    if max_tokens <= 0:
        return (), (), False

    tokens: list[int] = []
    separator_nats: list[float] = []

    if row.forest:
        node = row.parent
        if row.edges:
            last = row.edges[-1]
            child = tree.child(last.parent, last.rank)
            assert child is not None
            node = child

        distribution = tree[node].distribution
        assert 0 <= row.forest_start < len(distribution)
        if row.forest_start >= distribution.revealed:
            return (), (), False
        rank = row.forest_start
        tokens.append(distribution.token(rank))
        separator_nats.append(distribution.nats(rank))
        child = tree.child(node, rank)
        if child is None:
            return tuple(tokens), tuple(separator_nats), False
        node = child
    else:
        node = row.child
        if node is None:
            return (), (), False

    while len(tokens) < max_tokens:
        distribution = tree[node].distribution
        if len(distribution) == 0:
            return tuple(tokens), tuple(separator_nats), True
        if distribution.revealed == 0:
            return tuple(tokens), tuple(separator_nats), False

        rank = 0
        tokens.append(distribution.token(rank))
        separator_nats.append(distribution.nats(rank))
        child = tree.child(node, rank)
        if child is None:
            return tuple(tokens), tuple(separator_nats), False
        node = child

    return tuple(tokens), tuple(separator_nats), len(tree[node].distribution) == 0


def _tree_viewport(
    tree: Tree,
    view: View,
    *,
    selected: int,
    tree_lines: int,
) -> tuple[int, int, tuple[CompactRow, ...]]:
    """Choose partition rows while guaranteeing the selected root sibling is visible."""
    roots_above = min(max(2, tree_lines // 8), selected - view.first_rank)
    start = max(view.first_rank, selected - roots_above)

    def visible_from(first_rank: int) -> tuple[int, tuple[CompactRow, ...]]:
        above = first_rank - view.first_rank
        reserve_above = int(above > 0)
        row_budget = max(0, tree_lines - reserve_above)
        rendered = partition_rows(
            tree,
            view,
            row_limit=row_budget,
            first_rank=first_rank,
        )
        return above, rendered

    above, visible = visible_from(start)
    selected_visible = any(
        row.parent == view.node and row.rank == selected and not row.forest for row in visible
    )
    if not selected_visible and start != selected:
        start = selected
        above, visible = visible_from(start)

    return start, above, visible


def _viewport_reveal_target(
    tree: Tree,
    view: View,
    *,
    tree_lines: int,
) -> int:
    """Return the concrete rank horizon needed for this viewport plus one viewport ahead."""
    distribution = tree[view.node].distribution
    if distribution.revealed == 0:
        return min(len(distribution), _REVEAL_PAGE)

    selected = min(view.selected_rank, distribution.revealed - 1)
    start, above, _visible = _tree_viewport(
        tree,
        view,
        selected=selected,
        tree_lines=tree_lines,
    )
    row_budget = max(0, tree_lines - int(above > 0))
    return min(len(distribution), start + 2 * row_budget)


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


def _fit_spans(
    spans: tuple[tuple[str, str], ...],
    width: int,
    *,
    color: bool,
) -> str:
    """Clip and pad styled spans to exactly ``width`` terminal cells."""
    if width <= 0:
        return ""

    plain_width = sum(_cell_width(text) for text, _style in spans)
    clipped = plain_width > width
    target = width - 1 if clipped else width
    out: list[tuple[str, str]] = []
    used = 0

    def append(char: str, style: str) -> None:
        nonlocal used
        if out and out[-1][1] == style:
            text, _ = out[-1]
            out[-1] = (text + char, style)
        else:
            out.append((char, style))
        used += _cell_width(char)

    for text, style in spans:
        for char in text:
            char_width = _cell_width(char)
            if char_width == 0:
                if out:
                    append(char, style)
                continue
            if used + char_width > target:
                break
            append(char, style)
        else:
            continue
        break

    if clipped:
        append("…", out[-1][1] if out else "")

    rendered = "".join(_paint(text, style, color=color) for text, style in out)
    return rendered + " " * max(0, width - used)


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
        spans.append((suggestion_text, _SUGGESTION_STYLE))
    return tuple(spans)


def _row_display_nats(tree: Tree, root: NodeId, view: View, row: CompactRow) -> float:
    """Stable endpoint surprisal from the committed root, independent of scrolling."""
    return tree.path_nats(view.node, ancestor=root) + row.path_nats


def _row_token_styles(
    row: CompactRow,
    suggestion: set[Suggestion],
    *,
    selected: bool,
) -> tuple[str, ...]:
    """Color compact-row edges from structural suggestion/UI state only."""
    styles: list[str] = []
    for edge in row.edges:
        if edge in suggestion:
            style = _SUGGESTION_STYLE
        elif selected:
            style = _SELECTED_STYLE
        elif row.forest:
            style = _FOREST_STYLE
        else:
            style = ""
        styles.append(style)
    return tuple(styles)


def _styled_cells(cells: list[str], styles: list[str], *, color: bool) -> str:
    if len(cells) != len(styles):
        raise ValueError("cell and style counts must match")
    spans: list[tuple[str, str]] = []
    for cell, style in zip(cells, styles, strict=True):
        if spans and spans[-1][1] == style:
            text, _ = spans[-1]
            spans[-1] = (text + cell, style)
        else:
            spans.append((cell, style))
    return "".join(_paint(text, style, color=color) for text, style in spans)


def _structure_prefix(
    row: CompactRow,
    *,
    branch_column: int,
    ancestor_columns: tuple[int, ...],
    ancestor_branch_nats: tuple[float, ...],
    branch_nats: float,
    branch_reference: float,
    color: bool,
) -> tuple[str, int]:
    """Draw radix connectors and return their exact unpainted cell width."""
    if len(ancestor_columns) != len(row.ancestor_last):
        raise ValueError("ancestor column count must match tree depth")
    if len(ancestor_branch_nats) != len(row.ancestor_last):
        raise ValueError("ancestor branch-nat count must match tree depth")

    root_branch = branch_column == 0
    branch = ("└─ " if row.is_last else "├─ ") if root_branch else ("└─" if row.is_last else "├─")
    width = branch_column + _cell_width(branch)
    cells = [" "] * width
    styles = [""] * width

    for was_last, column, ancestor_nats in zip(
        row.ancestor_last,
        ancestor_columns,
        ancestor_branch_nats,
        strict=True,
    ):
        if was_last:
            continue
        if not 0 <= column < width:
            raise ValueError("ancestor connector must precede child branch")
        cells[column] = "│"
        styles[column] = _grayscale(_relative_probability(ancestor_nats, branch_reference))

    glyph_style = _grayscale(_relative_probability(branch_nats, branch_reference))
    for offset, char in enumerate(branch):
        cells[branch_column + offset] = char
        styles[branch_column + offset] = glyph_style
    return _styled_cells(cells, styles, color=color), width


def _format_tree_row(
    row: CompactRow,
    describe: DescribeToken,
    *,
    selected: bool,
    columns: int,
    color: bool,
    tokens: tuple[int, ...] = (),
    display_nats: float | None = None,
    nat_reference: float | None = None,
    branch_nats: float | None = None,
    branch_reference: float | None = None,
    ancestor_branch_nats: tuple[float, ...] | None = None,
    token_styles: tuple[str, ...] | None = None,
    ancestor_columns: tuple[int, ...] | None = None,
    branch_column: int | None = None,
    inline_branches: tuple[int, ...] = (),
    separator_nats: tuple[float, ...] | None = None,
    preview_tokens: tuple[int, ...] = (),
    preview_separator_nats: tuple[float, ...] = (),
    preview_complete: bool = True,
) -> str:
    display_nats = row.path_nats if display_nats is None else display_nats
    nat_reference = display_nats if nat_reference is None else nat_reference
    branch_nats = row.edge_nats if branch_nats is None else branch_nats
    branch_reference = branch_nats if branch_reference is None else branch_reference
    if ancestor_branch_nats is None:
        ancestor_branch_nats = row.ancestor_nats or (branch_nats,) * len(row.ancestor_last)
    if len(ancestor_branch_nats) != len(row.ancestor_last):
        raise ValueError("ancestor branch-nat count must match tree depth")

    if branch_column is None:
        branch_column = 3 * len(row.ancestor_last)
    if ancestor_columns is None:
        ancestor_columns = tuple(3 * index for index in range(len(row.ancestor_last)))

    marker = "❯ " if selected else "  "
    suffix = f"  {display_nats:7.3f} nat"
    structure, structure_width = _structure_prefix(
        row,
        branch_column=branch_column,
        ancestor_columns=ancestor_columns,
        ancestor_branch_nats=ancestor_branch_nats,
        branch_nats=branch_nats,
        branch_reference=branch_reference,
        color=color,
    )

    if token_styles is None:
        fallback = _SELECTED_STYLE if selected else (_FOREST_STYLE if row.forest else "")
        token_styles = (fallback,) * len(tokens)
    if len(token_styles) != len(tokens):
        raise ValueError("token style count must match compact row token count")

    expected_separators = max(0, len(tokens) - 1)
    if separator_nats is None:
        separator_nats = (branch_nats,) * expected_separators
    if len(separator_nats) != expected_separators:
        raise ValueError("separator edge-nat count must match token boundaries")
    if len(preview_separator_nats) != len(preview_tokens):
        raise ValueError("preview edge-nat count must match preview tokens")

    branch_offsets = set(inline_branches)
    label_spans: list[tuple[str, str]] = []
    for index, (token, style) in enumerate(zip(tokens, token_styles, strict=True)):
        if index:
            separator_style = _grayscale(math.exp(-separator_nats[index - 1]))
            separator = " ┬ " if index in branch_offsets else " · "
            label_spans.append((separator, separator_style))
        label_spans.append((describe(token), style))
    if row.forest or (row.open_ended and not preview_tokens):
        if label_spans:
            separator_style = _grayscale(_relative_probability(row.path_nats, branch_reference))
            separator = " ┬ " if len(tokens) in branch_offsets else " · "
            label_spans.append((separator, separator_style))
        label_spans.append(("…", _FOREST_STYLE))

    for token, preview_nats in zip(preview_tokens, preview_separator_nats, strict=True):
        if label_spans:
            separator_style = _grayscale(math.exp(-preview_nats))
            label_spans.append((" · ", separator_style))
        label_spans.append((describe(token), _PREDICTION_STYLE))

    if preview_tokens and not preview_complete:
        separator_style = _grayscale(math.exp(-preview_separator_nats[-1]))
        label_spans.append((" · ", separator_style))
        label_spans.append(("…", _PREDICTION_STYLE))

    marker_width = _cell_width(marker)
    suffix_width = _cell_width(suffix)
    room = max(0, columns - marker_width - structure_width - suffix_width)
    if room == 0 and marker_width + structure_width + suffix_width > columns:
        fallback_ancestor_columns = tuple(3 * index for index in range(len(row.ancestor_last)))
        fallback_branch_column = 3 * len(row.ancestor_last)
        structure, structure_width = _structure_prefix(
            row,
            branch_column=fallback_branch_column,
            ancestor_columns=fallback_ancestor_columns,
            ancestor_branch_nats=ancestor_branch_nats,
            branch_nats=branch_nats,
            branch_reference=branch_reference,
            color=color,
        )
        room = max(0, columns - marker_width - structure_width - suffix_width)

    label = _fit_spans(tuple(label_spans), room, color=color)
    nat_style = _viridis(_relative_probability(display_nats, nat_reference))
    return (
        _paint(marker, _SELECTED_STYLE if selected else "", color=color)
        + structure
        + label
        + _paint(suffix, nat_style, color=color)
    )


def _format_forest_summary(
    direction: str,
    count: int,
    nats: float,
    *,
    columns: int,
    color: bool,
    nat_reference: float | None = None,
) -> str:
    nat_reference = nats if nat_reference is None else nat_reference
    prefix = f"  {direction} … {count} ranks {'above' if direction == '↑' else 'below'}"
    suffix = f"  {nats:7.3f} nat"
    room = max(0, columns - _cell_width(suffix))
    return _paint(_fit(prefix, room), _FOREST_STYLE, color=color) + _paint(
        suffix,
        _viridis(_relative_probability(nats, nat_reference)),
        color=color,
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


def _write_frame(frame: list[str]) -> None:
    """Write one frame without exposing a cleared terminal between redraws."""
    if sys.stdout.isatty():
        output = "\033[H" + "\033[K\n".join(frame) + "\033[J"
    else:
        output = "\n".join(frame) + "\n"
    sys.stdout.write(output)
    sys.stdout.flush()


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
    suggestion: Suggestion | None,
    max_tokens: int,
    lines: int | None,
    debug: bool,
    rewind_depth: int,
    engine_root: NodeId | None = None,
    pending_commands: int = 0,
) -> int:
    """Render one frame from replicated tree state without contacting the engine."""
    columns, terminal_rows = _dimensions()
    color = sys.stdout.isatty()
    rule = "─" * columns

    suggested_tokens = suggestion_tokens(tree, view.node, suggestion)
    suggested_edges = set(suggestion_edges(tree, view.node, suggestion)) if suggestion else set()
    suggested_complete = suggestion_complete(
        tree,
        view.node,
        suggestion,
        max_nats=budget_nats,
        max_tokens=max_tokens,
    )

    frame: list[str] = []
    frame.append(_paint(_line(title, columns), "1", color=color))
    frame.append(_line(rule, columns))
    frame.append(
        _line(
            f"suggestion ≤ {budget_nats:.2f} nat"
            f"  ·  {len(tree.nodes)} nodes"
            f"  ·  {frontier} frontier",
            columns,
        )
    )
    if debug:
        confirmed_root = root if engine_root is None else engine_root
        frame.append(
            _line(
                f"cursor {root}  ·  engine {confirmed_root}"
                f"  ·  rank {view.first_rank}/{view.selected_rank}"
                f"  ·  queued {pending_commands}  ·  rewind {rewind_depth}",
                columns,
            )
        )

    frame.append(_line(rule, columns))
    committed = tree.path(root)
    context_text = _context_text(context, describe, committed, decode_tokens)
    suggestion_text = _sequence_text(describe, suggested_tokens, decode_tokens)
    if suggestion_text and not suggested_complete:
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
            "↑↓ rank  ·  ←/Backspace parent  ·  → child  ·  Space accept",
            columns,
        ),
        _line(
            "Tab/Shift-Tab suggestion  ·  [ ] suggestion limit  ·  d debug  ·  q quit",
            columns,
        ),
    )
    tree_lines = max(2, terminal_rows - len(frame) - len(footer))
    if lines is not None:
        tree_lines = min(tree_lines, max(2, lines))

    distribution = tree[view.node].distribution
    if len(distribution) == 0:
        frame.append(_line("  ∅ terminal", columns))
    elif distribution.revealed == 0:
        frame.append(_line("  … unrevealed", columns))
    else:
        selected = min(view.selected_rank, distribution.revealed - 1)
        start, above, visible = _tree_viewport(
            tree,
            view,
            selected=selected,
            tree_lines=tree_lines,
        )

        view_base_nats = tree.path_nats(view.node, ancestor=root)
        above_nats = (
            view_base_nats + forest_nats(distribution, view.first_rank, start) if above else None
        )
        row_display_nats = [_row_display_nats(tree, root, view, row) for row in visible]
        row_branch_nats = [row.edge_nats for row in visible]
        row_ancestor_branch_nats = _ancestor_branch_nats(visible)
        visible_branch_prefixes = _branch_prefixes(tree, view, visible)

        nat_reference = _minimum_finite(
            [
                *row_display_nats,
                *([above_nats] if above_nats is not None else []),
            ]
        )
        branch_reference = _minimum_finite(
            [
                *row_branch_nats,
                *([forest_nats(distribution, view.first_rank, start)] if above else []),
            ]
        )

        if above_nats is not None:
            frame.append(
                _format_forest_summary(
                    "↑",
                    above,
                    above_nats,
                    columns=columns,
                    color=color,
                    nat_reference=nat_reference,
                )
            )

        for row, display_nats, branch_nats, ancestor_branch_nats in zip(
            visible,
            row_display_nats,
            row_branch_nats,
            row_ancestor_branch_nats,
            strict=True,
        ):
            tokens = row_tokens(tree, row)
            row_selected = not row.forest and row.parent == view.node and row.rank == selected
            ancestor_columns, branch_column = _row_branch_columns(
                tree,
                view,
                row,
                visible_branch_prefixes,
                describe,
            )
            preview_tokens, preview_separator_nats, preview_complete = _row_preview(
                tree,
                row,
                max_tokens=min(max_tokens, 64),
            )
            frame.append(
                _format_tree_row(
                    row,
                    describe,
                    tokens=tokens,
                    selected=row_selected,
                    columns=columns,
                    color=color,
                    display_nats=display_nats,
                    nat_reference=nat_reference,
                    branch_nats=branch_nats,
                    branch_reference=branch_reference,
                    ancestor_branch_nats=ancestor_branch_nats,
                    ancestor_columns=ancestor_columns,
                    branch_column=branch_column,
                    inline_branches=_row_inline_branches(
                        tree,
                        view,
                        row,
                        visible_branch_prefixes,
                    ),
                    separator_nats=_row_separator_nats(tree, row),
                    preview_tokens=preview_tokens,
                    preview_separator_nats=preview_separator_nats,
                    preview_complete=preview_complete,
                    token_styles=_row_token_styles(
                        row,
                        suggested_edges,
                        selected=row_selected,
                    ),
                )
            )

    frame.extend(footer)
    _write_frame(frame)
    return tree_lines


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
        max_tree_bytes: int | None = None,
    ) -> None:
        self.engine = EngineClient(factory, max_tree_bytes=max_tree_bytes)
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
        self._tree_lines = max(2, lines) if lines is not None else 30

        self.view = View(node=self.root)
        self.suggestion: Suggestion | None = None
        self._pending_known: list[tuple[CommandId, NodeId]] = []
        self._pending_unknown: CommandId | None = None
        self.debug = False
        self.quit_requested = False
        self.last_render = 0.0
        self._refresh_suggestion()

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
    def pending(self) -> tuple[CommandId, ...]:
        """Queued causal command ids in execution order."""
        known = tuple(command_id for command_id, _ in self._pending_known)
        if self._pending_unknown is None:
            return known
        return (*known, self._pending_unknown)

    @property
    def navigation_blocked(self) -> bool:
        """Whether the queued tail ends at a child not yet present in the replica."""
        return self._pending_unknown is not None

    def _refresh_suggestion(self) -> bool:
        previous = self.suggestion
        self.suggestion = normalize_suggestion(
            self.tree,
            self.view.node,
            previous,
            max_nats=self.budget_nats,
            max_tokens=self.max_tokens,
        )
        return self.suggestion != previous

    def poll(self) -> bool:
        """Apply engine progress without snapping past queued navigation targets."""
        changed = self.engine.poll() > 0
        completed = 0

        for command_id, target in self._pending_known:
            done = self.engine.take_done(command_id)
            if done is None:
                break
            if done.node != target:
                raise RuntimeError(
                    f"queued navigation diverged: expected node {target}, got {done.node}"
                )
            completed += 1

        if completed:
            del self._pending_known[:completed]

        if not self._pending_known and self._pending_unknown is not None:
            done = self.engine.take_done(self._pending_unknown)
            if done is not None:
                self._pending_unknown = None
                self.view = View(node=done.node)
                changed = True

        if not self.pending and self.view.node != self.root:
            self.view = View(node=self.root)
            changed = True

        return self._refresh_suggestion() or changed

    def render(self) -> None:
        self._refresh_suggestion()
        cursor = self.view.node
        self._tree_lines = _render(
            self.tree,
            cursor,
            self.engine.frontier,
            self.describe,
            self.view,
            title=self.title,
            context=self.context,
            decode_tokens=self.decode_tokens,
            budget_nats=self.budget_nats,
            suggestion=self.suggestion,
            max_tokens=self.max_tokens,
            lines=self.lines,
            debug=self.debug,
            rewind_depth=len(self.tree.path(cursor)),
            engine_root=self.root,
            pending_commands=len(self.pending),
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
            self._refresh_suggestion()
            return True
        if key == "]":
            self.budget_nats += self.budget_step
            self._refresh_suggestion()
            return True
        if key == "TAB":
            self.suggestion = cycle_suggestion(
                self.tree,
                self.view.node,
                self.suggestion,
                1,
                max_nats=self.budget_nats,
                max_tokens=self.max_tokens,
            )
            return True
        if key == "BACKTAB":
            self.suggestion = cycle_suggestion(
                self.tree,
                self.view.node,
                self.suggestion,
                -1,
                max_nats=self.budget_nats,
                max_tokens=self.max_tokens,
            )
            return True
        if key == "UP":
            self.view = move(self.tree, self.view, -1)
            return True
        if key == "DOWN":
            return self._move_down()
        if key == "LEFT" or key in ("\x7f", "\b"):
            return self._rewind()
        if key in (" ", "\r", "\n"):
            return self._accept()
        if key == "RIGHT":
            return self._advance_selected()
        return False

    def _prefetch_viewport(self) -> None:
        distribution = self.tree[self.view.node].distribution
        if distribution.revealed >= len(distribution):
            return

        target = _viewport_reveal_target(
            self.tree,
            self.view,
            tree_lines=self._tree_lines,
        )
        if target <= distribution.revealed:
            return

        self.engine.reveal(
            self.view.node,
            max(target, distribution.revealed + _REVEAL_PAGE),
        )

    def _move_down(self) -> bool:
        distribution = self.tree[self.view.node].distribution
        if distribution.revealed == 0:
            if len(distribution):
                self.engine.reveal(self.view.node, _REVEAL_PAGE)
            return False
        if self.view.selected_rank >= distribution.revealed - 1:
            self._prefetch_viewport()
            return False

        self.view = move(self.tree, self.view, 1)
        self._prefetch_viewport()
        return True

    def _rewind(self) -> bool:
        if self.navigation_blocked:
            return False
        parent = self.tree[self.view.node].parent
        if parent is None:
            return False

        command_id = self.engine.rewind()
        self._pending_known.append((command_id, parent))
        self.view = View(node=parent)
        self._refresh_suggestion()
        return True

    def _accept(self) -> bool:
        if self.pending:
            return False
        self._refresh_suggestion()
        tokens = suggestion_tokens(self.tree, self.view.node, self.suggestion)
        if not tokens:
            return False
        self._pending_unknown = self.engine.advance(tokens)
        return False

    def _advance_selected(self) -> bool:
        if self.navigation_blocked:
            return False
        distribution = self.tree[self.view.node].distribution
        if distribution.revealed == 0:
            return False

        selected = min(self.view.selected_rank, distribution.revealed - 1)
        token = distribution.token(selected)
        child = self.tree.child(self.view.node, selected)
        command_id = self.engine.advance((token,))
        if child is None:
            self._pending_unknown = command_id
            return False

        self._pending_known.append((command_id, child))
        self.view = View(node=child)
        self._refresh_suggestion()
        return True


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
    max_tree_bytes: int | None = None,
) -> None:
    """Run a responsive terminal client around a process-isolated model engine.

    ``budget_nats`` is the cumulative-surprisal limit for the highlighted and
    accepted suggestion; it does not limit autonomous background search.
    ``budget_step`` is how much ``[`` and ``]`` adjust that suggestion limit.

    ``max_tree_bytes`` optionally limits retained authoritative probability-tree
    distribution storage. At the limit autonomous search pauses while explicit
    navigation remains available. ``None`` means unlimited.
    """
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
        max_tree_bytes=max_tree_bytes,
    )
    hard_stop = False

    try:
        with _terminal():
            app.render()
            while True:
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
