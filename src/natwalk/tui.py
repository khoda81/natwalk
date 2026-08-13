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
from functools import cache

from .engine import CommandId, CursorFactory, EngineClient
from .query import (
    Suggestion,
    cycle_suggestion,
    normalize_suggestion,
    suggestion_complete,
    suggestion_edges,
    suggestion_tokens,
)
from .tree import Edge, NodeId, Tree
from .view import BranchRole, CompactRow, View, partition_rows, row_tokens

type DescribeToken = Callable[[int], str]
type DecodeTokens = Callable[[tuple[int, ...]], str]
type _StyledSpan = tuple[str, str]

_KEY_POLL_SECONDS = 0.05
_REDRAW_SECONDS = 0.25
_REVEAL_PAGE = 128
_SUGGESTION_STYLE = "1;38;5;45"
_FOREST_STYLE = "2;38;5;244"
_PREDICTION_STYLE = "38;5;241"
_CONTINUATION_CONNECTOR = "┬ "
_BRANCH_CONNECTOR = "├─"
_LAST_BRANCH_CONNECTOR = "└─"
_SEQUENCE_SEPARATOR = " · "
_BRANCH_SEPARATOR = " ┬ "
_TREE_GUTTER = "  "
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
    b"\x1b[H": "HOME",
    b"\x1bOH": "HOME",
    b"\x1b[1~": "HOME",
    b"\x1b[7~": "HOME",
    b"\x1b[F": "END",
    b"\x1bOF": "END",
    b"\x1b[4~": "END",
    b"\x1b[8~": "END",
    b"\x1b[5~": "PAGE_UP",
    b"\x1b[6~": "PAGE_DOWN",
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


_CONNECTOR_WIDTH = _cell_width(_CONTINUATION_CONNECTOR)
assert all(
    _cell_width(connector) == _CONNECTOR_WIDTH
    for connector in (_BRANCH_CONNECTOR, _LAST_BRANCH_CONNECTOR)
), "leading radix connectors must have one shared width"
_SEPARATOR_WIDTH = _cell_width(_SEQUENCE_SEPARATOR)
assert _cell_width(_BRANCH_SEPARATOR) == _SEPARATOR_WIDTH, (
    "radix separators must have one shared width"
)
_BRANCH_JUNCTION_OFFSET = _cell_width(_BRANCH_SEPARATOR[: _BRANCH_SEPARATOR.index("┬")])


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


def _row_separator_nats(tree: Tree, row: CompactRow) -> tuple[float, ...]:
    """Return local surprisal for each collapsed token boundary."""
    return tuple(tree[edge.parent].distribution.nats(edge.rank) for edge in row.edges[1:])


@dataclass(frozen=True, slots=True)
class _Preview:
    """Read-only greedy edges shown beyond one measured partition event."""

    edges: tuple[Edge, ...]
    complete: bool


def _row_preview(
    tree: Tree,
    row: CompactRow,
    *,
    max_tokens: int = 64,
) -> _Preview:
    """Return read-only best-known context beyond one measured row event."""
    if max_tokens <= 0:
        return _Preview((), False)

    edges: list[Edge] = []

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
            return _Preview((), False)
        edge = Edge(node, row.forest_start)
        edges.append(edge)
        child = tree.child(edge.parent, edge.rank)
        if child is None:
            return _Preview(tuple(edges), False)
        node = child
    else:
        node = row.child
        if node is None:
            return _Preview((), False)

    while len(edges) < max_tokens:
        distribution = tree[node].distribution
        if len(distribution) == 0:
            return _Preview(tuple(edges), True)
        if distribution.revealed == 0:
            return _Preview(tuple(edges), False)

        edge = Edge(node, 0)
        edges.append(edge)
        child = tree.child(edge.parent, edge.rank)
        if child is None:
            return _Preview(tuple(edges), False)
        node = child

    return _Preview(tuple(edges), len(tree[node].distribution) == 0)


def _unrevealed_forest_node(tree: Tree, row: CompactRow) -> NodeId | None:
    """Return the distribution node when a forest row is only a replica boundary."""
    if not row.forest:
        return None

    node = row.parent
    if row.edges:
        edge = row.edges[-1]
        child = tree.child(edge.parent, edge.rank)
        assert child is not None
        node = child

    distribution = tree[node].distribution
    if row.forest_start < distribution.revealed or distribution.revealed >= len(distribution):
        return None
    return node


def _viewport_reveal_demands(
    tree: Tree,
    rows: tuple[CompactRow, ...],
    *,
    tree_lines: int,
) -> tuple[tuple[NodeId, int], ...]:
    """Report replica prefixes needed before their transport boundary becomes visible."""
    targets: dict[NodeId, int] = {}
    reveal_span = max(_REVEAL_PAGE, 2 * tree_lines)
    for row in rows:
        node = _unrevealed_forest_node(tree, row)
        if node is None:
            continue
        distribution = tree[node].distribution
        stop = min(len(distribution), distribution.revealed + reveal_span)
        targets[node] = max(targets.get(node, 0), stop)
    return tuple(targets.items())


def _tree_viewport(
    tree: Tree,
    view: View,
    *,
    tree_lines: int,
) -> tuple[tuple[CompactRow, ...], tuple[tuple[NodeId, int], ...]]:
    """Render the visible sibling suffix and prefetch its near future."""
    visible = partition_rows(tree, view, row_limit=tree_lines)
    probe = partition_rows(tree, view, row_limit=2 * tree_lines)
    reveal_demands = _viewport_reveal_demands(
        tree,
        (*visible, *probe),
        tree_lines=tree_lines,
    )

    # The unrevealed suffix is real probability mass but its current split point
    # is transport state, not a semantic UI event. Keep ordinary aggregate forests.
    visible = tuple(row for row in visible if _unrevealed_forest_node(tree, row) is None)
    return visible, reveal_demands


def _wrap_spans(
    spans: tuple[_StyledSpan, ...],
    width: int,
    *,
    color: bool,
) -> tuple[str, ...]:
    """Wrap styled plain-text spans without counting ANSI escapes as cells."""
    if width <= 0:
        return ("",)

    lines: list[list[_StyledSpan]] = [[]]
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
    spans: tuple[_StyledSpan, ...],
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
    out: list[_StyledSpan] = []
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
    return _SEQUENCE_SEPARATOR.join(describe(token) for token in tokens)


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
    return context + _SEQUENCE_SEPARATOR + suffix


def _context_spans(
    context_text: str,
    suggestion_text: str,
    decode: DecodeTokens | None,
) -> tuple[_StyledSpan, ...]:
    """Compose context and highlighted completion without corrupting exact text."""
    spans: list[_StyledSpan] = []
    if context_text:
        spans.append((context_text, ""))
    elif not suggestion_text:
        spans.append(("∅", ""))

    if context_text and suggestion_text and decode is None:
        spans.append((_SEQUENCE_SEPARATOR, ""))
    if suggestion_text:
        spans.append((suggestion_text, _SUGGESTION_STYLE))
    return tuple(spans)


def _row_token_styles(
    row: CompactRow,
    suggestion: set[Suggestion],
) -> tuple[str, ...]:
    """Color compact-row edges from structural suggestion state only."""
    styles: list[str] = []
    for edge in row.edges:
        if edge in suggestion:
            style = _SUGGESTION_STYLE
        elif row.forest:
            style = _FOREST_STYLE
        else:
            style = ""
        styles.append(style)
    return tuple(styles)


def _spans_width(spans: tuple[_StyledSpan, ...]) -> int:
    return sum(_cell_width(text) for text, _style in spans)


def _paint_spans(spans: tuple[_StyledSpan, ...], *, color: bool) -> str:
    return "".join(_paint(text, style, color=color) for text, style in spans)


def _styled_cell_spans(cells: list[str], styles: list[str]) -> tuple[_StyledSpan, ...]:
    assert len(cells) == len(styles)
    spans: list[_StyledSpan] = []
    for cell, style in zip(cells, styles, strict=True):
        if spans and spans[-1][1] == style:
            text, _ = spans[-1]
            spans[-1] = (text + cell, style)
        else:
            spans.append((cell, style))
    return tuple(spans)


def _branch_text(row: CompactRow) -> str:
    """Return the connector implied by the row's semantic branch role."""
    if row.branch_role is BranchRole.CONTINUATION:
        return _CONTINUATION_CONNECTOR
    return _LAST_BRANCH_CONNECTOR if row.is_last else _BRANCH_CONNECTOR


@dataclass(frozen=True, slots=True)
class _TreeRowLayout:
    """One fully laid-out tree row, before terminal clipping and ANSI painting."""

    structure: tuple[_StyledSpan, ...]
    label: tuple[_StyledSpan, ...]
    suffix: _StyledSpan


def _format_tree_row(
    row: _TreeRowLayout,
    *,
    columns: int,
    color: bool,
) -> str:
    """Paint one already-laid-out tree row into a terminal width."""
    suffix_text, suffix_style = row.suffix
    gutter_width = _cell_width(_TREE_GUTTER)
    structure_width = _spans_width(row.structure)
    suffix_width = _cell_width(suffix_text)

    # Coordinates belong to layout, not clipping. If the branch is outside the
    # drawable viewport, say so instead of inventing alternate indentation.
    if gutter_width + structure_width + suffix_width > columns:
        notice = "… branch off-screen →"
        notice_room = columns - gutter_width - suffix_width
        if notice_room > 0:
            return (
                _TREE_GUTTER
                + _fit_spans(((notice, _FOREST_STYLE),), notice_room, color=color)
                + _paint(suffix_text, suffix_style, color=color)
            )
        return _fit_spans(
            (
                (_TREE_GUTTER, ""),
                (notice, _FOREST_STYLE),
            ),
            columns,
            color=color,
        )

    room = max(0, columns - gutter_width - structure_width - suffix_width)
    return (
        _TREE_GUTTER
        + _paint_spans(row.structure, color=color)
        + _fit_spans(row.label, room, color=color)
        + _paint(suffix_text, suffix_style, color=color)
    )


class _TreeRenderer:
    """Turn semantic probability-partition rows into terminal tree rows."""

    def __init__(
        self,
        tree: Tree,
        root: NodeId,
        view: View,
        rows: tuple[CompactRow, ...],
        describe: DescribeToken,
        *,
        suggestion: Suggestion | None,
        max_preview_tokens: int,
    ) -> None:
        self.tree = tree
        self.view = view
        self.rows = rows
        self.describe = describe
        self.max_preview_tokens = max_preview_tokens
        self.suggested_edges = (
            set(suggestion_edges(tree, view.node, suggestion)) if suggestion else set()
        )
        self.branch_nodes = {row.parent for row in rows if row.branch_role is BranchRole.SIBLING}
        self.view_base_nats = tree.path_nats(view.node, ancestor=root)

        tree_ref = tree
        view_node = view.node
        describe_ref = describe

        @cache
        def node_geometry(node: NodeId) -> tuple[int, float]:
            """Return branch column and view-relative path surprisal for one node."""
            if node == view_node:
                return 0, 0.0

            item = tree_ref[node]
            parent = item.parent
            assert parent is not None
            parent_column, parent_nats = node_geometry(parent)
            token_width = _cell_width(describe_ref(tree_ref.token(node)))
            column = (
                _CONNECTOR_WIDTH + token_width + _BRANCH_JUNCTION_OFFSET
                if parent == view_node
                else parent_column + _SEPARATOR_WIDTH + token_width
            )
            return column, parent_nats + tree_ref.edge_nats(node)

        self._node_geometry = node_geometry

        display_nats = [self.view_base_nats + row.path_nats for row in rows]
        branch_nats = [row.edge_nats for row in rows]
        self.nat_reference = _minimum_finite(display_nats)
        self.branch_reference = _minimum_finite(branch_nats)

    def render(self, *, columns: int, color: bool) -> tuple[str, ...]:
        return tuple(
            _format_tree_row(self._layout_row(row), columns=columns, color=color)
            for row in self.rows
        )

    def _ancestor_children(self, node: NodeId) -> tuple[NodeId, ...]:
        """Return child endpoints of visible ancestor branch edges."""
        children: list[NodeId] = []
        current = node
        while current != self.view.node:
            item = self.tree[current]
            parent = item.parent
            assert parent is not None
            if parent in self.branch_nodes:
                children.append(current)
            current = parent
        children.reverse()
        return tuple(children)

    def _structure_spans(self, row: CompactRow) -> tuple[_StyledSpan, ...]:
        """Lay out one row's structural connectors directly from the tree."""
        branch_column, _ = self._node_geometry(row.parent)
        branch = _branch_text(row)
        assert all(_cell_width(char) == 1 for char in branch)

        width = branch_column + _cell_width(branch)
        cells = [" "] * width
        styles = [""] * width

        for child in self._ancestor_children(row.parent):
            item = self.tree[child]
            parent = item.parent
            assert parent is not None
            if item.rank == len(self.tree[parent].distribution) - 1:
                continue

            column, _ = self._node_geometry(parent)
            _, nats = self._node_geometry(child)
            assert 0 <= column < width
            cells[column] = "│"
            styles[column] = _grayscale(_relative_probability(nats, self.branch_reference))

        glyph_style = _grayscale(_relative_probability(row.edge_nats, self.branch_reference))
        for offset, char in enumerate(branch):
            cells[branch_column + offset] = char
            styles[branch_column + offset] = glyph_style
        return _styled_cell_spans(cells, styles)

    def _inline_branches(self, row: CompactRow) -> set[int]:
        """Return token offsets whose endpoint is another visible branch point."""
        branches = {
            index
            for index, edge in enumerate(row.edges[1:], start=1)
            if edge.parent in self.branch_nodes
        }
        if not row.edges:
            return branches

        if row.forest:
            last = row.edges[-1]
            endpoint = self.tree.child(last.parent, last.rank)
            assert endpoint is not None
        else:
            endpoint = row.child

        if endpoint is not None and endpoint in self.branch_nodes:
            branches.add(len(row.edges))
        return branches

    def _layout_row(self, row: CompactRow) -> _TreeRowLayout:
        display_nats = self.view_base_nats + row.path_nats
        return _TreeRowLayout(
            structure=self._structure_spans(row),
            label=self._label_spans(row),
            suffix=(
                f"  {display_nats:7.3f} nat",
                _viridis(_relative_probability(display_nats, self.nat_reference)),
            ),
        )

    def _label_spans(self, row: CompactRow) -> tuple[_StyledSpan, ...]:
        tokens = row_tokens(self.tree, row)
        token_styles = _row_token_styles(row, self.suggested_edges)
        separator_nats = _row_separator_nats(self.tree, row)
        preview = _row_preview(
            self.tree,
            row,
            max_tokens=self.max_preview_tokens,
        )
        inline_branches = self._inline_branches(row)

        assert len(token_styles) == len(tokens)
        assert len(separator_nats) == max(0, len(tokens) - 1)

        spans: list[_StyledSpan] = []
        if tokens:
            spans.append((self.describe(tokens[0]), token_styles[0]))

        for index, (token, style, token_nats) in enumerate(
            zip(tokens[1:], token_styles[1:], separator_nats, strict=True),
            start=1,
        ):
            separator = _BRANCH_SEPARATOR if index in inline_branches else _SEQUENCE_SEPARATOR
            separator_style = _grayscale(math.exp(-token_nats))
            spans.append((separator, separator_style))
            spans.append((self.describe(token), style))

        for edge in preview.edges:
            distribution = self.tree[edge.parent].distribution
            preview_nats = distribution.nats(edge.rank)
            separator_style = _grayscale(math.exp(-preview_nats))
            token_text = self.describe(distribution.token(edge.rank))
            spans.append((_SEQUENCE_SEPARATOR, separator_style))
            spans.append((token_text, _PREDICTION_STYLE))

        separator = _BRANCH_SEPARATOR if len(tokens) in inline_branches else _SEQUENCE_SEPARATOR
        separator_style = _grayscale(_relative_probability(row.path_nats, self.branch_reference))

        if preview.edges and not preview.complete:
            last = preview.edges[-1]
            preview_nats = self.tree[last.parent].distribution.nats(last.rank)
            separator_style = _grayscale(math.exp(-preview_nats))
            spans.append((_SEQUENCE_SEPARATOR, separator_style))
            spans.append(("…", _PREDICTION_STYLE))
        elif not preview.edges and (row.forest or row.open_ended):
            if tokens:
                spans.append((separator, separator_style))
            spans.append(("…", _FOREST_STYLE))

        return tuple(spans)


@contextmanager
def _terminal():
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sys.stdout.write("\033[?1049h\033[H\033[?25l\033[?1007h")
    sys.stdout.flush()
    tty.setcbreak(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[?1007l\033[?25h\033[?1049l")
        sys.stdout.flush()


def _decode_escape(sequence: bytes) -> str | None:
    """Decode one complete terminal escape sequence."""
    key = _ESCAPE_KEYS.get(sequence)
    if key is not None:
        return key

    if sequence.startswith(b"\x1b[<") and sequence[-1:] in {b"M", b"m"}:
        try:
            button = int(sequence[3:].split(b";", 1)[0])
        except ValueError:
            return "ESC"

        button &= ~0x1C
        if button == 64:
            return "UP"
        if button == 65:
            return "DOWN"
        return "MOUSE"

    return None


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
    while len(sequence) < 64:
        key = _decode_escape(bytes(sequence))
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
        if sequence.startswith(b"\x1b[<") and sequence[-1] in (ord("M"), ord("m")):
            break

    return _decode_escape(bytes(sequence)) or "ESC"


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
) -> tuple[tuple[NodeId, int], ...]:
    """Render one frame and report replica reveal demand without contacting the engine."""
    columns, terminal_rows = _dimensions()
    color = sys.stdout.isatty()
    rule = "─" * columns

    suggested_tokens = suggestion_tokens(tree, view.node, suggestion)
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
                f"  ·  rank {view.first_rank}"
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
            "↑↓/wheel scroll  ·  ←/Backspace parent  ·  → child  ·  Space accept",
            columns,
        ),
        _line(
            "Home/End top/bottom  ·  PgUp/PgDn page  ·  Tab/Shift-Tab suggestion",
            columns,
        ),
        _line(
            "[ ] suggestion limit  ·  d debug  ·  q quit",
            columns,
        ),
    )
    tree_lines = max(2, terminal_rows - len(frame) - len(footer))
    if lines is not None:
        tree_lines = min(tree_lines, max(2, lines))

    reveal_demands: tuple[tuple[NodeId, int], ...] = ()
    distribution = tree[view.node].distribution
    if len(distribution) == 0:
        frame.append(_line("  ∅ terminal", columns))
    elif distribution.revealed == 0:
        frame.append(_line("  … unrevealed", columns))
    else:
        visible, reveal_demands = _tree_viewport(
            tree,
            view,
            tree_lines=tree_lines,
        )
        renderer = _TreeRenderer(
            tree,
            root,
            view,
            visible,
            describe,
            suggestion=suggestion,
            max_preview_tokens=min(max_tokens, 64),
        )
        frame.extend(renderer.render(columns=columns, color=color))

    frame.extend(footer)
    _write_frame(frame)
    return reveal_demands


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
        reveal_demands = _render(
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
        for node, stop in reveal_demands:
            self.engine.reveal(node, stop)
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
            return self._scroll(-1)
        if key == "DOWN":
            return self._scroll(1)
        if key == "HOME":
            return self._scroll_to(0)
        if key == "END":
            return self._scroll_to_end()
        if key == "PAGE_UP":
            return self._scroll_page(-1)
        if key == "PAGE_DOWN":
            return self._scroll_page(1)
        if key == "LEFT" or key in ("\x7f", "\b"):
            return self._rewind()
        if key in (" ", "\r", "\n"):
            return self._accept()
        if key == "RIGHT":
            return self._advance_visible()
        return False

    def _scroll_to(self, rank: int) -> bool:
        distribution = self.tree[self.view.node].distribution
        if distribution.revealed == 0:
            if len(distribution):
                self.engine.reveal(self.view.node, _REVEAL_PAGE)
            return False

        rank = min(max(0, rank), distribution.revealed - 1)
        if rank == self.view.first_rank:
            return False

        self.view = View(node=self.view.node, first_rank=rank)
        return True

    def _scroll(self, delta: int) -> bool:
        distribution = self.tree[self.view.node].distribution
        if delta > 0 and self.view.first_rank >= distribution.revealed - 1:
            if distribution.revealed < len(distribution):
                self.engine.reveal(self.view.node, distribution.revealed + _REVEAL_PAGE)
            return False
        return self._scroll_to(self.view.first_rank + delta)

    def _scroll_to_end(self) -> bool:
        distribution = self.tree[self.view.node].distribution
        if distribution.revealed == 0:
            if len(distribution):
                self.engine.reveal(self.view.node, _REVEAL_PAGE)
            return False
        return self._scroll_to(distribution.revealed - 1)

    def _scroll_page(self, direction: int) -> bool:
        _, terminal_rows = _dimensions()
        page = self.lines if self.lines is not None else max(1, terminal_rows // 2)
        target = self.view.first_rank + direction * max(1, page)

        distribution = self.tree[self.view.node].distribution
        if (
            direction > 0
            and target >= distribution.revealed
            and distribution.revealed < len(distribution)
        ):
            stop = min(
                len(distribution),
                max(distribution.revealed + _REVEAL_PAGE, target + 1),
            )
            self.engine.reveal(self.view.node, stop)

        return self._scroll_to(target)

    def _rewind(self) -> bool:
        if self.navigation_blocked:
            return False

        node = self.tree[self.view.node]
        if node.parent is None:
            return False

        command_id = self.engine.rewind()
        self._pending_known.append((command_id, node.parent))
        self.view = View(node=node.parent, first_rank=node.rank)
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

    def _advance_visible(self) -> bool:
        if self.navigation_blocked:
            return False
        distribution = self.tree[self.view.node].distribution
        if distribution.revealed == 0:
            return False

        rank = min(self.view.first_rank, distribution.revealed - 1)
        token = distribution.token(rank)
        child = self.tree.child(self.view.node, rank)
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
