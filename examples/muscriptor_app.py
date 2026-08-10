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
import math
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


# Probability is encoded as luminance. A literal linear p -> brightness mapping
# would make almost every child of a large softmax invisible, so use a gentle
# fourth-root transfer: brightness ~ p**0.25 with a small readable floor.
def _probability_intensity(cost_nats: float) -> float:
    if not math.isfinite(cost_nats):
        return 0.16
    probability = math.exp(-max(0.0, cost_nats))
    return 0.16 + 0.84 * probability**0.25


def _scale_rgb(rgb: tuple[int, int, int], intensity: float) -> tuple[int, int, int]:
    return tuple(round(channel * intensity) for channel in rgb)


def _rgb(text: str, rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    return f"\033[38;2;{red};{green};{blue}m{text}\033[0m"


# Filled while building the current view. Compact corridors retain their token
# paths, so this lets the renderer recover every token's local surprisal even
# though several tree nodes have been compressed into one terminal row.
_edge_nats_by_path: dict[tuple[int, ...], float] = {}


def _styled_corridor_label(
    row: compact._DisplayRow,
    *,
    base: tuple[int, int, int],
    row_intensity: float,
    max_chars: int,
    collapsed: bool,
) -> tuple[str, int]:
    """Color each token by its own conditional probability and fit the row."""
    node = row.node
    if node.is_ellipsis or not node.corridor_paths:
        plain = node.label
        if collapsed and not node.is_ellipsis:
            plain = f"▸ {plain}"
        elif len(node.corridor_paths) > 1:
            plain = f"{plain}  ▸"
        if len(plain) > max_chars:
            plain = plain[: max(1, max_chars - 1)] + "…"
        return _rgb(plain, _scale_rgb(base, row_intensity)), len(plain)

    labels = node.label.split(" · ")
    if len(labels) != len(node.corridor_paths):
        # Descriptions currently never contain the separator, but keep a safe
        # fallback if a future tokenizer does.
        plain = node.label
        if len(plain) > max_chars:
            plain = plain[: max(1, max_chars - 1)] + "…"
        return _rgb(plain, _scale_rgb(base, row_intensity)), len(plain)

    rendered: list[str] = []
    used = 0
    if collapsed and max_chars >= 2:
        rendered.append(_rgb("▸ ", _scale_rgb(base, row_intensity)))
        used = 2

    complete = True
    for index, (label, path) in enumerate(zip(labels, node.corridor_paths, strict=True)):
        prefix = "" if index == 0 else " · "
        chunk = prefix + label
        remaining = max_chars - used
        if remaining <= 0:
            complete = False
            break

        cost = _edge_nats_by_path.get(path, node.cost_nats)
        token_rgb = _scale_rgb(base, _probability_intensity(cost))
        if len(chunk) <= remaining:
            rendered.append(_rgb(chunk, token_rgb))
            used += len(chunk)
            continue

        visible = max(1, remaining - 1)
        rendered.append(_rgb(chunk[:visible] + "…", token_rgb))
        used += min(remaining, visible + 1)
        complete = False
        break

    if complete and not collapsed and len(node.corridor_paths) > 1:
        marker = "  ▸"
        if used + len(marker) <= max_chars:
            rendered.append(_rgb(marker, _scale_rgb(base, row_intensity)))
            used += len(marker)

    return "".join(rendered), used


def _probability_render_row(
    row: compact._DisplayRow,
    *,
    selected: bool,
    collapsed: bool,
    width: int,
) -> str:
    """Render local token probability while keeping cumulative corridor cost."""
    guides = "".join("    " if last else "│   " for last in row.ancestor_last)
    connector = "└── " if row.is_last else "├── "
    highlighted = row.node.highlighted
    suggestion_marker = "▶ " if highlighted else "  "
    select_marker = "❯ " if selected else "  "

    suffix = f"{max(0.0, row.node.cost_nats):7.3f} nat"
    room = max(1, width - len(select_marker) - len(suggestion_marker) - len(suffix) - 1)
    prefix = f"{guides}{connector}"
    label_room = max(1, room - len(prefix))

    if not compact.cli.sys.stdout.isatty():
        label = row.node.label
        if collapsed and not row.node.is_ellipsis:
            label = f"▸ {label}"
        elif len(row.node.corridor_paths) > 1:
            label = f"{label}  ▸"
        body = prefix + label
        if len(body) > room:
            body = body[: max(1, room - 1)] + "…"
        return f"{select_marker}{suggestion_marker}{body:<{room}} {suffix}"

    if selected and highlighted:
        base = (255, 120, 255)
    elif selected:
        base = (255, 220, 80)
    elif highlighted:
        base = (80, 220, 255)
    else:
        base = (245, 245, 245)

    row_intensity = _probability_intensity(row.node.cost_nats)
    scaffold = _rgb(prefix, _scale_rgb(base, row_intensity))
    label, label_len = _styled_corridor_label(
        row,
        base=base,
        row_intensity=row_intensity,
        max_chars=label_room,
        collapsed=collapsed,
    )
    body_len = len(prefix) + label_len
    padding = " " * max(0, room - body_len)
    suffix_text = _rgb(suffix, _scale_rgb(base, row_intensity))

    # Selection/navigation markers remain fully saturated so an improbable row
    # never hides the user's location. Token text itself carries probability.
    if selected:
        select_marker = _rgb(select_marker, (255, 220, 80))
    if highlighted:
        suggestion_marker = _rgb(suggestion_marker, (80, 220, 255))
    return f"{select_marker}{suggestion_marker}{scaffold}{label}{padding} {suffix_text}"


compact._render_row = _probability_render_row


# Inspection is deliberately separate from Dijkstra. The search worker decides
# what model prefixes deserve compute; the yellow cursor chooses which already-
# known conditional distribution the human wants to inspect.
_focus_path: tuple[int, ...] = ()
_desired_selection_path: tuple[int, ...] | None = None


def _focused_tree_entries(
    explorer: compact.cli.TokenTreeExplorer,
) -> tuple[compact.cli.TreeEntry, ...]:
    """Render Dijkstra state cheaply, but fully expose the focused distribution.

    Dijkstra materializes children lazily. The UI does not have to: every
    expanded node already stores its complete ranked next-token distribution.
    The focused node therefore exposes every child as a virtual TreeEntry
    without consuming the Dijkstra node budget.

    The ordinary residual calculation has a fast path for the common case
    where the arithmetic interval covers the whole node. Instead of scanning
    the complete vocabulary for every expanded node on every frame, residual
    mass is simply one minus the already-materialized child mass.
    """
    _edge_nats_by_path.clear()
    with explorer._condition:  # noqa: SLF001 - example view over explorer cache
        explorer._raise_worker_error_locked()  # noqa: SLF001
        active_lo, active_hi = explorer._active_interval_locked()  # noqa: SLF001
        nodes = explorer._nodes  # noqa: SLF001

        children: dict[tuple[int, ...], list[object]] = {}
        for node in nodes.values():
            if node.parent is None:
                continue
            if explorer._intersection(node.lo, node.hi, active_lo, active_hi) <= 0:  # noqa: SLF001
                continue
            children.setdefault(node.parent, []).append(node)
        for siblings in children.values():
            siblings.sort(key=lambda node: node.rank)

        entries: list[compact.cli.TreeEntry] = []

        def append_actual(node: object, depth: int) -> None:
            _edge_nats_by_path[node.path] = node.edge_nats
            entries.append(
                compact.cli.TreeEntry(
                    path=node.path,
                    depth=depth,
                    token=node.token,
                    edge_nats=node.edge_nats,
                    path_nats=node.path_nats,
                    expanded=node.expanded,
                )
            )

        def hidden_summary(parent: object) -> tuple[int, float]:
            ranked = parent.ranked
            if ranked is None:
                return 0, 0.0

            # Full coverage is overwhelmingly the common case before an
            # arithmetic digit narrows the interval. This makes the residual
            # O(number of materialized siblings), not O(vocabulary).
            if active_lo <= parent.lo and parent.hi <= active_hi:
                hidden_count = len(ranked.tokens) - len(parent.materialized_ranks)
                materialized_mass = math.fsum(
                    ranked.probabilities[rank] for rank in parent.materialized_ranks
                )
                return hidden_count, max(0.0, 1.0 - materialized_mass)

            # Narrowed arithmetic regions are uncommon and can use the exact
            # core fallback until this optimization moves into TokenTreeExplorer.
            return explorer._hidden_summary_locked(parent)  # noqa: SLF001

        def visit(parent_path: tuple[int, ...], depth: int) -> None:
            parent = nodes.get(parent_path)
            ranked = None if parent is None else parent.ranked

            if parent_path == _focus_path and ranked is not None:
                # Focus is a flat view of one complete conditional
                # distribution. Do not recursively render each child's search
                # subtree here; Right-arrow chooses the one to descend into.
                actual_by_rank = {node.rank: node for node in children.get(parent_path, [])}
                for rank, token in enumerate(ranked.tokens):
                    actual = actual_by_rank.get(rank)
                    if actual is not None:
                        append_actual(actual, depth)
                        continue

                    probability = ranked.probabilities[rank]
                    edge_nats = -math.log(probability) if probability > 0.0 else math.inf
                    path = (*parent_path, token)
                    _edge_nats_by_path[path] = edge_nats
                    entries.append(
                        compact.cli.TreeEntry(
                            path=path,
                            depth=depth,
                            token=token,
                            edge_nats=edge_nats,
                            path_nats=parent.path_nats + edge_nats,
                            expanded=False,
                        )
                    )
                return

            for child in children.get(parent_path, []):
                append_actual(child, depth)
                visit(child.path, depth + 1)

            if parent is not None and parent.expanded:
                hidden_count, hidden_mass = hidden_summary(parent)
                if hidden_count and hidden_mass > 0.0:
                    entries.append(
                        compact.cli.TreeEntry(
                            path=parent_path,
                            depth=depth,
                            token=None,
                            edge_nats=0.0,
                            path_nats=parent.path_nats,
                            expanded=False,
                            is_ellipsis=True,
                            hidden_count=hidden_count,
                            hidden_nats=-math.log(hidden_mass),
                        )
                    )

        visit((), 0)
        return tuple(entries)


# This is intentionally a view-layer override for now. It keeps the generic
# explorer API small while we iterate on what "inspection" should mean.
compact.cli.TokenTreeExplorer.tree_entries = _focused_tree_entries


_base_render_screen = compact.render_screen


def _render_screen_with_focus(*args: object, **kwargs: object):
    """Keep the logical focus node selected when moving up/down the tree."""
    global _desired_selection_path
    if _desired_selection_path is not None:
        kwargs["selected_key"] = (_desired_selection_path, False)
        _desired_selection_path = None
    return _base_render_screen(*args, **kwargs)


compact.render_screen = _render_screen_with_focus
compact.cli.render_screen = _render_screen_with_focus


_base_navigation_override = compact._navigation_override


def _focus_navigation(key: str) -> str:
    """Left/Right move between conditional distributions; Up/Down browse one."""
    global _desired_selection_path, _focus_path

    if key not in {"LEFT", "RIGHT"}:
        return _base_navigation_override(key)

    selected = compact._selected_node
    if key == "LEFT":
        if not _focus_path:
            return "REFRESH"
        old_focus = _focus_path
        _focus_path = old_focus[:-1]
        # The node we just left is a child of the new focus, so keep the
        # yellow cursor on it after returning to the parent distribution.
        _desired_selection_path = old_focus
        return "REFRESH"

    if selected is None or selected.is_ellipsis:
        return "REFRESH"

    entry = selected.representative
    # Virtual children reveal the parent's known logits but do not themselves
    # have a cached next-token distribution yet. Dijkstra can discover them in
    # the background; once expanded, Right descends without any model call in
    # the renderer.
    if not entry.expanded:
        return "REFRESH"

    _focus_path = entry.path
    _desired_selection_path = entry.path
    return "REFRESH"


compact._navigation_override = _focus_navigation


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
