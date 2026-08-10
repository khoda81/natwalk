"""Compact corridor renderer for the interactive MuScriptor natwalk demo.

This is intentionally a view-layer experiment: it reuses the exact navigator,
Dijkstra worker, suggestion traversal, and key handling from ``muscriptor_cli``
and only replaces the tree renderer.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field

import muscriptor_cli as cli


@dataclass
class _CompactNode:
    representative: cli.TreeEntry
    label: str
    cost_nats: float
    highlighted: bool
    is_ellipsis: bool
    corridor_paths: tuple[tuple[int, ...], ...] = ()
    children: list[_CompactNode] = field(default_factory=list)


@dataclass(frozen=True)
class _DisplayRow:
    node: _CompactNode
    depth: int
    ancestor_last: tuple[bool, ...]
    is_last: bool


# Compact corridors are the default view. Right-arrow can temporarily expose
# their constituent token rows without changing the model/search tree itself.
_expanded_corridors: set[tuple[int, ...]] = set()
_selected_node: _CompactNode | None = None
_collapsed_ref: set[tuple[int, ...]] | None = None


def _normal_ellipsis(entry: cli.TreeEntry) -> _CompactNode:
    cost = max(0.0, entry.hidden_nats or 0.0)
    return _CompactNode(
        representative=entry,
        label=f"…  +{entry.hidden_count} hidden",
        cost_nats=cost,
        highlighted=False,
        is_ellipsis=True,
    )


def _deviation_ellipsis(
    sources: list[tuple[cli.TreeEntry, cli.TreeEntry]],
) -> _CompactNode | None:
    """Merge unexpanded alternatives skipped by one compressed corridor.

    Each pair is ``(ellipsis_at_parent, chosen_child)``. Conditioned on having
    entered the first corridor token, the aggregate deviation mass is the sum
    of each hidden residual weighted by the probability of surviving the
    earlier corridor edges.
    """
    if not sources:
        return None

    survival = 1.0
    deviation_mass = 0.0
    for ellipsis, chosen_child in sources:
        hidden_mass = math.exp(-max(0.0, ellipsis.hidden_nats or 0.0))
        deviation_mass += survival * hidden_mass
        survival *= math.exp(-max(0.0, chosen_child.edge_nats))

    deviation_mass = min(max(deviation_mass, 0.0), 1.0)
    if deviation_mass <= 0.0:
        return None

    steps = len(sources)
    label = f"…  deviations along {steps} step{'s' if steps != 1 else ''}"
    return _CompactNode(
        representative=sources[0][0],
        label=label,
        cost_nats=-math.log(deviation_mass),
        highlighted=False,
        is_ellipsis=True,
    )


def _compact_tree(
    entries: list[cli.TreeEntry],
    ctx: cli.MuscriptorContext,
    highlights: set[tuple[int, ...]],
) -> list[_CompactNode]:
    token_children: dict[tuple[int, ...], list[cli.TreeEntry]] = {}
    ellipsis_by_parent: dict[tuple[int, ...], cli.TreeEntry] = {}

    for entry in entries:
        if entry.is_ellipsis:
            ellipsis_by_parent[entry.path] = entry
        else:
            token_children.setdefault(entry.path[:-1], []).append(entry)

    def build_token(start: cli.TreeEntry) -> _CompactNode:
        corridor = [start]
        current = start
        highlighted = start.path in highlights
        deviations: list[tuple[cli.TreeEntry, cli.TreeEntry]] = []

        while current.path not in _expanded_corridors:
            children = token_children.get(current.path, [])
            if len(children) != 1:
                break

            child = children[0]
            # Keep the suggestion boundary visible instead of coloring tokens
            # that lie beyond the current information budget.
            if (child.path in highlights) != highlighted:
                break

            ellipsis = ellipsis_by_parent.get(current.path)
            if ellipsis is not None:
                deviations.append((ellipsis, child))
            corridor.append(child)
            current = child

        label = " · ".join(
            ctx.describe(entry.token) for entry in corridor if entry.token is not None
        )
        node = _CompactNode(
            representative=start,
            label=label,
            cost_nats=sum(max(0.0, entry.edge_nats) for entry in corridor),
            highlighted=highlighted,
            is_ellipsis=False,
            corridor_paths=tuple(entry.path for entry in corridor),
        )

        node.children.extend(build_token(child) for child in token_children.get(current.path, []))
        merged = _deviation_ellipsis(deviations)
        end_ellipsis = ellipsis_by_parent.get(current.path)
        # A terminal residual is usually just "the worker has not expanded
        # beyond this corridor yet". If the corridor already has an exact
        # aggregate deviation row, showing both is redundant visual noise.
        if end_ellipsis is not None and merged is None:
            node.children.append(_normal_ellipsis(end_ellipsis))
        if merged is not None:
            node.children.append(merged)
        return node

    roots = [build_token(child) for child in token_children.get((), [])]
    root_ellipsis = ellipsis_by_parent.get(())
    if root_ellipsis is not None:
        roots.append(_normal_ellipsis(root_ellipsis))
    return roots


def _flatten(
    nodes: list[_CompactNode],
    collapsed: set[tuple[int, ...]],
) -> list[_DisplayRow]:
    """Flatten after corridor compression, so one compact corridor is one row."""
    rows: list[_DisplayRow] = []

    def visit(
        siblings: list[_CompactNode],
        depth: int,
        ancestor_last: tuple[bool, ...],
    ) -> None:
        for index, node in enumerate(siblings):
            is_last = index == len(siblings) - 1
            rows.append(
                _DisplayRow(
                    node=node,
                    depth=depth,
                    ancestor_last=ancestor_last,
                    is_last=is_last,
                )
            )
            if node.representative.path not in collapsed:
                visit(node.children, depth + 1, (*ancestor_last, is_last))

    visit(nodes, 0, ())
    return rows


def _styled(line: str, *, selected: bool, highlighted: bool, ellipsis: bool) -> str:
    if not cli.sys.stdout.isatty():
        return line
    if selected and highlighted:
        return f"\033[1;35m{line}\033[0m"  # magenta: both cursors overlap
    if selected:
        return f"\033[1;33m{line}\033[0m"  # yellow: inspection cursor
    if highlighted:
        return f"\033[1;36m{line}\033[0m"  # cyan: active completion
    if ellipsis:
        return cli.dim(line)
    return line


def _render_row(
    row: _DisplayRow,
    *,
    selected: bool,
    collapsed: bool,
    width: int,
) -> str:
    guides = "".join("    " if last else "│   " for last in row.ancestor_last)
    connector = "└── " if row.is_last else "├── "
    marker = "▶ " if row.node.highlighted else "  "
    select_marker = "❯ " if selected else "  "

    label = row.node.label
    if collapsed and not row.node.is_ellipsis:
        label = f"▸ {label}"
    elif len(row.node.corridor_paths) > 1:
        label = f"{label}  ▸"

    body = f"{guides}{connector}{label}"
    suffix = f"{max(0.0, row.node.cost_nats):7.3f} nat"
    room = max(1, width - len(select_marker) - len(marker) - len(suffix) - 1)
    if len(body) > room:
        body = body[: max(1, room - 1)] + "…"
    line = f"{select_marker}{marker}{body:<{room}} {suffix}"
    return _styled(
        line,
        selected=selected,
        highlighted=row.node.highlighted,
        ellipsis=row.node.is_ellipsis,
    )


def _tree_line_budget(requested: int, *, debug: bool, narrowed: bool) -> int:
    """Fit tree rows into the current terminal; explicit values are maxima."""
    terminal_lines = shutil.get_terminal_size((100, 30)).lines
    # Header/footer is 16 rows in the common case, plus optional debug/range
    # notices. Corridor compression has already happened before this budget is
    # applied, so a many-token corridor costs exactly one screen row.
    reserved = 16 + int(debug) + int(narrowed)
    available = max(4, terminal_lines - reserved)
    if requested <= 0:
        return available
    return min(max(4, requested), available)


def render_screen(
    explorer: cli.TokenTreeExplorer,
    ctx: cli.MuscriptorContext,
    *,
    budget_nats: float,
    suggestion_index: int,
    selected_key: tuple[tuple[int, ...], bool] | None,
    collapsed: set[tuple[int, ...]],
    tree_lines: int,
    max_tokens: int,
    debug: bool,
) -> tuple[
    list[cli.TreeEntry],
    tuple[tuple[int, ...], bool] | None,
    tuple[cli.GreedySuggestion, ...],
    int,
]:
    global _collapsed_ref, _selected_node

    cli.clear()
    state = explorer.snapshot
    stats = explorer.stats()
    suggestions = cli.cached_budget_completions(
        explorer,
        max_nats=budget_nats,
        max_tokens=max_tokens,
    )
    if suggestions:
        suggestion_index %= len(suggestions)
        suggestion = suggestions[suggestion_index]
    else:
        suggestion_index = 0
        suggestion = explorer.cached_greedy_suggestion(
            max_bits=budget_nats / math.log(2),
            max_tokens=max_tokens,
        )

    raw_entries = explorer.tree_entries()
    highlights = cli.suggestion_prefixes(suggestion)
    rows = _flatten(_compact_tree(raw_entries, ctx, highlights), collapsed)
    entries = [row.node.representative for row in rows]

    if selected_key is None and entries:
        selected_key = cli.entry_key(entries[0])
    selected_index = 0
    if selected_key is not None and entries:
        for i, entry in enumerate(entries):
            if cli.entry_key(entry) == selected_key:
                selected_index = i
                break
        else:
            selected_key = cli.entry_key(entries[0])

    _collapsed_ref = collapsed
    _selected_node = rows[selected_index].node if rows else None

    print("natwalk · MuScriptor · compact")
    print("=" * 78)
    print(
        f"Budget: {budget_nats:.2f} nat ({budget_nats / math.log(2):.2f} bit)    "
        f"binary action: {explorer.nats_per_action:.3f} nat = {explorer.bits_per_action:.2f} bit"
    )
    print()
    print("Committed:")
    print(cli.fmt_tokens(ctx, state.prefix, limit=18))

    if suggestion.tokens:
        tail = " · …" if not suggestion.complete else ""
        ordinal = f" {suggestion_index + 1}/{len(suggestions)}" if suggestions else ""
        print()
        print(
            f"Suggestion{ordinal} [{suggestion.nats:.3f}/{budget_nats:.2f} nat]"
            f"{'  ⟳' if not suggestion.complete else ''}:"
        )
        suggestion_text = cli.fmt_tokens(ctx, suggestion.tokens, limit=18) + tail
        if cli.sys.stdout.isatty():
            print(f"\033[1;36m{suggestion_text}\033[0m")
        else:
            print(suggestion_text)
    elif suggestion.complete and suggestion.next_token_nats is not None:
        print()
        print(
            f"Suggestion: —   next greedy token costs {suggestion.next_token_nats:.3f} nat "
            f"(budget {budget_nats:.2f})"
        )
    else:
        print()
        print("Suggestion: ⟳ computing…")

    print()
    status = (
        f"tree: {stats.nodes} nodes · {stats.expanded} model expansions · "
        f"{stats.frontier} frontier"
        f"{' · ⟳' if stats.computing else ''}"
        f"{' · memory cap' if stats.saturated else ''}"
    )
    print(status)

    if debug:
        print(
            f"range=[{state.lo:.9f}, {state.hi:.9f}) · "
            f"binary={explorer.supplied_nats:.3f} nat · "
            f"path={state.path_surprisal:.3f} nat · undo={state.undo_depth}"
        )

    print()
    width = max(60, shutil.get_terminal_size((100, 30)).columns)
    if not rows:
        print("  ⟳ expanding root…")
    else:
        line_count = _tree_line_budget(
            tree_lines,
            debug=debug,
            narrowed=state.lo != 0.0 or state.hi != 1.0,
        )
        half = line_count // 2
        start = max(0, selected_index - half)
        start = min(start, max(0, len(rows) - line_count))
        end = min(len(rows), start + line_count)

        if start > 0:
            print(cli.dim(f"  ↑ … {start} rows above"))
        for row in rows[start:end]:
            key = cli.entry_key(row.node.representative)
            print(
                _render_row(
                    row,
                    selected=key == selected_key,
                    collapsed=row.node.representative.path in collapsed,
                    width=width,
                )
            )
        if end < len(rows):
            print(cli.dim(f"  ↓ … {len(rows) - end} rows below"))

    print()
    if explorer.choices == 2:
        print("0 / 1: choose exact half    Space: accept suggestion    Backspace: undo")
    else:
        print(f"0–{explorer.choices - 1}: choose exact bucket    Space: accept suggestion")
    print("Tab/Shift-Tab: completion (cyan)    [/]: ± budget nat    d: debug    q: quit")
    print("↑/↓ inspect (yellow)    ← collapse/recompact    → reopen/decompress")
    if state.lo != 0.0 or state.hi != 1.0:
        print(cli.dim("Space is an explicit accept: it resets the narrowed arithmetic range."))

    return entries, selected_key, suggestions, suggestion_index


def _navigation_override(key: str) -> str:
    """Give compact corridors meaningful left/right semantics."""
    node = _selected_node
    if key not in {"LEFT", "RIGHT"} or node is None:
        return key
    if node.is_ellipsis:
        return "REFRESH"

    path = node.representative.path
    if key == "RIGHT":
        # First reopen a genuinely collapsed subtree using the main CLI logic.
        if _collapsed_ref is not None and path in _collapsed_ref:
            return key
        # Otherwise expose every token represented by this compact corridor.
        if len(node.corridor_paths) > 1:
            _expanded_corridors.update(node.corridor_paths)
            return "REFRESH"
        return "REFRESH"

    # Left on an explicitly expanded region means "recompact from here".
    expanded_below = {
        expanded
        for expanded in _expanded_corridors
        if len(expanded) >= len(path) and expanded[: len(path)] == path
    }
    if expanded_below:
        _expanded_corridors.difference_update(expanded_below)
        return "REFRESH"

    # Otherwise preserve the main CLI's ordinary subtree-collapse behavior.
    return key


def read_key(timeout: float = 0.20) -> str | None:
    """Read one terminal key without mixing fd polling and TextIO buffering."""
    if not cli.sys.stdin.isatty():
        return input("> ").strip()[:1]

    import termios
    import tty

    fd = cli.sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = cli.select.select([fd], [], [], timeout)
        if not ready:
            return None

        first = os.read(fd, 1)
        if first == b"\x03":
            raise KeyboardInterrupt
        if first == b"\t":
            return "TAB"
        if first != b"\x1b":
            return first.decode("utf-8", errors="ignore")

        seq = bytearray(first)
        deadline = cli.time.monotonic() + 0.05
        while len(seq) < 16:
            remaining = deadline - cli.time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = cli.select.select([fd], [], [], remaining)
            if not ready:
                break
            chunk = os.read(fd, 16 - len(seq))
            if not chunk:
                break
            seq.extend(chunk)
            if seq[-1] == ord("~") or seq[-1] in b"ABCDZ":
                break
        text = bytes(seq).decode("ascii", errors="ignore")
        return _navigation_override(cli._decode_escape_sequence(text))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


cli.render_screen = render_screen
cli.read_key = read_key


if __name__ == "__main__":
    cli.main()
