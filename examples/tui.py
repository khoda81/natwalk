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
from collections.abc import Callable
from contextlib import contextmanager

from natwalk.model import Cursor
from natwalk.navigation import Navigation
from natwalk.query import Suggestion, completions, greedy
from natwalk.session import Session
from natwalk.view import View, enter, move, parent
from natwalk.worker import SearchWorker

DescribeToken = Callable[[int], str]


def _tokens(describe: DescribeToken, tokens: tuple[int, ...], limit: int = 18) -> str:
    if not tokens:
        return "∅"
    shown = tokens[-limit:]
    prefix = "… · " if len(tokens) > limit else ""
    return prefix + " · ".join(describe(token) for token in shown)


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


def _read_key(timeout: float = 0.20) -> str | None:
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
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            break
        chunk = os.read(fd, 16 - len(sequence))
        if not chunk:
            break
        sequence.extend(chunk)
        if sequence[-1] == ord("~") or sequence[-1] in b"ABCDZ":
            break

    return {
        b"\x1b[A": "UP",
        b"\x1b[B": "DOWN",
        b"\x1b[C": "RIGHT",
        b"\x1b[D": "LEFT",
        b"\x1bOA": "UP",
        b"\x1bOB": "DOWN",
        b"\x1bOC": "RIGHT",
        b"\x1bOD": "LEFT",
        b"\x1b[Z": "BACKTAB",
    }.get(bytes(sequence), "ESC")


def _render(
    session: Session,
    navigation: Navigation,
    describe: DescribeToken,
    view: View,
    *,
    title: str,
    budget_nats: float,
    completion_index: int,
    max_tokens: int,
    lines: int,
    debug: bool,
) -> tuple[tuple[Suggestion, ...], int]:
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")

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
        suggestion = greedy(tree, session.root, max_nats=budget_nats, max_tokens=max_tokens)

    print(title)
    print("=" * 78)
    print(
        f"budget={budget_nats:.2f} nat ({budget_nats / math.log(2):.2f} bit) · "
        f"action={math.log2(navigation.choices):.2f} bit"
    )
    print()
    print("Committed:")
    print(_tokens(describe, tree.path(session.root)))
    print()
    if suggestion.tokens:
        suffix = " · …" if not suggestion.complete else ""
        print(f"Suggestion [{suggestion.nats:.3f}/{budget_nats:.2f} nat]:")
        print(_tokens(describe, suggestion.tokens) + suffix)
    elif suggestion.next_nats is not None:
        print(f"Suggestion: — next token costs {suggestion.next_nats:.3f} nat")
    else:
        print("Suggestion: ⟳ search has not reached the greedy tail yet")

    known = sum(node.distribution is not None for node in tree.nodes)
    print()
    print(
        f"search: {len(tree.nodes)} nodes · {known} distributions · "
        f"{len(session.search.frontier)} frontier"
    )
    if debug:
        print(
            f"interval=[{navigation.state.lo:.9f}, {navigation.state.hi:.9f}) · "
            f"actions={navigation.state.actions} · undo={navigation.undo_depth}"
        )

    distribution = session.inspect(view.node)
    focus = tree.path_from(session.root, view.node)
    print(f"focus: {_tokens(describe, focus, limit=10)}")
    print()

    width = max(60, shutil.get_terminal_size((100, 30)).columns)
    count = len(distribution)
    if count == 0:
        print("  ∅ terminal")
    else:
        selected = min(view.selected_rank, count - 1)
        line_count = max(4, lines)
        start = max(view.first_rank, selected - line_count // 2)
        start = min(start, max(view.first_rank, count - line_count))
        end = min(count, start + line_count)
        if start > view.first_rank:
            print(f"  ↑ … {start - view.first_rank} ranks above")
        for rank in range(start, end):
            token = distribution.tokens[rank]
            marker = "❯ " if rank == selected else "  "
            label = describe(token)
            cost = distribution.nats(rank)
            room = max(8, width - 18)
            if len(label) > room:
                label = label[: room - 1] + "…"
            print(f"{marker}{rank:5d}  {label:<{room}} {cost:7.3f} nat")
        if end < count:
            print(f"  ↓ … {count - end} ranks below")

    print()
    print("↑/↓ rank · ← parent · → inspect child · 0..9 choose · Space accept · Backspace undo")
    print("Tab/Shift-Tab completion · [/]: budget · d debug · q quit")
    sys.stdout.flush()
    return suggestions, completion_index


def run_tui(
    cursor: Cursor,
    describe: DescribeToken,
    *,
    title: str,
    choices: int = 2,
    max_tokens: int = 256,
    budget_nats: float = 1.5,
    budget_step: float = 0.25,
    lines: int = 16,
) -> None:
    session = Session(cursor)
    navigation = Navigation(session, choices=choices)
    view = View(node=session.root)
    completion_index = 0
    debug = False

    try:
        with _terminal(), SearchWorker(session.search) as worker:
            while True:
                with worker.access():
                    suggestions, completion_index = _render(
                        session,
                        navigation,
                        describe,
                        view,
                        title=title,
                        budget_nats=budget_nats,
                        completion_index=completion_index,
                        max_tokens=max_tokens,
                        lines=lines,
                        debug=debug,
                    )
                    terminal = not session.distribution().tokens
                if terminal:
                    break

                key = _read_key()
                if key is None:
                    continue
                if key.lower() == "q":
                    break
                if key.lower() == "d":
                    debug = not debug
                    continue
                if key == "[":
                    budget_nats = max(0.0, budget_nats - budget_step)
                    completion_index = 0
                    continue
                if key == "]":
                    budget_nats += budget_step
                    completion_index = 0
                    continue
                if key == "TAB" and suggestions:
                    completion_index = (completion_index + 1) % len(suggestions)
                    continue
                if key == "BACKTAB" and suggestions:
                    completion_index = (completion_index - 1) % len(suggestions)
                    continue

                with worker.access():
                    if key in ("\x7f", "\b"):
                        if navigation.undo():
                            view = View(node=session.root)
                            completion_index = 0
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
                            navigation.accept(suggestion.tokens)
                            view = View(node=session.root)
                            completion_index = 0
                    elif key.isdigit() and int(key) < choices:
                        navigation.choose(int(key))
                        view = View(node=session.root)
                        completion_index = 0
                    elif key == "UP":
                        view = move(session.tree, view, -1)
                    elif key == "DOWN":
                        view = move(session.tree, view, 1)
                    elif key == "LEFT" and view.node != session.root:
                        view = parent(session.tree, view)
                    elif key == "RIGHT":
                        distribution = session.inspect(view.node)
                        if distribution.tokens:
                            view = enter(session.tree, view)
                            session.inspect(view.node)
    except KeyboardInterrupt:
        pass
