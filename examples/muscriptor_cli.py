"""Interactive MuScriptor demo for natwalk.

The model tree is searched in information distance (Dijkstra / uniform-cost
search). Tree browsing never commits probability mass. Binary/K-ary digits
narrow the exact arithmetic interval; Space explicitly accepts a highlighted
continuation inside an adjustable nat budget.
"""

from __future__ import annotations

import argparse
import copy
import math
import select
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from muscriptor import TranscriptionModel
from muscriptor.modules.streaming import increment_steps, init_states

from natwalk import (
    GreedySuggestion,
    Navigator,
    TokenTreeExplorer,
    TreeEntry,
    accept_completion,
    cached_budget_completions,
)

VALID_CARD = 1393
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 5
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE


def midi_name(pitch: int) -> str:
    names = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def clone_model_state(model_state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Expensive compatibility fallback used only by Cursor.clone()."""
    out: dict[str, dict[str, Any]] = {}
    for module_name, state in model_state.items():
        cloned: dict[str, Any] = {}
        for key, value in state.items():
            cloned[key] = value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        out[module_name] = cloned
    return out


def snapshot_control_state(
    model_state: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Snapshot tiny streaming controls while deliberately sharing KV storage."""
    out: dict[str, dict[str, Any]] = {}
    for module_name, state in model_state.items():
        controls: dict[str, Any] = {}
        for key, value in state.items():
            if key == "cache":
                continue
            controls[key] = (
                value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            )
        out[module_name] = controls
    return out


def restore_control_state(
    model_state: dict[str, dict[str, Any]],
    snapshot: dict[str, dict[str, Any]],
) -> None:
    for module_name, controls in snapshot.items():
        state = model_state[module_name]
        for key, saved in controls.items():
            current = state[key]
            if isinstance(current, torch.Tensor):
                current.copy_(saved)
            else:
                state[key] = copy.deepcopy(saved)


class MuscriptorContext:
    def __init__(
        self,
        tm: TranscriptionModel,
        audio_path: str | Path,
        *,
        chunk_index: int,
        max_tokens: int,
    ) -> None:
        self.tm = tm
        self.lm = tm._model
        self.tokenizer = tm._tokenizer
        self.device = tm._device
        self.max_tokens = max_tokens

        wav = tm._load_wav(audio_path, None)
        start = chunk_index * CHUNK_SAMPLES
        if start >= wav.shape[-1]:
            raise ValueError(
                f"chunk {chunk_index} starts after audio ends ({wav.shape[-1] / SAMPLE_RATE:.2f}s)"
            )
        chunk = wav[:, start : start + CHUNK_SAMPLES]
        if chunk.shape[-1] < CHUNK_SAMPLES:
            chunk = F.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[-1]))

        conditions = tm._build_conditions(chunk, None)
        prepared = self.lm.condition_provider.tokenize(conditions)
        self.cfg_conditions = self.lm.condition_provider(prepared)
        self.prepend_length = sum(cond.shape[1] for cond, _ in self.cfg_conditions.values())

    def new_cursor(self) -> MuscriptorCursor:
        return MuscriptorCursor(self)

    def describe(self, token: int) -> str:
        event = self.tokenizer._vocab[token]
        kind, value = event.type, event.value
        if kind in {"PAD", "EOS", "UNK"}:
            return kind
        if kind == "shift":
            return f"t={value / self.tokenizer.frame_rate:.2f}s"
        if kind == "pitch":
            return midi_name(value)
        if kind == "velocity":
            return "note_on" if value else "note_off"
        if kind == "tie":
            return "tie"
        if kind == "program":
            return self.tm._instrument_for_program(value)
        if kind == "drum":
            return f"drum({midi_name(value)})"
        return f"{kind}({value})"


class MuscriptorCursor:
    """Complete-distribution cursor over one MuScriptor audio chunk."""

    def __init__(self, ctx: MuscriptorContext) -> None:
        self.ctx = ctx
        self.lm = ctx.lm
        self.device = ctx.device
        self.prefix: tuple[int, ...] = ()
        self.ended = False

        cache_len = ctx.prepend_length + ctx.max_tokens + 1
        self.model_state = init_states(self.lm, batch_size=1, sequence_length=cache_len)
        self.input = torch.tensor(
            [[self.lm.initial_token_id]], device=self.device, dtype=torch.long
        )
        self.first_step = True
        self._probs: torch.Tensor | None = None

    def clone(self) -> MuscriptorCursor:
        """Full KV clone fallback; natwalk normally uses checkpoint/restore."""
        self.predict()
        other = object.__new__(MuscriptorCursor)
        other.ctx = self.ctx
        other.lm = self.lm
        other.device = self.device
        other.prefix = self.prefix
        other.ended = self.ended
        other.model_state = clone_model_state(self.model_state)
        other.input = self.input.clone()
        other.first_step = self.first_step
        other._probs = self._probs
        return other

    def checkpoint(self) -> object:
        """Cheap branch point: no KV tensors are copied."""
        self.predict()
        return (
            self.prefix,
            self.ended,
            self.input.clone(),
            self.first_step,
            self._probs,
            snapshot_control_state(self.model_state),
        )

    def restore(self, checkpoint: object) -> None:
        prefix, ended, input_, first_step, probs, controls = checkpoint
        restore_control_state(self.model_state, controls)
        self.prefix = prefix
        self.ended = ended
        self.input = input_
        self.first_step = first_step
        self._probs = probs

    @torch.inference_mode()
    def predict(self) -> Sequence[float]:
        if self.ended:
            p = torch.zeros(VALID_CARD, dtype=torch.float64)
            p[self.ctx.tokenizer.eos_id] = 1.0
            return p
        if self._probs is not None:
            return self._probs

        with self.lm.autocast:
            logits = self.lm._compute_logits(
                self.input,
                self.ctx.cfg_conditions,
                self.model_state,
                first_step=self.first_step,
                cfg_coef=1.0,
                forbidden_tokens=None,
            )[0]

        logits = logits.float()
        logits[VALID_CARD:] = -torch.inf
        probs = torch.softmax(logits, dim=-1)[:VALID_CARD].double().cpu()
        probs /= probs.sum()
        self._probs = probs
        return probs

    def observe(self, token: int) -> None:
        if self.ended:
            raise RuntimeError("cannot observe after EOS")
        if not 0 <= token < VALID_CARD:
            raise ValueError(token)

        probs = self.predict()
        if float(probs[token]) <= 0.0:
            raise ValueError(f"token {token} has zero probability")

        increment = self.input.shape[-1]
        if self.first_step:
            increment += self.ctx.prepend_length
        increment_steps(self.lm.transformer, self.model_state, increment=increment)

        self.prefix = (*self.prefix, token)
        self.ended = token == self.ctx.tokenizer.eos_id
        self.input = torch.tensor([[token]], device=self.device, dtype=torch.long)
        self.first_step = False
        self._probs = None


def fmt_tokens(ctx: MuscriptorContext, tokens: Sequence[int], limit: int = 16) -> str:
    if not tokens:
        return "∅"
    if len(tokens) > limit:
        shown = ["…", *(ctx.describe(token) for token in tokens[-limit:])]
    else:
        shown = [ctx.describe(token) for token in tokens]
    return " · ".join(shown)


def clear() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def bold(text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[1m{text}\033[0m"


def dim(text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[2m{text}\033[0m"


def collapse_entries(
    entries: Sequence[TreeEntry],
    collapsed: set[tuple[int, ...]],
) -> list[TreeEntry]:
    out: list[TreeEntry] = []
    for entry in entries:
        if any(
            len(entry.path) > len(parent) and entry.path[: len(parent)] == parent
            for parent in collapsed
        ):
            continue
        out.append(entry)
    return out


def entry_key(entry: TreeEntry) -> tuple[tuple[int, ...], bool]:
    return entry.path, entry.is_ellipsis


def entry_parent(entry: TreeEntry) -> tuple[int, ...]:
    return entry.path if entry.is_ellipsis else entry.path[:-1]


def last_children(
    entries: Sequence[TreeEntry],
) -> dict[tuple[int, ...], tuple[tuple[int, ...], bool]]:
    last: dict[tuple[int, ...], tuple[tuple[int, ...], bool]] = {}
    for entry in entries:
        last[entry_parent(entry)] = entry_key(entry)
    return last


def suggestion_prefixes(suggestion: GreedySuggestion) -> set[tuple[int, ...]]:
    return {tuple(suggestion.tokens[:i]) for i in range(1, len(suggestion.tokens) + 1)}


def render_tree_line(
    entry: TreeEntry,
    ctx: MuscriptorContext,
    *,
    selected: bool,
    highlighted: bool,
    collapsed: bool,
    last_by_parent: dict[tuple[int, ...], tuple[tuple[int, ...], bool]],
    width: int,
) -> str:
    guides: list[str] = []
    for depth in range(entry.depth):
        ancestor_path = entry.path[: depth + 1]
        ancestor_key = (ancestor_path, False)
        ancestor_parent = ancestor_path[:-1]
        guides.append("    " if last_by_parent.get(ancestor_parent) == ancestor_key else "│   ")

    parent = entry_parent(entry)
    connector = "└── " if last_by_parent.get(parent) == entry_key(entry) else "├── "
    marker = "▶ " if highlighted else "  "
    select_marker = "❯ " if selected else "  "

    if entry.is_ellipsis:
        label = f"…  +{entry.hidden_count} hidden"
    else:
        label = ctx.describe(entry.token) if entry.token is not None else "."
        if collapsed:
            label = f"▸ {label}"

    body = f"{''.join(guides)}{connector}{label}"
    cost = max(0.0, entry.hidden_nats if entry.is_ellipsis else entry.edge_nats)
    suffix = f"{cost:7.3f} nat"
    room = max(1, width - len(select_marker) - len(marker) - len(suffix) - 1)
    if len(body) > room:
        body = body[: max(1, room - 1)] + "…"
    line = f"{select_marker}{marker}{body:<{room}} {suffix}"
    if selected or highlighted:
        line = bold(line)
    elif entry.is_ellipsis:
        line = dim(line)
    return line


def render_screen(
    explorer: TokenTreeExplorer,
    ctx: MuscriptorContext,
    *,
    budget_nats: float,
    suggestion_index: int,
    selected_key: tuple[tuple[int, ...], bool] | None,
    collapsed: set[tuple[int, ...]],
    tree_lines: int,
    max_tokens: int,
    debug: bool,
) -> tuple[
    list[TreeEntry],
    tuple[tuple[int, ...], bool] | None,
    tuple[GreedySuggestion, ...],
    int,
]:
    clear()
    state = explorer.snapshot
    stats = explorer.stats()
    suggestions = cached_budget_completions(
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

    all_entries = collapse_entries(explorer.tree_entries(), collapsed)
    last_by_parent = last_children(all_entries)
    highlights = suggestion_prefixes(suggestion)

    if selected_key is None and all_entries:
        selected_key = entry_key(all_entries[0])
    selected_index = 0
    if selected_key is not None and all_entries:
        for i, entry in enumerate(all_entries):
            if entry_key(entry) == selected_key:
                selected_index = i
                break
        else:
            selected_key = entry_key(all_entries[0])

    print("natwalk · MuScriptor")
    print("=" * 78)
    print(
        f"Budget: {budget_nats:.2f} nat ({budget_nats / math.log(2):.2f} bit)    "
        f"binary action: {explorer.nats_per_action:.3f} nat = {explorer.bits_per_action:.2f} bit"
    )
    print()
    print("Committed:")
    print(fmt_tokens(ctx, state.prefix, limit=18))

    if suggestion.tokens:
        tail = " · …" if not suggestion.complete else ""
        ordinal = f" {suggestion_index + 1}/{len(suggestions)}" if suggestions else ""
        print()
        print(
            f"Suggestion{ordinal} [{suggestion.nats:.3f}/{budget_nats:.2f} nat]"
            f"{'  ⟳' if not suggestion.complete else ''}:"
        )
        print(bold(fmt_tokens(ctx, suggestion.tokens, limit=18) + tail))
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
    if not all_entries:
        print("  ⟳ expanding root…")
    else:
        line_count = max(4, tree_lines)
        half = line_count // 2
        start = max(0, selected_index - half)
        start = min(start, max(0, len(all_entries) - line_count))
        end = min(len(all_entries), start + line_count)

        if start > 0:
            print(dim(f"  ↑ … {start} rows above"))
        for entry in all_entries[start:end]:
            key = entry_key(entry)
            print(
                render_tree_line(
                    entry,
                    ctx,
                    selected=key == selected_key,
                    highlighted=(not entry.is_ellipsis and entry.path in highlights),
                    collapsed=entry.path in collapsed,
                    last_by_parent=last_by_parent,
                    width=width,
                )
            )
        if end < len(all_entries):
            print(dim(f"  ↓ … {len(all_entries) - end} rows below"))

    print()
    if explorer.choices == 2:
        print("0 / 1: choose exact half    Space: accept suggestion    Backspace: undo")
    else:
        print(f"0–{explorer.choices - 1}: choose exact bucket    Space: accept suggestion")
    print("Tab/Shift-Tab: suggestion    [/]: ± budget nat    d: debug    q: quit")
    print("↑/↓ browse tree    ←/→ collapse/expand")
    if state.lo != 0.0 or state.hi != 1.0:
        print(
            dim("Space is an explicit accept: it resets the currently narrowed arithmetic range.")
        )

    return all_entries, selected_key, suggestions, suggestion_index


def _decode_escape_sequence(seq: str) -> str:
    if seq == "\x1b[Z":
        return "BACKTAB"
    if not (seq.startswith("\x1b[") or seq.startswith("\x1bO")):
        return "ESC"
    return {
        "A": "UP",
        "B": "DOWN",
        "C": "RIGHT",
        "D": "LEFT",
        "Z": "BACKTAB",
    }.get(seq[-1], "ESC")


def read_key(timeout: float = 0.20) -> str | None:
    if not sys.stdin.isatty():
        return input("> ").strip()[:1]

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\t":
            return "TAB"
        if ch != "\x1b":
            return ch

        seq = ch
        deadline = time.monotonic() + 0.05
        while len(seq) < 16:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([sys.stdin], [], [], remaining)
            if not ready:
                break
            next_char = sys.stdin.read(1)
            seq += next_char
            if next_char == "~":
                break
            if len(seq) >= 3 and next_char in "ABCDZ":
                break
        return _decode_escape_sequence(seq)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--model", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=0, help="5-second chunk index")
    ap.add_argument("--choices", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--tree-nodes",
        type=int,
        default=10_000,
        help="materialized token-tree node cap; 0 means unlimited",
    )
    ap.add_argument("--tree-lines", type=int, default=16)
    ap.add_argument("--budget-nats", type=float, default=1.5)
    ap.add_argument(
        "--budget-step",
        type=float,
        default=0.25,
        help="linear nat increment used by [ and ]",
    )
    args = ap.parse_args()

    if args.budget_nats < 0:
        ap.error("--budget-nats must be >= 0")
    if args.budget_step <= 0:
        ap.error("--budget-step must be > 0")

    print(f"Loading MuScriptor {args.model} on {args.device} …", file=sys.stderr)
    tm = TranscriptionModel.load_model(args.model, device=args.device)
    print(
        f"Encoding chunk {args.chunk} "
        f"({args.chunk * CHUNK_SECONDS:.1f}–{(args.chunk + 1) * CHUNK_SECONDS:.1f}s) …",
        file=sys.stderr,
    )
    ctx = MuscriptorContext(tm, args.audio, chunk_index=args.chunk, max_tokens=args.max_tokens)
    nav = Navigator(ctx.new_cursor(), choices=args.choices)

    budget_nats = args.budget_nats
    suggestion_index = 0
    selected_key: tuple[tuple[int, ...], bool] | None = None
    collapsed: set[tuple[int, ...]] = set()
    debug = False

    try:
        with TokenTreeExplorer(nav, max_nodes=args.tree_nodes) as explorer:
            while True:
                entries, selected_key, suggestions, suggestion_index = render_screen(
                    explorer,
                    ctx,
                    budget_nats=budget_nats,
                    suggestion_index=suggestion_index,
                    selected_key=selected_key,
                    collapsed=collapsed,
                    tree_lines=args.tree_lines,
                    max_tokens=args.max_tokens,
                    debug=debug,
                )
                if explorer.snapshot.ended:
                    break

                key = read_key()
                if key is None:
                    continue
                if key.lower() == "q":
                    break
                if key.lower() == "d":
                    debug = not debug
                    continue

                if key == "[":
                    budget_nats = max(0.0, budget_nats - args.budget_step)
                    suggestion_index = 0
                    continue
                if key == "]":
                    budget_nats += args.budget_step
                    suggestion_index = 0
                    continue

                if key == "TAB":
                    if suggestions:
                        suggestion_index = (suggestion_index + 1) % len(suggestions)
                    continue
                if key == "BACKTAB":
                    if suggestions:
                        suggestion_index = (suggestion_index - 1) % len(suggestions)
                    continue

                if key in ("\x7f", "\b"):
                    if explorer.undo():
                        suggestion_index = 0
                        selected_key = None
                        collapsed.clear()
                    continue

                if key in (" ", "\r", "\n"):
                    if suggestions:
                        suggestion = suggestions[suggestion_index % len(suggestions)]
                    else:
                        suggestion = explorer.cached_greedy_suggestion(
                            max_bits=budget_nats / math.log(2),
                            max_tokens=args.max_tokens,
                        )
                    if suggestion.tokens:
                        accept_completion(explorer, suggestion.tokens)
                        suggestion_index = 0
                        selected_key = None
                        collapsed.clear()
                    continue

                if key.isdigit():
                    bucket = int(key)
                    if 0 <= bucket < explorer.choices:
                        explorer.choose(bucket)
                        suggestion_index = 0
                        selected_key = None
                        collapsed.clear()
                    continue

                if not entries:
                    continue

                try:
                    index = next(
                        i for i, entry in enumerate(entries) if entry_key(entry) == selected_key
                    )
                except StopIteration:
                    index = 0

                if key == "UP":
                    index = max(0, index - 1)
                    selected_key = entry_key(entries[index])
                elif key == "DOWN":
                    index = min(len(entries) - 1, index + 1)
                    selected_key = entry_key(entries[index])
                elif key in {"LEFT", "RIGHT"}:
                    entry = entries[index]
                    if entry.is_ellipsis:
                        continue
                    if key == "LEFT":
                        collapsed.add(entry.path)
                    else:
                        collapsed.discard(entry.path)

    except KeyboardInterrupt:
        pass
    finally:
        state = nav.snapshot()
        print()
        print(
            f"final: binary_actions={state.actions}, binary={nav.supplied_nats:.6f} nat "
            f"({nav.supplied_bits:.6f} bit), committed_tokens={len(state.prefix)}, "
            f"path_surprisal={state.path_surprisal:.6f} nat"
        )


if __name__ == "__main__":
    main()
