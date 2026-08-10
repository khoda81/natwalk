# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "llama-cpp-python>=0.3.16",
#   "numpy>=1.26",
# ]
# ///

"""Interactive natwalk probability explorer for local GGUF language models.

The default model path is resolved from Ollama's local model store, so an
existing ``ollama pull`` does not require another model download. Inference is
done directly through llama.cpp because natwalk requires the complete next-token
distribution and cheap rewindable model state.

Example::

    uv run examples/llm_app.py \
      --model ministral-3:14b \
      "The most surprising thing about information theory is"

This example intentionally treats the argument as a raw continuation prefix.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import select
import shutil
import subprocess
import sys
import termios
import tty
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from llama_cpp import Llama

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from natwalk import Navigator, RankedDistribution, TokenTreeExplorer, accept_completion  # noqa: E402


RESET = "\033[0m"
WHITE = (245, 245, 245)
CYAN = (80, 220, 255)
YELLOW = (255, 220, 80)
MAGENTA = (255, 120, 255)
DIM = (110, 110, 110)


def resolve_ollama_gguf(model: str) -> Path:
    """Return the GGUF blob behind an Ollama model without copying it."""
    try:
        result = subprocess.run(
            ["ollama", "show", "--modelfile", model],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ollama is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"ollama could not resolve {model!r}: {message}") from exc

    for line in result.stdout.splitlines():
        if not line.startswith("FROM "):
            continue
        source = line.removeprefix("FROM ").strip().strip('"')
        path = Path(source).expanduser()
        if path.is_file():
            return path
        raise RuntimeError(
            f"Ollama resolved {model!r} to {source!r}, but that is not a local GGUF file"
        )
    raise RuntimeError(f"ollama show --modelfile {model!r} did not contain a FROM path")


class LlamaCursor:
    """Rewindable complete-distribution cursor over one llama.cpp context.

    Checkpoints are deliberately tiny: a prefix, token count, and immutable
    probability-array reference. Branch evaluation writes KV positions after
    the checkpoint; restoring ``n_tokens`` makes llama.cpp discard/overwrite
    that suffix on the next ``eval`` while preserving the committed prefix.
    """

    def __init__(self, llm: Llama, prompt_tokens: Sequence[int]) -> None:
        if not prompt_tokens:
            raise ValueError("prompt tokenization produced no tokens")
        self.llm = llm
        self.prompt_tokens = tuple(int(token) for token in prompt_tokens)
        self.prefix: tuple[int, ...] = ()
        self.ended = False
        self._probs: np.ndarray | None = None
        self.llm.reset()
        self.llm.eval(self.prompt_tokens)

    def clone(self) -> LlamaCursor:
        raise RuntimeError("LlamaCursor is rewindable and intentionally does not clone its KV cache")

    def checkpoint(self) -> object:
        return self.prefix, self.ended, self.llm.n_tokens, self._probs

    def restore(self, checkpoint: object) -> None:
        prefix, ended, n_tokens, probs = checkpoint  # type: ignore[misc]
        self.prefix = prefix
        self.ended = ended
        self.llm.n_tokens = n_tokens
        self._probs = probs

    def predict(self) -> Sequence[float]:
        if self.ended:
            probs = np.zeros(self.llm.n_vocab(), dtype=np.float64)
            eos = self.llm.token_eos()
            if eos >= 0:
                probs[eos] = 1.0
            return probs
        if self._probs is not None:
            return self._probs

        # With logits_all=False llama.cpp asks for logits only at the final
        # position. The high-level wrapper does not copy that row into its large
        # score matrix, but the native context still exposes it directly.
        logits_ptr = self.llm._ctx.get_logits()  # noqa: SLF001
        logits = np.ctypeslib.as_array(logits_ptr, shape=(self.llm.n_vocab(),)).astype(
            np.float64,
            copy=True,
        )
        finite = np.isfinite(logits)
        if not finite.any():
            raise RuntimeError("llama.cpp returned no finite next-token logits")
        logits[~finite] = -np.inf
        logits -= np.max(logits)
        probs = np.exp(logits)
        total = float(probs.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("llama.cpp logits produced an invalid probability distribution")
        probs /= total
        self._probs = probs
        return probs

    def observe(self, token: int) -> None:
        if self.ended:
            raise RuntimeError("cannot observe after end of generation")
        if not 0 <= token < self.llm.n_vocab():
            raise ValueError(token)
        if float(self.predict()[token]) <= 0.0:
            raise ValueError(f"token {token} has zero probability")

        self.llm.eval([int(token)])
        self.prefix = (*self.prefix, int(token))
        self._probs = None
        self.ended = token == self.llm.token_eos()


class TokenDisplay:
    def __init__(self, llm: Llama) -> None:
        self.llm = llm
        self._cache: dict[int, str] = {}

    def __call__(self, token: int) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached

        piece = self.llm.detokenize([token], special=True).decode("utf-8", errors="replace")
        if not piece:
            piece = f"<token:{token}>"
        piece = (
            piece.replace("\\", "\\\\")
            .replace("\n", "↵")
            .replace("\r", "↩")
            .replace("\t", "⇥")
            .replace(" ", "␠")
        )
        self._cache[token] = piece
        return piece


def rgb(text: str, color: tuple[int, int, int]) -> str:
    red, green, blue = color
    return f"\033[38;2;{red};{green};{blue}m{text}{RESET}"


def scale_rgb(color: tuple[int, int, int], intensity: float) -> tuple[int, int, int]:
    return tuple(round(channel * intensity) for channel in color)


def probability_intensity(cost_nats: float) -> float:
    if not math.isfinite(cost_nats):
        return 0.14
    probability = math.exp(-max(0.0, cost_nats))
    return 0.14 + 0.86 * probability**0.25


def fmt_tokens(describe: TokenDisplay, tokens: Sequence[int], limit: int = 18) -> str:
    if not tokens:
        return "∅"
    shown = list(tokens)
    prefix = ""
    if len(shown) > limit:
        shown = shown[-limit:]
        prefix = "… · "
    return prefix + " · ".join(describe(token) for token in shown)


def truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return "…"
    return text[: width - 1] + "…"


@contextlib.contextmanager
def terminal_session() -> Iterator[None]:
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


_pending_input = bytearray()


def read_byte(fd: int, timeout: float) -> bytes | None:
    if _pending_input:
        return bytes((_pending_input.pop(0),))
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    data = os.read(fd, 1)
    return data or None


def decode_escape(sequence: bytes) -> str:
    if sequence == b"\x1b[Z":
        return "BACKTAB"
    if not (sequence.startswith(b"\x1b[") or sequence.startswith(b"\x1bO")):
        return "ESC"
    return {
        ord("A"): "UP",
        ord("B"): "DOWN",
        ord("C"): "RIGHT",
        ord("D"): "LEFT",
        ord("Z"): "BACKTAB",
    }.get(sequence[-1], "ESC")


def read_escape(fd: int) -> str:
    sequence = bytearray(b"\x1b")
    second = read_byte(fd, 0.05)
    if second is None:
        return "ESC"
    sequence.extend(second)
    if second not in {b"[", b"O"}:
        _pending_input[:0] = second
        return "ESC"

    for _ in range(30):
        byte = read_byte(fd, 0.05)
        if byte is None:
            break
        sequence.extend(byte)
        if 0x40 <= byte[0] <= 0x7E:
            break
    return decode_escape(bytes(sequence))


def read_key(timeout: float = 0.20) -> str | None:
    if not sys.stdin.isatty():
        return input("> ").strip()[:1]
    fd = sys.stdin.fileno()
    first = read_byte(fd, timeout)
    if first is None:
        return None
    if first == b"\x03":
        raise KeyboardInterrupt
    if first == b"\t":
        return "TAB"
    if first == b"\x1b":
        return read_escape(fd)
    return first.decode("utf-8", errors="ignore")


def clear() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")


def focus_distribution(
    explorer: TokenTreeExplorer,
    navigator: Navigator,
    path: tuple[int, ...],
    cache: dict[tuple[int, ...], RankedDistribution],
) -> RankedDistribution | None:
    cached = cache.get(path)
    if cached is not None:
        return cached

    # Serialize with Dijkstra because both intentionally share one llama.cpp KV
    # context. Inspection may briefly pause search, but never corrupts its state.
    with explorer._compute_lock:  # noqa: SLF001
        with navigator.temporary_cursor() as cursor:
            for token in path:
                if cursor.ended:
                    return None
                cursor.observe(token)
            if cursor.ended:
                return None
            ranked = navigator.rank(cursor)
    cache[path] = ranked
    return ranked


def suggestion_costs(
    explorer: TokenTreeExplorer,
    tokens: Sequence[int],
) -> list[float]:
    costs: list[float] = []
    path: tuple[int, ...] = ()
    with explorer._condition:  # noqa: SLF001
        for token in tokens:
            node = explorer._nodes.get(path)  # noqa: SLF001
            if node is None or node.ranked is None:
                break
            try:
                rank = node.ranked.tokens.index(token)
            except ValueError:
                break
            probability = node.ranked.probabilities[rank]
            costs.append(-math.log(probability))
            path = (*path, token)
    return costs


def render_suggestion(
    describe: TokenDisplay,
    tokens: Sequence[int],
    costs: Sequence[float],
) -> str:
    parts: list[str] = []
    for index, token in enumerate(tokens):
        text = describe(token)
        cost = costs[index] if index < len(costs) else 0.0
        intensity = probability_intensity(cost)
        parts.append(rgb(text, scale_rgb(CYAN, intensity)))
    return rgb(" · ", CYAN).join(parts)


def render_distribution(
    ranked: RankedDistribution,
    describe: TokenDisplay,
    *,
    selected_rank: int,
    highlighted_token: int | None,
    lines: int,
    width: int,
) -> None:
    count = len(ranked.tokens)
    if count == 0:
        print("  ∅")
        return

    selected_rank = min(max(selected_rank, 0), count - 1)
    half = lines // 2
    start = max(0, selected_rank - half)
    start = min(start, max(0, count - lines))
    end = min(count, start + lines)

    if start:
        print(rgb(f"  ↑ … {start} ranks above", DIM))

    suffix_width = 14
    for rank in range(start, end):
        token = ranked.tokens[rank]
        probability = ranked.probabilities[rank]
        cost = -math.log(probability) if probability > 0.0 else math.inf
        selected = rank == selected_rank
        highlighted = token == highlighted_token

        select_marker = "❯ " if selected else "  "
        suggestion_marker = "▶ " if highlighted else "  "
        connector = "└── " if rank == count - 1 else "├── "
        label = describe(token)
        body_room = max(8, width - 4 - suffix_width)
        body = truncate(f"{connector}{label}", body_room)
        suffix = f"{cost:7.3f} nat"

        if selected and highlighted:
            base = MAGENTA
        elif selected:
            base = YELLOW
        elif highlighted:
            base = CYAN
        else:
            base = WHITE
        intensity = probability_intensity(cost)
        payload = f"{body:<{body_room}} {suffix}"

        if sys.stdout.isatty():
            if selected:
                select_marker = rgb(select_marker, YELLOW)
            if highlighted:
                suggestion_marker = rgb(suggestion_marker, CYAN)
            payload = rgb(payload, scale_rgb(base, intensity))
        print(f"{select_marker}{suggestion_marker}{payload}")

    if end < count:
        print(rgb(f"  ↓ … {count - end} ranks below", DIM))


def tree_line_budget(debug: bool) -> int:
    terminal_lines = shutil.get_terminal_size((100, 30)).lines
    reserved = 18 + int(debug)
    return max(5, terminal_lines - reserved)


def render_screen(
    explorer: TokenTreeExplorer,
    navigator: Navigator,
    describe: TokenDisplay,
    prompt: str,
    focus_path: tuple[int, ...],
    ranked: RankedDistribution | None,
    selected_rank: int,
    *,
    budget_nats: float,
    max_tokens: int,
    debug: bool,
) -> None:
    clear()
    width = max(60, shutil.get_terminal_size((100, 30)).columns)
    state = explorer.snapshot
    stats = explorer.stats()
    suggestion = explorer.cached_greedy_suggestion(
        max_bits=budget_nats / math.log(2),
        max_tokens=max_tokens,
    )

    print("natwalk · llama.cpp")
    print("=" * min(78, width))
    print(
        f"Budget: {budget_nats:.2f} nat ({budget_nats / math.log(2):.2f} bit)    "
        f"binary action: {explorer.nats_per_action:.3f} nat = {explorer.bits_per_action:.2f} bit"
    )
    print()
    print("Prompt:")
    print(truncate(prompt.replace("\n", "↵"), width))
    print()
    print("Committed:")
    print(fmt_tokens(describe, state.prefix))

    print()
    if suggestion.tokens:
        tail = " · …" if not suggestion.complete else ""
        print(f"Suggestion [{suggestion.nats:.3f}/{budget_nats:.2f} nat]:")
        costs = suggestion_costs(explorer, suggestion.tokens)
        print(render_suggestion(describe, suggestion.tokens, costs) + tail)
    elif suggestion.complete and suggestion.next_token_nats is not None:
        print(
            f"Suggestion: —   next greedy token costs {suggestion.next_token_nats:.3f} nat"
        )
    else:
        print("Suggestion: ⟳ computing…")

    print()
    print(
        f"search: {stats.nodes} nodes · {stats.expanded} model expansions · "
        f"{stats.frontier} frontier"
        f"{' · ⟳' if stats.computing else ''}"
        f"{' · memory cap' if stats.saturated else ''}"
    )
    print(f"focus: {fmt_tokens(describe, focus_path, limit=12)}")
    if debug:
        print(
            f"range=[{state.lo:.9f}, {state.hi:.9f}) · "
            f"binary={explorer.supplied_nats:.3f} nat · "
            f"path={state.path_surprisal:.3f} nat · focus_depth={len(focus_path)}"
        )

    print()
    highlighted_token: int | None = None
    if len(focus_path) < len(suggestion.tokens):
        if tuple(suggestion.tokens[: len(focus_path)]) == focus_path:
            highlighted_token = suggestion.tokens[len(focus_path)]

    if ranked is None:
        print("  ∅ end of generation")
    else:
        render_distribution(
            ranked,
            describe,
            selected_rank=selected_rank,
            highlighted_token=highlighted_token,
            lines=tree_line_budget(debug),
            width=width,
        )

    print()
    print("0 / 1: choose exact half    Space: accept suggestion    Backspace: undo")
    print("[/]: ± budget nat    d: debug    q: quit")
    print("↑/↓ inspect rank    ← parent distribution    → selected child distribution")


def load_model(args: argparse.Namespace) -> tuple[Llama, Path]:
    path = Path(args.model_path).expanduser() if args.model_path else resolve_ollama_gguf(args.model)
    print(f"Loading {path} …", file=sys.stderr)
    kwargs: dict[str, object] = {
        "model_path": str(path),
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "n_gpu_layers": args.n_gpu_layers,
        "logits_all": False,
        "verbose": args.llama_verbose,
    }
    if args.threads is not None:
        kwargs["n_threads"] = args.threads
        kwargs["n_threads_batch"] = args.threads
    llm = Llama(**kwargs)
    return llm, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="raw text prefix to continue")
    parser.add_argument("--model", default="ministral-3:14b", help="Ollama model name")
    parser.add_argument("--model-path", help="GGUF path; bypasses Ollama lookup")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="llama.cpp GPU-offloaded layers; requires a GPU-enabled llama-cpp-python build",
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument("--tree-nodes", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--choices", type=int, default=2)
    parser.add_argument("--budget-nats", type=float, default=1.5)
    parser.add_argument("--budget-step", type=float, default=0.25)
    parser.add_argument("--llama-verbose", action="store_true")
    args = parser.parse_args()
    if args.budget_nats < 0:
        parser.error("--budget-nats must be >= 0")
    if args.budget_step <= 0:
        parser.error("--budget-step must be > 0")
    if args.tree_nodes == 1 or args.tree_nodes < 0:
        parser.error("--tree-nodes must be 0 (unlimited) or >= 2")
    return args


def main() -> None:
    args = parse_args()
    llm, model_path = load_model(args)
    describe = TokenDisplay(llm)
    prompt_tokens = llm.tokenize(args.prompt.encode("utf-8"), add_bos=True, special=True)
    if len(prompt_tokens) >= args.n_ctx:
        raise ValueError(
            f"prompt uses {len(prompt_tokens)} tokens but --n-ctx is only {args.n_ctx}"
        )

    cursor = LlamaCursor(llm, prompt_tokens)
    navigator = Navigator(cursor, choices=args.choices)
    budget_nats = args.budget_nats
    focus_path: tuple[int, ...] = ()
    selected_rank = 0
    focus_cache: dict[tuple[int, ...], RankedDistribution] = {}
    debug = False

    print(
        f"Model: {model_path}\n"
        f"Prompt tokens: {len(prompt_tokens)} · vocab: {llm.n_vocab()} · ctx: {args.n_ctx}",
        file=sys.stderr,
    )

    try:
        with TokenTreeExplorer(navigator, max_nodes=args.tree_nodes) as explorer:
            with terminal_session():
                while True:
                    ranked = focus_distribution(explorer, navigator, focus_path, focus_cache)
                    if ranked is not None:
                        selected_rank = min(selected_rank, len(ranked.tokens) - 1)
                    else:
                        selected_rank = 0

                    render_screen(
                        explorer,
                        navigator,
                        describe,
                        args.prompt,
                        focus_path,
                        ranked,
                        selected_rank,
                        budget_nats=budget_nats,
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
                        continue
                    if key == "]":
                        budget_nats += args.budget_step
                        continue

                    if key in ("\x7f", "\b"):
                        if explorer.undo():
                            focus_path = ()
                            selected_rank = 0
                            focus_cache.clear()
                        continue

                    if key in (" ", "\r", "\n"):
                        suggestion = explorer.cached_greedy_suggestion(
                            max_bits=budget_nats / math.log(2),
                            max_tokens=args.max_tokens,
                        )
                        if suggestion.tokens:
                            accept_completion(explorer, suggestion.tokens)
                            focus_path = ()
                            selected_rank = 0
                            focus_cache.clear()
                        continue

                    if key.isdigit():
                        bucket = int(key)
                        if 0 <= bucket < explorer.choices:
                            explorer.choose(bucket)
                            focus_path = ()
                            selected_rank = 0
                            focus_cache.clear()
                        continue

                    if ranked is None:
                        if key == "LEFT" and focus_path:
                            focus_path = focus_path[:-1]
                            selected_rank = 0
                        continue
                    if key == "UP":
                        selected_rank = max(0, selected_rank - 1)
                    elif key == "DOWN":
                        selected_rank = min(len(ranked.tokens) - 1, selected_rank + 1)
                    elif key == "LEFT":
                        if focus_path:
                            child = focus_path[-1]
                            old_path = focus_path
                            focus_path = focus_path[:-1]
                            parent = focus_distribution(
                                explorer,
                                navigator,
                                focus_path,
                                focus_cache,
                            )
                            if parent is not None:
                                try:
                                    selected_rank = parent.tokens.index(child)
                                except ValueError:
                                    selected_rank = 0
                            else:
                                selected_rank = 0
                            focus_cache.setdefault(old_path, ranked)
                    elif key == "RIGHT":
                        token = ranked.tokens[selected_rank]
                        new_path = (*focus_path, token)
                        child = focus_distribution(
                            explorer,
                            navigator,
                            new_path,
                            focus_cache,
                        )
                        if child is not None:
                            focus_path = new_path
                            selected_rank = 0

    except KeyboardInterrupt:
        pass
    finally:
        state = navigator.snapshot()
        print(
            f"final: binary_actions={state.actions}, binary={navigator.supplied_nats:.6f} nat "
            f"({navigator.supplied_bits:.6f} bit), committed_tokens={len(state.prefix)}, "
            f"path_surprisal={state.path_surprisal:.6f} nat"
        )
        llm.close()


if __name__ == "__main__":
    main()
