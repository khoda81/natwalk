"""MuScriptor demo backend for natwalk.

The adapter uses MuScriptor private APIs because natwalk needs the complete
next-token distribution and direct access to its streaming model state.
"""

from __future__ import annotations

import argparse
import copy
import select
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from muscriptor import TranscriptionModel
from muscriptor.modules.streaming import increment_steps, init_states

from natwalk import Navigator, TreeExplorer

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
            return f"shift({value / self.tokenizer.frame_rate:.2f}s)"
        if kind == "pitch":
            return f"pitch({midi_name(value)})"
        if kind == "velocity":
            return "note_on" if value else "note_off"
        if kind == "tie":
            return "tie"
        if kind == "program":
            return f"program({self.tm._instrument_for_program(value)})"
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
        """Full KV clone fallback; natwalk previews prefer checkpoint/restore."""
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


def fmt_tokens(ctx: MuscriptorContext, tokens: Sequence[int], limit: int = 8) -> str:
    if not tokens:
        return "∅"
    shown = [ctx.describe(token) for token in tokens[:limit]]
    if len(tokens) > limit:
        shown.append("…")
    return " · ".join(shown)


def clear() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def print_screen(explorer: TreeExplorer, ctx: MuscriptorContext) -> None:
    clear()
    state = explorer.snapshot
    stats = explorer.stats()

    print("MuScriptor information-space navigator")
    print("=" * 78)
    print(
        f"K={explorer.choices}  |  exact cost/action = {explorer.nats_per_action:.6f} nat "
        f"= {explorer.bits_per_action:.6f} bit"
    )
    print(
        f"actions={state.actions}  supplied={explorer.supplied_nats:.4f} nat  |  "
        f"committed path surprisal={state.path_surprisal:.4f} nat"
    )
    print(
        f"unresolved local interval=[{state.lo:.9f}, {state.hi:.9f})  "
        f"width={state.hi - state.lo:.6g}"
    )
    print(f"committed tokens={len(state.prefix)}")
    if state.prefix:
        print("tail:", fmt_tokens(ctx, state.prefix[-10:], limit=10))
    print(
        f"prefetch: cached={stats.cached} queued={stats.queued} "
        f"expanded={stats.expanded}{'  ⟳' if stats.computing else ''}"
    )
    print()

    if state.ended:
        print("EOS is forced. Done.")
        return

    previews = explorer.current_previews()
    for bucket, preview in enumerate(previews):
        if preview is None:
            print(f"[{bucket + 1}] ⟳        computing…")
        elif preview.forced:
            print(f"[{bucket + 1}] FORCES   {fmt_tokens(ctx, preview.forced)}")
            if preview.representative:
                print(f"    then ~ {fmt_tokens(ctx, preview.representative)}")
        else:
            print(f"[{bucket + 1}] preview  {fmt_tokens(ctx, preview.representative)}")

    print(f"[{explorer.choices}] …        residual / none of the above")
    print()
    print(f"1-{explorer.choices}: choose bucket   SPACE/ENTER: bucket 1   q: quit")
    print("FORCES is exact; preview/~ is only a representative code point.")


def read_key(timeout: float = 0.25) -> str | None:
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
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--model", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=0, help="5-second chunk index")
    ap.add_argument("--choices", type=int, default=5)
    ap.add_argument("--preview-tokens", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prefetch-depth", type=int, default=2)
    ap.add_argument("--prefetch-cache", type=int, default=128)
    args = ap.parse_args()

    print(f"Loading MuScriptor {args.model} on {args.device} …", file=sys.stderr)
    tm = TranscriptionModel.load_model(args.model, device=args.device)
    print(
        f"Encoding chunk {args.chunk} "
        f"({args.chunk * CHUNK_SECONDS:.1f}–{(args.chunk + 1) * CHUNK_SECONDS:.1f}s) …",
        file=sys.stderr,
    )
    ctx = MuscriptorContext(tm, args.audio, chunk_index=args.chunk, max_tokens=args.max_tokens)
    nav = Navigator(ctx.new_cursor(), choices=args.choices, preview_tokens=args.preview_tokens)

    try:
        with TreeExplorer(
            nav,
            prefetch_depth=args.prefetch_depth,
            max_cached=args.prefetch_cache,
        ) as explorer:
            while True:
                print_screen(explorer, ctx)
                if explorer.snapshot.ended:
                    break

                key = read_key()
                if key is None:
                    continue
                if key.lower() == "q":
                    break
                if key in (" ", "\r", "\n"):
                    bucket = 0
                elif key.isdigit() and 1 <= int(key) <= explorer.choices:
                    bucket = int(key) - 1
                else:
                    continue
                explorer.choose(bucket)
    except KeyboardInterrupt:
        pass
    finally:
        state = nav.state
        print()
        print(
            f"final: actions={state.actions}, supplied={nav.supplied_nats:.6f} nat "
            f"({nav.supplied_bits:.6f} bit), committed_tokens={len(state.cursor.prefix)}, "
            f"path_surprisal={state.path_surprisal:.6f} nat"
        )


if __name__ == "__main__":
    main()
