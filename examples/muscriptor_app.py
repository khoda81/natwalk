# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "muscriptor>=0.3.0",
# ]
# ///

"""Explore MuScriptor's causal transcription distribution with natwalk.

The script declares MuScriptor as an inline dependency so it can be run
straight from a natwalk checkout with ``uv run examples/muscriptor_app.py``.
The current natwalk checkout is imported from ``../src`` on purpose: this is a
demo of the code being developed, not an installed release.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from muscriptor import TranscriptionModel
from muscriptor.modules.streaming import increment_steps, init_states
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES, MT3Tokenizer
from muscriptor.tokenizer.notes import DRUM_PROGRAM

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from natwalk.tui import run_tui  # noqa: E402


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show concrete defaults while leaving semantic ``None`` defaults to prose."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None:
            return action.help or ""
        return super()._get_help_string(action)


VALID_CARD = 1393
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 5
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE


def midi_name(pitch: int) -> str:
    names = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def snapshot_control_state(
    model_state: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Snapshot streaming controls while sharing preallocated KV storage."""
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


@torch.inference_mode()
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


class MuscriptorDisplay:
    """Lightweight token display independent of the inference model."""

    def __init__(self) -> None:
        self.tokenizer = MT3Tokenizer(
            instrument_vocabulary="MT3_FULL_PLUS",
            max_shift_steps=1001,
        )
        group_map = self.tokenizer.group_program_map
        self.program_to_name = {
            group_map[group][0]: name
            for name, group in MT3_FULL_PLUS_GROUP_NAMES.items()
            if group in group_map and group_map[group]
        }

    def __call__(self, token: int) -> str:
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
            if value == DRUM_PROGRAM:
                return "drums"
            return self.program_to_name.get(value, f"program_{value}")
        if kind == "drum":
            return f"drum({midi_name(value)})"
        return f"{kind}({value})"


class MuscriptorCursor:
    """Checkpoint-only complete-distribution cursor over one audio chunk.

    Observed tokens are buffered until a prediction is needed. MuScriptor's
    streaming attention only supports a square first-step prefill or T_q=1
    incremental decoding, so speculative replay is materialized one token at a
    time through the model before the final next-token distribution is returned.
    """

    def __init__(self, ctx: MuscriptorContext) -> None:
        self.ctx = ctx
        self.lm = ctx.lm
        self.device = ctx.device
        cache_len = ctx.prepend_length + ctx.max_tokens + 1
        self.model_state = init_states(self.lm, batch_size=1, sequence_length=cache_len)
        self._pending = [self.lm.initial_token_id]
        self._first_step = True
        self._last_token: int | None = None
        self._probs: torch.Tensor | None = None

    def checkpoint(self) -> object:
        self.predict()
        return (
            tuple(self._pending),
            self._first_step,
            self._last_token,
            self._probs,
            snapshot_control_state(self.model_state),
        )

    def restore(self, checkpoint: object) -> None:
        pending, first_step, last_token, probs, controls = checkpoint
        restore_control_state(self.model_state, controls)
        self._pending = list(pending)
        self._first_step = first_step
        self._last_token = last_token
        self._probs = probs

    @torch.inference_mode()
    def predict(self) -> Sequence[float]:
        if self._last_token == self.ctx.tokenizer.eos_id:
            return ()
        if self._probs is not None:
            return self._probs
        if not self._pending:
            raise RuntimeError("MuScriptor cursor has no pending input to evaluate")

        logits: torch.Tensor | None = None
        with self.lm.autocast:
            for token in self._pending:
                input_ = torch.tensor([[token]], device=self.device, dtype=torch.long)
                logits = self.lm._compute_logits(
                    input_,
                    self.ctx.cfg_conditions,
                    self.model_state,
                    first_step=self._first_step,
                    cfg_coef=1.0,
                    forbidden_tokens=None,
                )[0]

                increment = 1 + (self.ctx.prepend_length if self._first_step else 0)
                increment_steps(self.lm.transformer, self.model_state, increment=increment)
                self._first_step = False

        self._pending.clear()
        assert logits is not None
        logits = logits.float()
        logits[VALID_CARD:] = -torch.inf
        self._probs = torch.softmax(logits, dim=-1)[:VALID_CARD].double().cpu()
        return self._probs

    def observe(self, token: int) -> None:
        self._pending.append(token)
        self._last_token = token
        self._probs = None


@dataclass(frozen=True, slots=True)
class MuscriptorCursorFactory:
    model: str
    device: str
    audio: Path
    chunk: int
    max_tokens: int

    def __call__(self) -> MuscriptorCursor:
        tm = TranscriptionModel.load_model(self.model, device=self.device)
        ctx = MuscriptorContext(
            tm,
            self.audio,
            chunk_index=self.chunk,
            max_tokens=self.max_tokens,
        )
        return ctx.new_cursor()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore MuScriptor's causal transcription probability tree.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="audio file to transcribe")

    model = parser.add_argument_group("MuScriptor")
    model.add_argument(
        "--model",
        default="medium",
        choices=["small", "medium", "large"],
        help="MuScriptor model size",
    )
    model.add_argument("--device", default="cuda", help="PyTorch inference device")
    model.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="zero-based 5-second audio chunk to explore",
    )
    model.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="MuScriptor streaming-cache token capacity and maximum suggestion length",
    )

    natwalk = parser.add_argument_group("natwalk")
    natwalk.add_argument(
        "--tree-lines",
        type=int,
        help="maximum rendered tree rows; default uses the remaining terminal height",
    )
    natwalk.add_argument(
        "--budget-nats",
        type=float,
        default=1.5,
        help="maximum cumulative surprisal of the highlighted/accepted suggestion, in nats",
    )
    natwalk.add_argument(
        "--budget-step",
        type=float,
        default=0.25,
        help="amount '[' and ']' change --budget-nats by",
    )
    natwalk.add_argument(
        "--max-tree-bytes",
        type=int,
        help=(
            "soft limit on retained authoritative tree-distribution bytes; at the limit "
            "autonomous search pauses but explicit navigation still works; default unlimited"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading MuScriptor {args.model} on {args.device} …")
    run_tui(
        MuscriptorCursorFactory(
            model=args.model,
            device=args.device,
            audio=args.audio,
            chunk=args.chunk,
            max_tokens=args.max_tokens,
        ),
        MuscriptorDisplay(),
        title=f"natwalk · MuScriptor · {args.model} · chunk {args.chunk}",
        max_tokens=args.max_tokens,
        budget_nats=args.budget_nats,
        budget_step=args.budget_step,
        lines=args.tree_lines,
        max_tree_bytes=args.max_tree_bytes,
    )


if __name__ == "__main__":
    main()
