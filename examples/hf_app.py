# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26",
#   "torch>=2.2",
#   "transformers>=4.51",
# ]
# ///

"""Interactive natwalk explorer for Hugging Face causal language models."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from natwalk.cli import HelpFormatter, add_tui_arguments
from natwalk.tree import RankedDistribution
from natwalk.tui import run_tui


@dataclass(frozen=True, slots=True, eq=False)
class _NumpyDistribution:
    """Complete ranked distribution retained in compact CPU NumPy storage."""

    tokens: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        if self.tokens.dtype != np.dtype("uint32"):
            raise TypeError("ranked token ids must be uint32")
        if self.probabilities.dtype != np.dtype("float64"):
            raise TypeError("ranked probabilities must be float64")
        if self.tokens.ndim != 1 or self.probabilities.ndim != 1:
            raise ValueError("ranked arrays must be one-dimensional")
        if len(self.tokens) != len(self.probabilities):
            raise ValueError("ranked token/probability lengths differ")
        self.tokens.flags.writeable = False
        self.probabilities.flags.writeable = False

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def revealed(self) -> int:
        return len(self)

    @property
    def storage_bytes(self) -> int:
        return int(self.tokens.nbytes + self.probabilities.nbytes)

    def token(self, rank: int) -> int:
        return int(self.tokens[rank])

    def probability(self, rank: int) -> float:
        return float(self.probabilities[rank])

    def mass(self, start: int, end: int) -> float:
        start = min(max(start, 0), len(self))
        end = min(max(end, 0), len(self))
        if start >= end:
            return 0.0
        return float(np.sum(self.probabilities[start:end], dtype=np.float64))

    def rank(self, token: int) -> int:
        matches = np.flatnonzero(self.tokens == token)
        if not len(matches):
            raise ValueError(f"{token!r} is not in distribution")
        return int(matches[0])

    def nats(self, rank: int) -> float:
        probability = self.probability(rank)
        return -math.log(probability) if probability != 0.0 else math.inf

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _NumpyDistribution):
            return NotImplemented
        return np.array_equal(self.tokens, other.tokens) and np.array_equal(
            self.probabilities, other.probabilities
        )


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    tokens: tuple[int, ...]
    terminal: bool
    distribution: _NumpyDistribution | None


class HFCursor:
    """Replay-safe causal cursor using ordinary full-context Transformers forwards."""

    def __init__(
        self,
        model,
        prompt_tokens: Sequence[int],
        *,
        eos_token_ids: Sequence[int],
        terminal: bool,
        device: str,
        context_length: int | None,
    ) -> None:
        if not prompt_tokens:
            raise ValueError("Transformers requires at least one bootstrap token")
        self.model = model
        self._tokens = [int(token) for token in prompt_tokens]
        self._eos_token_ids = frozenset(int(token) for token in eos_token_ids)
        self.terminal = terminal
        self.device = device
        self.context_length = context_length
        self._distribution: _NumpyDistribution | None = None

    def checkpoint(self) -> object:
        return _Checkpoint(tuple(self._tokens), self.terminal, self._distribution)

    def restore(self, checkpoint: object) -> None:
        state = cast(_Checkpoint, checkpoint)
        self._tokens = list(state.tokens)
        self.terminal = state.terminal
        self._distribution = state.distribution

    def predict(self) -> Sequence[float] | RankedDistribution:
        if self.terminal:
            return ()
        if self._distribution is not None:
            return self._distribution

        tokens = self._tokens
        if self.context_length is not None:
            tokens = tokens[-self.context_length :]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, use_cache=False)
        logits = output.logits[0, -1].detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(dtype=np.float64)

        order = np.argsort(-probabilities, kind="stable")
        tokens_by_rank = np.ascontiguousarray(order, dtype=np.uint32)
        ranked_probabilities = np.ascontiguousarray(probabilities[order], dtype=np.float64)
        self._distribution = _NumpyDistribution(tokens_by_rank, ranked_probabilities)
        return self._distribution

    def observe(self, token: int) -> None:
        value = int(token)
        self._tokens.append(value)
        self.terminal = value in self._eos_token_ids
        self._distribution = None


def _context_length(config) -> int | None:
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


@dataclass(frozen=True, slots=True)
class HFCursorFactory:
    """Spawn-safe description of one Transformers causal-LM cursor."""

    model_id: str
    prompt_tokens: tuple[int, ...]
    eos_token_ids: tuple[int, ...]
    initial_terminal: bool
    device: str
    dtype: str
    context_length: int | None
    trust_remote_code: bool

    def __call__(self) -> HFCursor:
        kwargs: dict[str, object] = {"trust_remote_code": self.trust_remote_code}
        dtype = _resolve_dtype(self.dtype)
        if dtype is not None:
            kwargs["dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        model.to(self.device)
        model.eval()
        context_length = self.context_length
        if context_length is None:
            context_length = _context_length(model.config)
        return HFCursor(
            model,
            self.prompt_tokens,
            eos_token_ids=self.eos_token_ids,
            terminal=self.initial_terminal,
            device=self.device,
            context_length=context_length,
        )


def _token_ids(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return tuple(int(token) for token in value)


def _tokenize_prompt(tokenizer, prompt: str) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    """Return executable prompt tokens, EOS ids, and semantic terminal state."""
    tokens = tuple(int(token) for token in tokenizer.encode(prompt, add_special_tokens=True))
    eos_token_ids = _token_ids(tokenizer.eos_token_id)
    if tokens:
        return tokens, eos_token_ids, bool(prompt) and tokens[-1] in eos_token_ids
    if prompt:
        raise ValueError("prompt tokenization produced no tokens")

    bootstrap = tokenizer.bos_token_id
    if bootstrap is None and eos_token_ids:
        bootstrap = eos_token_ids[0]
    if bootstrap is None:
        raise ValueError(
            "empty prompt produced no tokens and this tokenizer defines neither BOS nor EOS "
            "for bootstrapping"
        )
    # The bootstrap establishes an executable model state only. Even when it is
    # also an EOS id, an empty user prompt is not semantically terminal.
    return (int(bootstrap),), eos_token_ids, False


class TokenDisplay:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self._cache: dict[int, str] = {}

    def __call__(self, token: int) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached

        piece = self.decode((token,))
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

    def decode(self, tokens: tuple[int, ...]) -> str:
        return self.tokenizer.decode(
            list(tokens),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore a Hugging Face causal LM's next-token probability tree.",
        formatter_class=HelpFormatter,
    )
    parser.add_argument("prompt", help="raw text prefix to continue")

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model id or local Transformers model directory",
    )
    model.add_argument(
        "--tokenizer",
        help="tokenizer id/directory; defaults to --model",
    )
    model.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (for example cpu, cuda, cuda:1, mps); default chooses automatically",
    )
    model.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="model weight dtype; auto keeps the checkpoint/config default",
    )
    model.add_argument(
        "--context-length",
        type=int,
        help="maximum trailing tokens sent to the model; default uses model configuration",
    )
    model.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow model/tokenizer repositories to execute custom Python code",
    )

    add_tui_arguments(
        parser,
        max_tokens_default=128,
        max_tokens_help=(
            "maximum token length of suggestions and row previews; does not cap background search"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.context_length is not None and args.context_length <= 0:
        raise ValueError("--context-length must be positive")

    device = _resolve_device(args.device)
    tokenizer_id = args.tokenizer or args.model
    print(f"Loading {args.model} on {device} …", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_tokens, eos_token_ids, initial_terminal = _tokenize_prompt(tokenizer, args.prompt)

    print(
        f"Model: {args.model}\n"
        f"Prompt tokens: {len(prompt_tokens)} · vocab: {len(tokenizer)} · "
        f"device: {device} · dtype: {args.dtype}",
        file=sys.stderr,
    )
    display = TokenDisplay(tokenizer)
    factory = HFCursorFactory(
        model_id=args.model,
        prompt_tokens=prompt_tokens,
        eos_token_ids=eos_token_ids,
        initial_terminal=initial_terminal,
        device=device,
        dtype=args.dtype,
        context_length=args.context_length,
        trust_remote_code=args.trust_remote_code,
    )
    run_tui(
        factory,
        display,
        title=f"natwalk · transformers · {args.model}",
        context=args.prompt,
        decode_tokens=display.decode,
        max_tokens=args.max_tokens,
        budget_nats=args.budget_nats,
        budget_step=args.budget_step,
        lines=args.tree_lines,
        max_tree_bytes=args.max_tree_bytes,
    )


if __name__ == "__main__":
    main()
