# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "llama-cpp-python>=0.3.16",
#   "numpy>=1.26",
# ]
# ///

"""Interactive natwalk explorer for local GGUF language models.

The default model path is resolved from Ollama's local model store so an
existing ``ollama pull`` does not require another model download.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from llama_cpp import Llama

from natwalk.tree import RankedDistribution
from natwalk.tui import run_tui


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


@dataclass(frozen=True, slots=True, eq=False)
class _NumpyDistribution:
    """Complete ranked distribution retained entirely in NumPy storage."""

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
        if not 0 <= start <= end <= len(self):
            raise IndexError((start, end))
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
    n_tokens: int
    last_token: int
    distribution: _NumpyDistribution | None


class LlamaCursor:
    """Checkpoint-only complete-distribution cursor over one llama.cpp context."""

    def __init__(self, llm: Llama, prompt_tokens: Sequence[int]) -> None:
        if not prompt_tokens:
            raise ValueError("prompt tokenization produced no tokens")
        self.llm = llm
        self.last_token = int(prompt_tokens[-1])
        self._distribution: _NumpyDistribution | None = None
        self.llm.reset()
        self.llm.eval([int(token) for token in prompt_tokens])

    def checkpoint(self) -> object:
        return _Checkpoint(self.llm.n_tokens, self.last_token, self._distribution)

    def restore(self, checkpoint: object) -> None:
        state = cast(_Checkpoint, checkpoint)
        self.llm.n_tokens = state.n_tokens
        self.last_token = state.last_token
        self._distribution = state.distribution

    def predict(self) -> Sequence[float] | RankedDistribution:
        if self.last_token == self.llm.token_eos():
            return ()
        if self._distribution is not None:
            return self._distribution

        logits_ptr = self.llm._ctx.get_logits()  # noqa: SLF001
        logits = np.ctypeslib.as_array(logits_ptr, shape=(self.llm.n_vocab(),)).astype(
            np.float64,
            copy=True,
        )
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        # Stable descending sort preserves token-id order for exact ties, matching
        # Natwalk's dependency-free ranking semantics without Pythonizing the vocab.
        order = np.argsort(-probabilities, kind="stable")
        tokens = np.ascontiguousarray(order, dtype=np.uint32)
        ranked_probabilities = np.ascontiguousarray(probabilities[order], dtype=np.float64)
        self._distribution = _NumpyDistribution(tokens, ranked_probabilities)
        return self._distribution

    def observe(self, token: int) -> None:
        self.llm.eval([int(token)])
        self.last_token = int(token)
        self._distribution = None


@dataclass(frozen=True, slots=True)
class LlamaCursorFactory:
    """Spawn-safe description of one full llama.cpp inference cursor."""

    model_path: str
    prompt_tokens: tuple[int, ...]
    n_ctx: int
    n_batch: int
    n_gpu_layers: int
    threads: int | None
    verbose: bool

    def __call__(self) -> LlamaCursor:
        kwargs: dict[str, object] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "logits_all": False,
            "verbose": self.verbose,
        }
        if self.threads is not None:
            kwargs["n_threads"] = self.threads
            kwargs["n_threads_batch"] = self.threads
        return LlamaCursor(Llama(**kwargs), self.prompt_tokens)


class TokenDisplay:
    def __init__(self, llm: Llama) -> None:
        self.llm = llm
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
        return self.llm.detokenize(list(tokens), special=True).decode(
            "utf-8",
            errors="replace",
        )


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
        default=-1,
        help="layers to offload to GPU; -1 means all layers, 0 forces CPU",
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--tree-lines",
        type=int,
        help="maximum visible distribution rows; default fills the terminal",
    )
    parser.add_argument("--budget-nats", type=float, default=1.5)
    parser.add_argument("--budget-step", type=float, default=0.25)
    parser.add_argument("--llama-verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = (
        Path(args.model_path).expanduser() if args.model_path else resolve_ollama_gguf(args.model)
    )
    print(f"Loading {model_path} …", file=sys.stderr)

    tokenizer = Llama(
        model_path=str(model_path),
        vocab_only=True,
        n_gpu_layers=0,
        verbose=False,
    )
    try:
        prompt_tokens = tuple(
            tokenizer.tokenize(args.prompt.encode("utf-8"), add_bos=True, special=True)
        )
        if len(prompt_tokens) >= args.n_ctx:
            raise ValueError(
                f"prompt uses {len(prompt_tokens)} tokens but --n-ctx is only {args.n_ctx}"
            )

        print(
            f"Model: {model_path}\n"
            f"Prompt tokens: {len(prompt_tokens)} · vocab: {tokenizer.n_vocab()} · ctx: {args.n_ctx}",
            file=sys.stderr,
        )
        display = TokenDisplay(tokenizer)
        factory = LlamaCursorFactory(
            model_path=str(model_path),
            prompt_tokens=prompt_tokens,
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
            n_gpu_layers=args.n_gpu_layers,
            threads=args.threads,
            verbose=args.llama_verbose,
        )
        run_tui(
            factory,
            display,
            title=f"natwalk · llama.cpp · {model_path.name}",
            context=args.prompt,
            decode_tokens=display.decode,
            max_tokens=args.max_tokens,
            budget_nats=args.budget_nats,
            budget_step=args.budget_step,
            lines=args.tree_lines,
        )
    finally:
        tokenizer.close()


if __name__ == "__main__":
    main()
