# /// script
# requires-python = ">=3.11"
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
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from llama_cpp import Llama

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


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    n_tokens: int
    last_token: int
    probs: np.ndarray | None


class LlamaCursor:
    """Checkpoint-only complete-distribution cursor over one llama.cpp context."""

    def __init__(self, llm: Llama, prompt_tokens: Sequence[int]) -> None:
        if not prompt_tokens:
            raise ValueError("prompt tokenization produced no tokens")
        self.llm = llm
        self.last_token = int(prompt_tokens[-1])
        self._probs: np.ndarray | None = None
        self.llm.reset()
        self.llm.eval([int(token) for token in prompt_tokens])

    def checkpoint(self) -> object:
        return _Checkpoint(self.llm.n_tokens, self.last_token, self._probs)

    def restore(self, checkpoint: object) -> None:
        state = cast(_Checkpoint, checkpoint)
        self.llm.n_tokens = state.n_tokens
        self.last_token = state.last_token
        self._probs = state.probs

    def predict(self) -> Sequence[float]:
        if self.last_token == self.llm.token_eos():
            return ()
        if self._probs is not None:
            return self._probs

        logits_ptr = self.llm._ctx.get_logits()  # noqa: SLF001
        logits = np.ctypeslib.as_array(logits_ptr, shape=(self.llm.n_vocab(),)).astype(
            np.float64,
            copy=True,
        )
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        self._probs = probs
        return probs

    def observe(self, token: int) -> None:
        self.llm.eval([int(token)])
        self.last_token = int(token)
        self._probs = None


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


def load_model(args: argparse.Namespace) -> tuple[Llama, Path]:
    path = (
        Path(args.model_path).expanduser() if args.model_path else resolve_ollama_gguf(args.model)
    )
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
    return Llama(**kwargs), path


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
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--choices", type=int, default=2)
    parser.add_argument("--tree-lines", type=int, default=16)
    parser.add_argument("--budget-nats", type=float, default=1.5)
    parser.add_argument("--budget-step", type=float, default=0.25)
    parser.add_argument("--llama-verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm, model_path = load_model(args)
    try:
        prompt_tokens = llm.tokenize(args.prompt.encode("utf-8"), add_bos=True, special=True)
        if len(prompt_tokens) >= args.n_ctx:
            raise ValueError(
                f"prompt uses {len(prompt_tokens)} tokens but --n-ctx is only {args.n_ctx}"
            )

        print(
            f"Model: {model_path}\n"
            f"Prompt tokens: {len(prompt_tokens)} · vocab: {llm.n_vocab()} · ctx: {args.n_ctx}",
            file=sys.stderr,
        )
        run_tui(
            LlamaCursor(llm, prompt_tokens),
            TokenDisplay(llm),
            title=f"natwalk · llama.cpp · {model_path.name}",
            choices=args.choices,
            max_tokens=args.max_tokens,
            budget_nats=args.budget_nats,
            budget_step=args.budget_step,
            lines=args.tree_lines,
        )
    finally:
        llm.close()


if __name__ == "__main__":
    main()
