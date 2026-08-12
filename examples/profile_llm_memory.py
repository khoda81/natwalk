# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "llama-cpp-python>=0.3.16",
#   "numpy>=1.26",
#   "psutil>=6",
# ]
# ///

"""Profile memory growth while Natwalk autonomously explores a GGUF model."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import psutil
from llama_cpp import Llama
from llm_app import LlamaCursorFactory, _tokenize_prompt, resolve_ollama_gguf

from natwalk.cli import HelpFormatter
from natwalk.engine import EngineClient

_MIB = 1024 * 1024


def _gpu_process_mib(pid: int) -> int | None:
    """Return NVIDIA memory attributed to ``pid``, if nvidia-smi is available."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    total = 0
    found = False
    for line in result.stdout.splitlines():
        process, separator, memory = line.partition(",")
        if not separator or process.strip() != str(pid):
            continue
        try:
            total += int(memory.strip())
        except ValueError:
            return None
        found = True
    return total if found else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run autonomous Natwalk search and emit CSV memory samples for the isolated "
            "llama.cpp engine process."
        ),
        formatter_class=HelpFormatter,
    )
    parser.add_argument("prompt", help="raw text prefix used for the benchmark")
    parser.add_argument(
        "--model",
        default="ministral-3:14b",
        help="Ollama model name used when --model-path is omitted",
    )
    parser.add_argument("--model-path", help="local GGUF path; bypasses Ollama lookup")
    parser.add_argument("--n-ctx", type=int, default=4096, help="llama.cpp context size")
    parser.add_argument("--n-batch", type=int, default=512, help="llama.cpp eval batch size")
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="layers to offload to GPU; -1 means all layers",
    )
    parser.add_argument("--threads", type=int, help="CPU threads for llama.cpp")
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="seconds to profile; <=0 runs until interrupted",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="seconds between CSV samples",
    )
    parser.add_argument(
        "--max-tree-bytes",
        type=int,
        help="optional authoritative distribution budget for comparison runs",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    model_path = (
        Path(args.model_path).expanduser() if args.model_path else resolve_ollama_gguf(args.model)
    )
    tokenizer = Llama(
        model_path=str(model_path),
        vocab_only=True,
        n_gpu_layers=0,
        verbose=False,
    )
    try:
        prompt_tokens, initial_terminal = _tokenize_prompt(tokenizer, args.prompt)
    finally:
        tokenizer.close()

    factory = LlamaCursorFactory(
        model_path=str(model_path),
        prompt_tokens=prompt_tokens,
        initial_terminal=initial_terminal,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_gpu_layers=args.n_gpu_layers,
        threads=args.threads,
        verbose=False,
    )
    client = EngineClient(factory, max_tree_bytes=args.max_tree_bytes)
    client.start()
    try:
        client.wait_ready()
        pid = client.pid
        if pid is None:
            raise RuntimeError("engine process has no pid after startup")
        process = psutil.Process(pid)
        started = time.monotonic()
        deadline = None if args.duration <= 0 else started + args.duration

        print(
            "elapsed_s,nodes,frontier,authoritative_distribution_mib,"
            "rss_mib,rss_minus_distribution_mib,gpu_process_mib",
            flush=True,
        )
        while True:
            client.poll()
            now = time.monotonic()
            rss = process.memory_info().rss
            distributions = client.authoritative_distribution_bytes
            gpu = _gpu_process_mib(pid)
            gpu_field = "" if gpu is None else str(gpu)
            print(
                f"{now - started:.3f},{len(client.tree.nodes)},{client.frontier},"
                f"{distributions / _MIB:.3f},{rss / _MIB:.3f},"
                f"{(rss - distributions) / _MIB:.3f},{gpu_field}",
                flush=True,
            )
            if deadline is not None and now >= deadline:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.terminate()


if __name__ == "__main__":
    main()
