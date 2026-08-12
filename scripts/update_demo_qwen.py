from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REVISION = "bc640142c66e1fdd12af0bd68f40445458f3869b"
MODEL_NAME = "Qwen3-4B-Q6_K.gguf"
DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF"
    / "snapshots"
    / REVISION
    / MODEL_NAME
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("NATWALK_QWEN_MODEL", DEFAULT_MODEL)
    ).expanduser()

    if not model.is_file():
        raise SystemExit(
            f"Qwen demo model not found at {model}\n\n"
            "Pass its path as the first argument, set NATWALK_QWEN_MODEL, or download the "
            "expected model with:\n"
            "  uvx --from huggingface-hub hf download Qwen/Qwen3-4B-GGUF "
            f"{MODEL_NAME} --revision {REVISION}"
        )

    for command in ("uv", "vhs"):
        if shutil.which(command) is None:
            raise SystemExit(f"{command} is required to regenerate the Qwen demo")

    env = os.environ.copy()
    env["NATWALK_QWEN_MODEL"] = str(model.resolve())
    subprocess.run(
        ["vhs", "assets/demo-qwen.tape"],
        cwd=root,
        env=env,
        check=True,
    )
    print(f"Updated {root / 'assets/demo-qwen.gif'}")


if __name__ == "__main__":
    main()
