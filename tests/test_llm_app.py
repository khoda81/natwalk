from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_llm_app():
    numpy = types.ModuleType("numpy")
    llama_cpp = types.ModuleType("llama_cpp")
    llama_cpp.Llama = type("Llama", (), {})

    name = "_natwalk_test_llm_app"
    path = Path(__file__).parents[1] / "examples" / "llm_app.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load llama.cpp example")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"numpy": numpy, "llama_cpp": llama_cpp, name: module},
    ):
        spec.loader.exec_module(module)
    return module


class _Tokenizer:
    def __init__(self, tokens: tuple[int, ...], *, bos: int, eos: int) -> None:
        self.tokens = tokens
        self.bos = bos
        self.eos = eos
        self.calls: list[tuple[bytes, bool, bool]] = []
        self.bos_calls = 0

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:
        self.calls.append((text, add_bos, special))
        return list(self.tokens)

    def token_bos(self) -> int:
        self.bos_calls += 1
        return self.bos

    def token_eos(self) -> int:
        return self.eos


class _Llama:
    def __init__(self, eos: int) -> None:
        self.eos = eos
        self.n_tokens = 0
        self.evaluated: list[tuple[int, ...]] = []

    def reset(self) -> None:
        self.n_tokens = 0

    def eval(self, tokens) -> None:
        values = tuple(int(token) for token in tokens)
        self.evaluated.append(values)
        self.n_tokens += len(values)

    def token_eos(self) -> int:
        return self.eos


class LlamaPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_llm_app()

    def test_empty_prompt_bootstraps_with_model_bos_without_becoming_terminal(self) -> None:
        tokenizer = _Tokenizer((), bos=9, eos=9)

        tokens, terminal = self.app._tokenize_prompt(tokenizer, "")

        self.assertEqual(tokens, (9,))
        self.assertFalse(terminal)
        self.assertEqual(tokenizer.calls, [(b"", True, True)])
        self.assertEqual(tokenizer.bos_calls, 1)

    def test_nonempty_prompt_tokenization_is_unchanged(self) -> None:
        tokenizer = _Tokenizer((3, 7), bos=9, eos=7)

        tokens, terminal = self.app._tokenize_prompt(tokenizer, "hello")

        self.assertEqual(tokens, (3, 7))
        self.assertTrue(terminal)
        self.assertEqual(tokenizer.calls, [(b"hello", True, True)])
        self.assertEqual(tokenizer.bos_calls, 0)

    def test_empty_prompt_without_bos_has_clear_adapter_error(self) -> None:
        tokenizer = _Tokenizer((), bos=-1, eos=7)

        with self.assertRaisesRegex(ValueError, "defines no BOS token"):
            self.app._tokenize_prompt(tokenizer, "")

    def test_bootstrap_terminal_state_survives_checkpoint_restore(self) -> None:
        llm = _Llama(eos=9)
        cursor = self.app.LlamaCursor(llm, (9,), terminal=False)
        marker = object()
        cursor._distribution = marker
        checkpoint = cursor.checkpoint()

        self.assertIs(cursor.predict(), marker)
        cursor.observe(9)
        self.assertEqual(cursor.predict(), ())

        cursor.restore(checkpoint)
        self.assertEqual(llm.n_tokens, 1)
        self.assertIs(cursor.predict(), marker)


if __name__ == "__main__":
    unittest.main()
