from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_hf_app():
    numpy = types.ModuleType("numpy")

    torch = types.ModuleType("torch")
    torch.float32 = object()
    torch.float16 = object()
    torch.bfloat16 = object()
    torch.float64 = object()
    torch.long = object()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
    transformers.AutoTokenizer = type("AutoTokenizer", (), {})

    name = "_natwalk_test_hf_app"
    path = Path(__file__).parents[1] / "examples" / "hf_app.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Hugging Face example")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "numpy": numpy,
            "torch": torch,
            "transformers": transformers,
            name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class _Tokenizer:
    def __init__(self, tokens: tuple[int, ...], *, bos, eos) -> None:
        self.tokens = tokens
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        self.calls.append((text, add_special_tokens))
        return list(self.tokens)


class HFCursorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_hf_app()

    def test_empty_prompt_bootstraps_without_becoming_terminal(self) -> None:
        tokenizer = _Tokenizer((), bos=9, eos=9)

        tokens, eos_token_ids, terminal = self.app._tokenize_prompt(tokenizer, "")

        self.assertEqual(tokens, (9,))
        self.assertEqual(eos_token_ids, (9,))
        self.assertFalse(terminal)
        self.assertEqual(tokenizer.calls, [("", True)])

    def test_empty_prompt_can_fall_back_to_eos_when_bos_is_missing(self) -> None:
        tokenizer = _Tokenizer((), bos=None, eos=7)

        tokens, eos_token_ids, terminal = self.app._tokenize_prompt(tokenizer, "")

        self.assertEqual(tokens, (7,))
        self.assertEqual(eos_token_ids, (7,))
        self.assertFalse(terminal)

    def test_nonempty_prompt_terminal_state_uses_all_eos_ids(self) -> None:
        tokenizer = _Tokenizer((3, 7), bos=9, eos=(7, 8))

        tokens, eos_token_ids, terminal = self.app._tokenize_prompt(tokenizer, "hello")

        self.assertEqual(tokens, (3, 7))
        self.assertEqual(eos_token_ids, (7, 8))
        self.assertTrue(terminal)

    def test_empty_prompt_without_bootstrap_token_has_clear_error(self) -> None:
        tokenizer = _Tokenizer((), bos=None, eos=None)

        with self.assertRaisesRegex(ValueError, "neither BOS nor EOS"):
            self.app._tokenize_prompt(tokenizer, "")

    def test_checkpoint_restores_tokens_terminal_state_and_cached_distribution(self) -> None:
        cursor = self.app.HFCursor(
            object(),
            (9,),
            eos_token_ids=(9,),
            terminal=False,
            device="cpu",
            context_length=16,
        )
        marker = object()
        cursor._distribution = marker
        checkpoint = cursor.checkpoint()

        self.assertIs(cursor.predict(), marker)
        cursor.observe(9)
        self.assertEqual(cursor.predict(), ())

        cursor.restore(checkpoint)
        self.assertEqual(cursor._tokens, [9])
        self.assertFalse(cursor.terminal)
        self.assertIs(cursor.predict(), marker)

    def test_checkpoint_is_a_token_snapshot_not_only_a_length(self) -> None:
        cursor = self.app.HFCursor(
            object(),
            (1,),
            eos_token_ids=(),
            terminal=False,
            device="cpu",
            context_length=None,
        )
        cursor.observe(2)
        branch_checkpoint = cursor.checkpoint()
        cursor.restore(self.app._Checkpoint((1,), False, None))
        cursor.observe(3)

        cursor.restore(branch_checkpoint)

        self.assertEqual(cursor._tokens, [1, 2])

    def test_auto_device_falls_back_to_cpu(self) -> None:
        self.assertEqual(self.app._resolve_device("auto"), "cpu")
        self.assertEqual(self.app._resolve_device("cuda:2"), "cuda:2")

    def test_context_length_uses_common_causal_lm_config_names(self) -> None:
        config = types.SimpleNamespace(max_position_embeddings=None, n_positions=1024)
        self.assertEqual(self.app._context_length(config), 1024)


if __name__ == "__main__":
    unittest.main()
