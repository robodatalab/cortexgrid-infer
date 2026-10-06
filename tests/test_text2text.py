"""Tests for cortexgrid_infer.serve_apps.text2text."""

from __future__ import annotations

import cortexgrid
import json
import torch
import unittest
from unittest import mock
from transformers import Qwen3Config, Qwen3ForCausalLM

from cortexgrid_infer.protocols.completion import ServedCompletingModel
from cortexgrid_infer.serve_apps.text2text import (
    Text2Text,
    allowed_next_tokens,
    continuation_loglikelihood,
)


class TestText2TextForTheImporter(unittest.TestCase):
    def test_client_speaks_the_deployed_app(self):
        deployment = cortexgrid.Deployment(
            key=cortexgrid.DeploymentKey("F", "S", "R"),
            config={},
            url="http://h/r/F/S/R",
            phase="running",
            bundle_fingerprint="",
            replaced_bundle_fingerprint="",
            experiment_name="",
            class_import_path="cortexgrid_infer.serve_apps.text2text:Text2Text",
        )

        model = Text2Text.client(deployment)

        self.assertEqual(
            model,
            ServedCompletingModel(
                key=cortexgrid.DeploymentKey("F", "S", "R"), url="http://h/r/F/S/R"
            ),
        )

    def test_loads_every_file_the_repo_ships(self):
        # Every file a causal LM repo ships is one `from_pretrained` may read.
        self.assertIsNone(Text2Text.ignore_patterns())

    def test_the_model_card_lets_the_model_think_unless_told_otherwise(self):
        self.assertEqual(Text2Text.config()["enable_thinking"], "true")

    def test_the_model_card_serves_eager_unless_told_otherwise(self):
        self.assertEqual(Text2Text.config()["compile"], "false")


class _ModelConfig:
    def __init__(self, max_position_embeddings=32768) -> None:
        if max_position_embeddings is not None:
            self.max_position_embeddings = max_position_embeddings


class _GenerationConfig:
    def __init__(self) -> None:
        self.cache_implementation = None
        self.max_cache_len = None
        self.compile_config = None


class _FakeCausalLM:
    def __init__(self, context=32768) -> None:
        self.config = _ModelConfig(context)
        self.generation_config = _GenerationConfig()
        self.compiled_with: dict | None = None
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def compile(self, **kwargs) -> None:
        self.compiled_with = kwargs


class _Device:
    def __init__(self, type_: str) -> None:
        self.type = type_


class TestText2TextCompiles(unittest.TestCase):
    SERVE = "cortexgrid_infer.serve_apps.text2text"

    def build(self, device: str = "cuda", config: dict | None = None,
              context: int = 32768) -> _FakeCausalLM:
        model = _FakeCausalLM(context)
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config",
                        return_value={"compile": "true", **(config or {})}), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained"), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained",
                        return_value=model), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device(device)):
            self.deployment = Text2Text(
                cortexgrid.DeploymentKey("family", "suffix", "imported")
            )
        return model

    def test_pins_the_cache_to_the_budget_not_the_models_context(self):
        # Every decoded token reads the whole cache, so a 32k context costs 8x
        # the 4k default in keys and values re-read per token.
        model = self.build()

        self.assertEqual(model.generation_config.cache_implementation, "static")
        self.assertEqual(model.generation_config.max_cache_len, 4096)

    def test_takes_the_budget_from_the_model_card(self):
        model = self.build(config={"max_total_tokens": "1024"})

        self.assertEqual(model.generation_config.max_cache_len, 1024)
        self.assertEqual(self.deployment._max_total_tokens, 1024)

    def test_never_asks_for_more_than_the_model_holds(self):
        model = self.build(config={"max_total_tokens": "65536"}, context=8192)

        self.assertEqual(model.generation_config.max_cache_len, 8192)

    def test_caps_a_reply_at_half_the_budget(self):
        # Otherwise a caller asking for more output than the replica is sized
        # for would leave the prompt no room at all.
        self.build()

        self.assertEqual(self.deployment._room_for_the_reply(256), 256)
        self.assertEqual(self.deployment._room_for_the_reply(16 * 1024), 2048)

    def test_asks_for_cuda_graphs_around_decode(self):
        model = self.build()

        self.assertEqual(model.generation_config.compile_config.mode, "reduce-overhead")

    def test_leaves_the_module_for_transformers_to_compile(self):
        # Compiling it here too would compile the same forward twice.
        self.assertIsNone(self.build().compiled_with)

    def test_stays_eager_off_cuda(self):
        model = self.build(device="cpu")

        self.assertFalse(self.deployment._compiled)
        self.assertIsNone(model.generation_config.cache_implementation)
        self.assertIsNone(model.generation_config.max_cache_len)

    def test_stays_eager_unless_the_model_card_asks(self):
        model = self.build(config={"compile": "false"})

        self.assertFalse(self.deployment._compiled)
        self.assertIsNone(model.generation_config.cache_implementation)
        self.assertIsNone(model.generation_config.compile_config)


class TestText2TextThinks(unittest.TestCase):
    SERVE = "cortexgrid_infer.serve_apps.text2text"

    def build(self, card: dict) -> Text2Text:
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config", return_value=card), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained"), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained",
                        return_value=_FakeCausalLM()), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device("cpu")):
            return Text2Text(cortexgrid.DeploymentKey("family", "suffix", "imported"))

    def test_takes_whether_to_think_from_the_model_card(self):
        self.assertFalse(self.build({"enable_thinking": "false"})._enable_thinking)

    def test_thinks_when_the_model_card_does_not_say(self):
        self.assertTrue(self.build({})._enable_thinking)


class TestPromptTruncation(unittest.TestCase):
    """An over-long prompt would resize the KV cache and recompile the decode
    loop, which costs more than the whole generation. It is cut instead."""

    SERVE = "cortexgrid_infer.serve_apps.text2text"

    def setUp(self) -> None:
        model = _FakeCausalLM(context=8)
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config", return_value={}), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained"), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained",
                        return_value=model), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device("cuda")):
            self.deployment = Text2Text(
                cortexgrid.DeploymentKey("family", "suffix", "imported")
            )

    def inputs(self, length: int) -> dict:
        row = torch.arange(length).unsqueeze(0)
        return {"input_ids": row, "attention_mask": torch.ones_like(row)}

    def test_leaves_a_prompt_within_the_limit_alone(self):
        kept = self.deployment._truncate(self.inputs(8), 8)

        self.assertEqual(kept["input_ids"].shape[-1], 8)

    def test_cuts_an_over_long_prompt_to_the_limit(self):
        with self.assertLogs(self.SERVE, "WARNING"):
            kept = self.deployment._truncate(self.inputs(20), 8)

        self.assertEqual(kept["input_ids"].shape[-1], 8)
        self.assertEqual(kept["attention_mask"].shape[-1], 8)

    def test_keeps_the_end_of_the_prompt(self):
        # A chat template puts the turn to answer last; dropping the tail would
        # cost the instruction to reply at all.
        with self.assertLogs(self.SERVE, "WARNING"):
            kept = self.deployment._truncate(self.inputs(20), 8)

        self.assertEqual(kept["input_ids"][0].tolist(), list(range(12, 20)))

    def test_says_what_it_dropped(self):
        with self.assertLogs(self.SERVE, "WARNING") as logged:
            self.deployment._truncate(self.inputs(20), 8)

        self.assertIn("20", logged.output[0])


class TestAllowedNextTokens(unittest.TestCase):
    def test_allows_only_what_continues_a_choice(self):
        cases = [
            ("the_first_token_of_every_choice", [[1, 2], [1, 3], [4]], [], [1, 4]),
            ("the_tokens_that_continue_the_choices_begun", [[1, 2], [1, 3], [4]], [1], [2, 3]),
            ("only_the_end_once_a_choice_is_complete", [[1, 2], [1, 3], [4]], [1, 2], [99]),
            ("only_the_end_after_a_one_token_choice", [[1, 2], [1, 3], [4]], [4], [99]),
            ("the_end_or_more_when_one_choice_continues_another", [[1], [1, 2]], [1], [2, 99]),
            ("only_the_end_off_every_choice", [[1, 2], [4]], [5], [99]),
            ("only_the_end_without_choices", [], [], [99]),
        ]
        for name, choices, generated, expected_tokens in cases:
            with self.subTest(name):
                self.assertEqual(allowed_next_tokens(choices, generated, 99), expected_tokens)


class TestReadingTheModel(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=256,
            )
        ).eval()

    def test_a_continuation_s_loglikelihood_is_the_full_sequence_s(self):
        cases = [
            ("one_token", [1, 2, 3, 4], [5]),
            ("several_tokens", [1, 2, 3, 4], [5, 6, 7]),
            ("a_one_token_prompt", [1], [5, 6]),
            ("a_continuation_longer_than_the_prompt", [1, 2], [5, 6, 7, 8, 9]),
            ("a_repeated_token", [3, 3, 3], [3, 3]),
        ]
        for name, prompt, continuation in cases:
            with self.subTest(name):
                loglikelihood = continuation_loglikelihood(self.model, prompt, continuation)

                with torch.inference_mode():
                    logits = self.model(input_ids=torch.tensor([prompt + continuation])).logits[0]
                log_probabilities = torch.log_softmax(logits.float(), dim=-1)
                predicting = torch.arange(len(prompt) - 1, len(prompt) + len(continuation) - 1)
                expected = float(log_probabilities[predicting, torch.tensor(continuation)].sum())
                self.assertAlmostEqual(loglikelihood, expected, places=4)


class _FakeTokenizer:
    def __init__(self, ids_of_each_text: dict[str, list[int]]) -> None:
        self._ids_of_each_text = ids_of_each_text
        self.pad_token = "<pad>"
        self.eos_token_id = 99

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": self._ids_of_each_text[text]}


class TestText2TextLoads(unittest.TestCase):
    SERVE = "cortexgrid_infer.serve_apps.text2text"

    def load(self, card: dict) -> mock.Mock:
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config", return_value=card), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained"), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained",
                        return_value=_FakeCausalLM()) as loads, \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device("cpu")):
            Text2Text(cortexgrid.DeploymentKey("family", "suffix", "imported"))
        return loads

    def test_loads_the_weights_in_the_dtype_the_card_names(self):
        cases = [
            ("float16_unless_the_card_says", {}, torch.float16),
            ("bfloat16", {"dtype": "bfloat16"}, torch.bfloat16),
            ("float32", {"dtype": "float32"}, torch.float32),
        ]
        for name, card, expected_dtype in cases:
            with self.subTest(name):
                self.assertEqual(self.load(card).call_args.kwargs["torch_dtype"], expected_dtype)


class TestText2TextReadsTheModel(unittest.IsolatedAsyncioTestCase):
    SERVE = "cortexgrid_infer.serve_apps.text2text"

    def build(self, ids_of_each_text: dict[str, list[int]]) -> Text2Text:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=4,
                max_position_embeddings=256,
            )
        ).eval()
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config", return_value={"enable_thinking": "false"}), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained",
                        return_value=_FakeTokenizer(ids_of_each_text)), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained", return_value=model), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=torch.device("cpu")):
            return Text2Text(cortexgrid.DeploymentKey("family", "suffix", "imported"))

    def test_generation_may_only_spell_out_one_of_the_choices(self):
        cases = [
            ("the_first_token_of_either", [7, 7, 7], [1, 3]),
            ("the_rest_of_the_choice_begun", [7, 7, 7, 1], [2]),
            ("only_the_end_once_a_choice_is_spelt", [7, 7, 7, 1, 2], [99]),
            ("only_the_end_after_the_one_token_choice", [7, 7, 7, 3], [99]),
        ]
        deployment = self.build({"yes": [1, 2], "no": [3]})
        allowed = deployment._only_the_choices(["yes", "no"], prompt_length=3)
        for name, sequence, expected_tokens in cases:
            with self.subTest(name):
                self.assertEqual(allowed(0, torch.tensor(sequence)), expected_tokens)

    async def test_hands_back_the_last_hidden_state_after_each_continuation_of_each_part(self):
        cases = [
            (
                "one_part",
                ["It rained.\n"],
                [["wet", "home"]],
                [[[-0.1351, -1.6366, -1.0490, -0.4350], [1.0009, 0.1602, -1.4286, 0.9532]]],
            ),
            (
                "the_second_part_reads_on_from_the_first",
                ["It rained.\n", "Ann left.\n"],
                [["home"], ["home"]],
                [[[1.0009, 0.1602, -1.4286, 0.9532]], [[1.0284, 0.1110, -1.4578, 0.8831]]],
            ),
            (
                "a_part_without_continuations",
                ["It rained.\n", "Ann left.\n"],
                [[], ["home"]],
                [[], [[1.0284, 0.1110, -1.4578, 0.8831]]],
            ),
            (
                "no_parts",
                [],
                [],
                [],
            ),
        ]
        deployment = self.build(
            {"It rained.\n": [5, 6, 7], "Ann left.\n": [8, 9], "wet": [10, 11], "home": [12]}
        )
        for name, parts, continuations_of_each_part, expected in cases:
            with self.subTest(name):
                response = await deployment.last_hidden_states(
                    {"parts": parts, "continuations_of_each_part": continuations_of_each_part}
                )
                lines = [line async for line in response.body_iterator]
                answered = [json.loads(line)["last_hidden_states"] for line in lines]

                torch.testing.assert_close(answered, expected, atol=1e-4, rtol=0)
