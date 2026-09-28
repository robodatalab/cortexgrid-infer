"""Tests for cortexgrid_infer.serve_apps.text2text."""

from __future__ import annotations

import json
from types import SimpleNamespace

import cortexgrid
import torch
import unittest
from unittest import mock
from transformers import Qwen3Config, Qwen3ForCausalLM

from cortexgrid_infer.protocols.completion import ServedCompletingModel
from cortexgrid_infer.serve_apps.text2text import (
    PrefixCache,
    Text2Text,
    allowed_next_tokens,
    continuation_loglikelihood,
    hidden_state_after,
    scaled_RoPE,
    shared_length,
)


class TestText2TextForTheImporter(unittest.TestCase):
    def test_client_speaks_the_deployed_app(self):
        model = Text2Text.client("http://h/r/F/S/R", "Qwen/Qwen2-2.5B-Instruct")
        self.assertIsInstance(model, ServedCompletingModel)
        self.assertEqual(model.url, "http://h/r/F/S/R")
        self.assertEqual(model.name, "Qwen/Qwen2-2.5B-Instruct")

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
             mock.patch(f"{self.SERVE}.AutoConfig.from_pretrained"), \
             mock.patch(f"{self.SERVE}.PrefixCache"), \
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
             mock.patch(f"{self.SERVE}.AutoConfig.from_pretrained"), \
             mock.patch(f"{self.SERVE}.PrefixCache"), \
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
             mock.patch(f"{self.SERVE}.AutoConfig.from_pretrained"), \
             mock.patch(f"{self.SERVE}.PrefixCache"), \
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


class TestSharedLength(unittest.TestCase):
    def test_counts_the_tokens_both_sequences_start_with(self):
        cases = [
            ("identical", [1, 2, 3], [1, 2, 3], 3),
            ("the_second_goes_on", [1, 2, 3], [1, 2, 3, 4, 5], 3),
            ("the_second_stops_early", [1, 2, 3, 4], [1, 2], 2),
            ("they_part_in_the_middle", [1, 2, 3, 4], [1, 2, 9, 4], 2),
            ("they_part_at_once", [1, 2, 3], [7, 2, 3], 0),
            ("the_first_is_empty", [], [1, 2], 0),
            ("both_are_empty", [], [], 0),
        ]
        for name, first, second, expected_length in cases:
            with self.subTest(name):
                self.assertEqual(shared_length(first, second), expected_length)


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


class TestPrefixCache(unittest.TestCase):
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

    def test_holds_exactly_the_last_prefix_it_read(self):
        cases = [
            ("one_prefix", [[1, 2, 3]], 3),
            ("a_longer_prefix_after", [[1, 2, 3], [1, 2, 3, 4, 5]], 5),
            ("a_shorter_prefix_after", [[1, 2, 3, 4], [1, 2]], 2),
            ("a_different_prefix_after", [[1, 2, 3], [7, 8]], 2),
            ("the_same_prefix_twice", [[1, 2, 3], [1, 2, 3]], 3),
            ("a_prefix_parting_in_the_middle", [[1, 2, 3, 4], [1, 2, 9, 9, 9]], 5),
            ("nothing", [[]], 0),
        ]
        for name, prefixes, expected_length in cases:
            with self.subTest(name):
                prefix_cache = PrefixCache(self.model)

                for prefix in prefixes:
                    prefix_cache.read(prefix)

                self.assertEqual(prefix_cache.length, expected_length)

    def test_reading_past_the_prefix_leaves_the_prefix(self):
        cases = [
            ("one_token", [1, 2, 3], [4], 3),
            ("several_tokens", [1, 2, 3], [4, 5, 6, 7], 3),
            ("more_than_the_prefix", [1], [2, 3, 4, 5, 6], 1),
        ]
        for name, prefix, past_it, expected_length in cases:
            with self.subTest(name):
                prefix_cache = PrefixCache(self.model)
                prefix_cache.read(prefix)

                prefix_cache.read_past_the_prefix(past_it)

                self.assertEqual(prefix_cache.length, expected_length)

    def test_a_continuation_s_loglikelihood_is_the_full_sequence_s(self):
        cases = [
            ("one_token", [], [1, 2, 3, 4], [5]),
            ("several_tokens", [], [1, 2, 3, 4], [5, 6, 7]),
            ("after_reading_the_same_prompt", [[1, 2, 3, 4]], [1, 2, 3, 4], [5, 6]),
            ("after_reading_a_longer_prompt", [[1, 2, 3, 4, 8, 9]], [1, 2, 3, 4], [5, 6]),
            ("after_reading_a_different_prompt", [[9, 9, 9]], [1, 2, 3, 4], [5, 6]),
            ("a_one_token_prompt", [], [1], [5, 6]),
        ]
        for name, read_before, prompt, continuation in cases:
            with self.subTest(name):
                prefix_cache = PrefixCache(self.model)
                for earlier in read_before:
                    prefix_cache.read(earlier)

                loglikelihood = continuation_loglikelihood(prefix_cache, prompt, continuation)

                with torch.inference_mode():
                    logits = self.model(input_ids=torch.tensor([prompt + continuation])).logits[0]
                log_probabilities = torch.log_softmax(logits.float(), dim=-1)
                predicting = torch.arange(len(prompt) - 1, len(prompt) + len(continuation) - 1)
                expected = float(log_probabilities[predicting, torch.tensor(continuation)].sum())
                self.assertAlmostEqual(loglikelihood, expected, places=4)

    def test_a_probe_s_hidden_state_is_the_full_sequence_s(self):
        cases = [
            ("the_last_layer", [], [1, 2, 3], [4, 5], -1),
            ("the_embeddings", [], [1, 2, 3], [4, 5], 0),
            ("a_middle_layer", [], [1, 2, 3], [4, 5], 1),
            ("a_one_token_probe", [], [1, 2, 3], [4], -1),
            ("after_reading_a_shorter_prefix", [[1, 2]], [1, 2, 3], [4, 5], -1),
            ("after_reading_another_probe", [[1, 2, 3]], [1, 2, 3], [6, 7, 8], -1),
        ]
        for name, read_before, prefix, probe, layer in cases:
            with self.subTest(name):
                prefix_cache = PrefixCache(self.model)
                for earlier in read_before:
                    prefix_cache.read(earlier)

                hidden_state = hidden_state_after(prefix_cache, prefix, probe, layer)

                with torch.inference_mode():
                    read_in_full = self.model(
                        input_ids=torch.tensor([prefix + probe]), output_hidden_states=True
                    ).hidden_states[layer][0, -1]
                torch.testing.assert_close(hidden_state, read_in_full, atol=1e-4, rtol=1e-4)


class TestScaledRoPE(unittest.TestCase):
    def test_lays_the_scaling_over_the_RoPE_and_stretches_the_context(self):
        cases = [
            (
                "yarn_from_the_original_context",
                SimpleNamespace(model_type="qwen3", max_position_embeddings=40960, rope_parameters={"rope_type": "default", "rope_theta": 1000000.0}),
                {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
                {"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 4.0, "original_max_position_embeddings": 32768},
                131072,
            ),
            (
                "linear_from_the_model_s_own_context",
                SimpleNamespace(model_type="llama", max_position_embeddings=4096, rope_parameters={"rope_type": "default", "rope_theta": 10000.0}),
                {"rope_type": "linear", "factor": 2.0},
                {"rope_type": "linear", "rope_theta": 10000.0, "factor": 2.0},
                8192,
            ),
            (
                "llama3_with_its_own_frequencies",
                SimpleNamespace(model_type="llama", max_position_embeddings=8192, rope_parameters={"rope_type": "default", "rope_theta": 500000.0}),
                {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
                {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
                65536,
            ),
            (
                "a_fractional_factor",
                SimpleNamespace(model_type="qwen3", max_position_embeddings=32768, rope_parameters={"rope_type": "default", "rope_theta": 1000000.0}),
                {"rope_type": "yarn", "factor": 1.5, "original_max_position_embeddings": 32768},
                {"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 1.5, "original_max_position_embeddings": 32768},
                49152,
            ),
        ]
        for name, config, scaling, expected_RoPE, expected_context in cases:
            with self.subTest(name):
                scaled = scaled_RoPE(config, scaling)

                self.assertEqual(scaled.rope_parameters, expected_RoPE)
                self.assertEqual(scaled.max_position_embeddings, expected_context)

    def test_refuses_a_model_without_RoPE(self):
        cases = [
            ("no_RoPE_parameters_at_all", SimpleNamespace(model_type="gpt2", max_position_embeddings=1024)),
            ("RoPE_parameters_left_empty", SimpleNamespace(model_type="gpt2", max_position_embeddings=1024, rope_parameters=None)),
        ]
        for name, config in cases:
            with self.subTest(name), self.assertRaises(ValueError):
                scaled_RoPE(config, {"rope_type": "linear", "factor": 2.0})


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
        config = SimpleNamespace(
            model_type="qwen3",
            max_position_embeddings=40960,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        )
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config", return_value=card), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained"), \
             mock.patch(f"{self.SERVE}.AutoConfig.from_pretrained", return_value=config), \
             mock.patch(f"{self.SERVE}.PrefixCache"), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained",
                        return_value=_FakeCausalLM()) as loads, \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device("cpu")):
            Text2Text(cortexgrid.DeploymentKey("family", "suffix", "imported"))
        return loads

    def test_loads_the_weights_with_the_scaled_RoPE_the_card_asks_for(self):
        loads = self.load(
            {
                "rope_scaling": json.dumps(
                    {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
                )
            }
        )

        loaded_with = loads.call_args.kwargs["config"]
        self.assertEqual(
            loaded_with.rope_parameters,
            {"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 4.0, "original_max_position_embeddings": 32768},
        )
        self.assertEqual(loaded_with.max_position_embeddings, 131072)

    def test_loads_the_config_as_it_is_when_the_card_asks_for_no_scaling(self):
        loaded_with = self.load({}).call_args.kwargs["config"]

        self.assertEqual(loaded_with.rope_parameters, {"rope_type": "default", "rope_theta": 1000000.0})
        self.assertEqual(loaded_with.max_position_embeddings, 40960)

    def test_loads_the_weights_in_the_dtype_the_card_names(self):
        cases = [
            ("float16_unless_the_card_says", {}, torch.float16),
            ("bfloat16", {"dtype": "bfloat16"}, torch.bfloat16),
            ("float32", {"dtype": "float32"}, torch.float32),
        ]
        for name, card, expected_dtype in cases:
            with self.subTest(name):
                self.assertEqual(self.load(card).call_args.kwargs["torch_dtype"], expected_dtype)


class TestText2TextReadsTheModel(unittest.TestCase):
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
             mock.patch(f"{self.SERVE}.AutoConfig.from_pretrained", return_value=model.config), \
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

    def test_returns_one_vector_of_the_model_s_width_per_probe(self):
        cases = [
            ("two_probes", ["a probe", "another"], [4, 4]),
            ("one_probe", ["another"], [4]),
            ("no_probes", [], []),
        ]
        deployment = self.build({"The story.": [5, 6, 7], "a probe": [8, 9], "another": [10]})
        for name, probes, expected_widths in cases:
            with self.subTest(name):
                vectors = deployment._hidden_states("The story.", probes, -1)

                self.assertEqual([len(vector) for vector in vectors], expected_widths)
