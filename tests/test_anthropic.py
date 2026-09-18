"""Tests for cortexgrid_infer.providers.anthropic and its serve app."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from cortexgrid_infer.completion import ServedCompletingModel
from cortexgrid_infer.providers.anthropic import AnthropicImport
from cortexgrid_infer.providers.anthropic_serve import (
    AnthropicDeployment,
    to_anthropic_messages,
    to_anthropic_tools,
)


class TestAnthropicImport(unittest.TestCase):
    def test_names_the_model_for_the_registry(self):
        imp = AnthropicImport("claude-sonnet-5")
        self.assertEqual((imp.family, imp.suffix), ("claude-sonnet", "5"))

    def test_bundles_the_forwarding_serve_app(self):
        self.assertIs(AnthropicImport("claude-sonnet-5").serve_app, AnthropicDeployment)

    def test_client_is_the_shared_completion_client(self):
        # The serve app speaks the same /complete protocol as a HuggingFace one,
        # so there is nothing Anthropic-specific left on the client side.
        model = AnthropicImport("claude-sonnet-5").client("http://h/r/F/S/R")
        self.assertIsInstance(model, ServedCompletingModel)
        self.assertEqual(model.name, "claude-sonnet-5")

    @mock.patch("cortexgrid_infer.providers.anthropic.cortexgrid.ModelRequirements")
    def test_asks_for_no_hardware(self, mock_requirements: mock.Mock):
        # A replica holds no weights and does no compute, so it should be
        # placeable on a CPU-only node.
        AnthropicImport("claude-sonnet-5").requirements()

        self.assertEqual(mock_requirements.call_args.kwargs.get("num_gpus", 0), 0)

    @mock.patch("cortexgrid_infer.providers.anthropic.cortexgrid.ModelRequirements")
    def test_carries_what_the_deployment_needs_to_call_out(
        self, mock_requirements: mock.Mock
    ):
        AnthropicImport("claude-sonnet-5").requirements()

        self.assertEqual(
            mock_requirements.call_args.kwargs["params"],
            {"model": "claude-sonnet-5", "api_key_secret": "ANTHROPIC_API_KEY"},
        )

    @mock.patch("cortexgrid_infer.providers.anthropic.cortexgrid.ModelRequirements")
    def test_stores_the_secret_name_never_the_key(self, mock_requirements: mock.Mock):
        # Registry entries are readable by anyone who can see the model, so the
        # entry names a cortexgrid secret and the deployment resolves it.
        AnthropicImport("claude-sonnet-5", api_key_secret="TEAM_KEY").requirements()

        self.assertEqual(
            mock_requirements.call_args.kwargs["params"]["api_key_secret"], "TEAM_KEY"
        )


def _saved(params: dict[str, str] | None) -> SimpleNamespace:
    return SimpleNamespace(requirements=SimpleNamespace(params=params or {}))


class TestAnthropicDeploymentConfig(unittest.TestCase):
    @mock.patch("cortexgrid_infer.providers.anthropic_serve.AsyncAnthropic")
    @mock.patch("cortexgrid_infer.providers.anthropic_serve.cortexgrid")
    def test_resolves_the_key_from_the_named_secret(
        self, mock_cortexgrid: mock.Mock, mock_client: mock.Mock
    ):
        mock_cortexgrid.model_registry_status.return_value = _saved(
            {"model": "claude-sonnet-5", "api_key_secret": "TEAM_KEY"}
        )
        mock_cortexgrid.get_secret.return_value = "sk-ant-xxx"

        AnthropicDeployment("claude-sonnet", "5", "imported")

        mock_cortexgrid.get_secret.assert_called_once_with("TEAM_KEY")
        mock_client.assert_called_once_with(api_key="sk-ant-xxx")

    @mock.patch("cortexgrid_infer.providers.anthropic_serve.AsyncAnthropic")
    @mock.patch("cortexgrid_infer.providers.anthropic_serve.cortexgrid")
    def test_unregistered_model_is_an_error(
        self, mock_cortexgrid: mock.Mock, _mock_client: mock.Mock
    ):
        mock_cortexgrid.model_registry_status.return_value = None

        with self.assertRaises(RuntimeError):
            AnthropicDeployment("claude-sonnet", "5", "imported")

    @mock.patch("cortexgrid_infer.providers.anthropic_serve.AsyncAnthropic")
    @mock.patch("cortexgrid_infer.providers.anthropic_serve.cortexgrid")
    def test_says_what_is_missing_from_the_entry(
        self, mock_cortexgrid: mock.Mock, _mock_client: mock.Mock
    ):
        # An entry edited on the model card can lose a parameter; failing at
        # construction names it rather than 401-ing on the first request.
        mock_cortexgrid.model_registry_status.return_value = _saved(
            {"model": "claude-sonnet-5"}
        )

        with self.assertRaises(RuntimeError) as caught:
            AnthropicDeployment("claude-sonnet", "5", "imported")

        self.assertIn("api_key_secret", str(caught.exception))


class TestMessageConversion(unittest.TestCase):
    def test_system_message_is_lifted_out(self):
        system, messages = to_anthropic_messages(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ]
        )

        self.assertEqual(system, "be terse")
        self.assertEqual(messages, [{"role": "user", "content": "hi"}])

    def test_assistant_tool_calls_become_tool_use_blocks(self):
        _system, messages = to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "add", "arguments": {"a": 1}},
                        }
                    ],
                }
            ]
        )

        self.assertEqual(
            messages[0]["content"],
            [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "call_1", "name": "add", "input": {"a": 1}},
            ],
        )

    def test_consecutive_tool_results_share_one_user_turn(self):
        # Anthropic expects the results of a parallel tool call in a single user
        # message, not one message each.
        _system, messages = to_anthropic_messages(
            [
                {"role": "tool", "tool_call_id": "call_1", "content": "3"},
                {"role": "tool", "tool_call_id": "call_2", "content": "4"},
            ]
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(len(messages[0]["content"]), 2)

    def test_tool_result_after_text_starts_a_new_turn(self):
        _system, messages = to_anthropic_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "tool", "tool_call_id": "call_1", "content": "3"},
            ]
        )

        self.assertEqual(len(messages), 2)


class TestToolConversion(unittest.TestCase):
    def test_no_tools_stays_none(self):
        self.assertIsNone(to_anthropic_tools(None))
        self.assertIsNone(to_anthropic_tools([]))

    def test_openai_spec_becomes_an_input_schema(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}

        result = to_anthropic_tools(
            [{"function": {"name": "add", "description": "sum", "parameters": schema}}]
        )

        self.assertEqual(
            result, [{"name": "add", "description": "sum", "input_schema": schema}]
        )

    def test_a_tool_without_parameters_gets_an_empty_schema(self):
        result = to_anthropic_tools([{"function": {"name": "now"}}])

        self.assertEqual(
            result[0]["input_schema"], {"type": "object", "properties": {}}
        )
