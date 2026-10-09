"""Tests for cortexgrid_infer.serve_apps.anthropic, registered as a Hosted entry."""

from __future__ import annotations

import unittest
from unittest import mock

import cortexgrid

from cortexgrid_infer.registry import Hosted
from cortexgrid_infer.serve_apps.anthropic import (
    AnthropicText2Text,
    to_anthropic_messages,
    to_anthropic_tools,
)

SERVE = "cortexgrid_infer.serve_apps.anthropic"


class TestHostedAnthropic(unittest.TestCase):
    def test_names_the_model_for_the_registry(self):
        entry = Hosted("claude-sonnet-5", AnthropicText2Text)
        self.assertEqual((entry.family, entry.suffix), ("claude-sonnet", "5"))

    def test_client_calls_the_complete_endpoint(self):
        deployment = cortexgrid.Deployment(
            key=cortexgrid.DeploymentKey("claude-sonnet", "5", "imported"),
            config={},
            url="http://h/r/F/S/R",
            phase="running",
            bundle_fingerprint="",
            replaced_bundle_fingerprint="",
            experiment_name="",
            class_import_path="cortexgrid_infer.serve_apps.anthropic:AnthropicText2Text",
        )

        model = AnthropicText2Text.client(deployment)

        self.assertIs(type(model), AnthropicText2Text.client)
        self.assertEqual(model.key, cortexgrid.DeploymentKey("claude-sonnet", "5", "imported"))
        self.assertEqual(model.url, "http://h/r/F/S/R")

    def test_asks_for_no_hardware(self):
        self.assertEqual(
            Hosted("claude-sonnet-5", AnthropicText2Text).requirements(),
            cortexgrid.ModelRequirements(),
        )

    def test_config_carries_what_the_deployment_needs_to_call_out(self):
        self.assertEqual(
            Hosted("claude-sonnet-5", AnthropicText2Text).config(),
            {"model": "claude-sonnet-5", "api_key_secret": "ANTHROPIC_API_KEY"},
        )

    def test_config_stores_the_secret_name_never_the_key(self):
        # Registry entries are readable by anyone who can see the model, so the
        # entry names a cortexgrid secret and the deployment resolves it.
        config = Hosted(
            "claude-sonnet-5", AnthropicText2Text, api_key_secret="TEAM_KEY"
        ).config()

        self.assertEqual(config["api_key_secret"], "TEAM_KEY")


class TestAnthropicText2TextConfig(unittest.TestCase):
    @mock.patch(f"{SERVE}.AsyncAnthropic")
    @mock.patch(f"{SERVE}.cortexgrid")
    def test_reads_its_settings_from_the_entry_it_was_built_for(
        self, mock_cortexgrid: mock.Mock, mock_client: mock.Mock
    ):
        mock_cortexgrid.model_config.return_value = {
            "model": "claude-sonnet-5",
            "api_key_secret": "TEAM_KEY",
        }
        mock_cortexgrid.get_secret.return_value = "sk-ant-xxx"

        key = cortexgrid.DeploymentKey("claude-sonnet", "5", "imported")

        deployment = AnthropicText2Text(key)

        mock_cortexgrid.model_config.assert_called_once_with(key)
        mock_cortexgrid.get_secret.assert_called_once_with("TEAM_KEY")
        mock_client.assert_called_once_with(api_key="sk-ant-xxx")
        self.assertEqual(deployment._model, "claude-sonnet-5")

    @mock.patch(f"{SERVE}.AsyncAnthropic")
    @mock.patch(f"{SERVE}.cortexgrid")
    def test_says_what_is_missing_from_the_entry(
        self, mock_cortexgrid: mock.Mock, _mock_client: mock.Mock
    ):
        # An entry edited on the model card can lose a key; failing at
        # construction names it rather than 401-ing on the first request.
        mock_cortexgrid.model_config.return_value = {"model": "claude-sonnet-5"}

        with self.assertRaises(RuntimeError) as caught:
            AnthropicText2Text(
                cortexgrid.DeploymentKey("claude-sonnet", "5", "imported")
            )

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


class TestAnthropicText2TextHiddenStates(unittest.TestCase):
    def test_its_client_offers_no_hidden_states(self):
        self.assertFalse(hasattr(AnthropicText2Text.client, "hidden_states_at_layer"))
