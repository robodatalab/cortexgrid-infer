"""Tests for cortexgrid_infer.serve_apps.gemini.base: the config every Gemini serve
app registers with, and reads back when it is deployed."""

from __future__ import annotations

import unittest
from unittest import mock

from cortexgrid_infer.registry import Hosted
from cortexgrid_infer.serve_apps.gemini.base import GeminiModel

BASE = "cortexgrid_infer.serve_apps.gemini.base"


class TestConfig(unittest.TestCase):
    def test_config_carries_what_the_deployment_needs_to_call_out(self):
        self.assertEqual(
            Hosted("gemini-2.5-flash", GeminiModel).config(),
            {"model": "gemini-2.5-flash", "api_key_secret": "GEMINI_API_KEY"},
        )

    def test_config_stores_the_secret_name_never_the_key(self):
        # Registry entries are readable by anyone who can see the model, so the
        # entry names a cortexgrid secret and the deployment resolves it.
        config = Hosted("gemini-2.5-pro", GeminiModel, api_key_secret="TEAM_KEY").config()

        self.assertEqual(config["api_key_secret"], "TEAM_KEY")


class TestConstruction(unittest.TestCase):
    @mock.patch(f"{BASE}.genai")
    @mock.patch(f"{BASE}.cortexgrid")
    def test_reads_its_settings_from_the_entry_it_was_built_for(
        self, mock_cortexgrid: mock.Mock, mock_genai: mock.Mock
    ):
        mock_cortexgrid.model_config.return_value = {
            "model": "gemini-2.5-pro",
            "api_key_secret": "TEAM_KEY",
        }
        mock_cortexgrid.get_secret.return_value = "AIza-xxx"

        deployment = GeminiModel("gemini-2.5", "pro", "imported")

        mock_cortexgrid.model_config.assert_called_once_with(
            "gemini-2.5", "pro", "imported"
        )
        mock_cortexgrid.get_secret.assert_called_once_with("TEAM_KEY")
        mock_genai.Client.assert_called_once_with(api_key="AIza-xxx")
        self.assertEqual(deployment._model, "gemini-2.5-pro")

    @mock.patch(f"{BASE}.genai")
    @mock.patch(f"{BASE}.cortexgrid")
    def test_says_what_is_missing_from_the_entry(
        self, mock_cortexgrid: mock.Mock, _mock_genai: mock.Mock
    ):
        # An entry edited on the model card can lose a key; failing at
        # construction names it rather than 401-ing on the first request.
        mock_cortexgrid.model_config.return_value = {"model": "gemini-2.5-pro"}

        with self.assertRaises(RuntimeError) as caught:
            GeminiModel("gemini-2.5", "pro", "imported")

        self.assertIn("api_key_secret", str(caught.exception))
