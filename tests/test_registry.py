"""Tests for cortexgrid_infer.registry."""

from __future__ import annotations

import unittest

import cortexgrid

from cortexgrid_infer.registry import Hosted, split_model_id
from cortexgrid_infer.serve_apps.base import HostedModel


class _Forwarder(HostedModel):
    @classmethod
    def config(cls, model_id: str, region: str = "eu") -> dict[str, str]:
        return {"model": model_id, "region": region}


class TestSplitModelId(unittest.TestCase):
    def test_org_and_dash(self):
        self.assertEqual(
            split_model_id("Qwen/Qwen2-2.5B-Instruct"), ("Qwen2-2.5B", "Instruct")
        )

    def test_org_no_dash(self):
        self.assertEqual(split_model_id("openai/gpt2"), ("gpt2", "base"))

    def test_no_org_no_dash(self):
        self.assertEqual(split_model_id("gpt2"), ("gpt2", "base"))

    def test_no_org_with_dash(self):
        self.assertEqual(split_model_id("model-v1"), ("model", "v1"))

    def test_multiple_dashes_split_on_last(self):
        self.assertEqual(split_model_id("a/b-c-d-e"), ("b-c-d", "e"))


class TestHosted(unittest.TestCase):
    def test_names_the_model_for_the_registry(self):
        entry = Hosted("fwd-large", _Forwarder)
        self.assertEqual((entry.family, entry.suffix), ("fwd", "large"))

    def test_bundles_the_forwarding_serve_app(self):
        self.assertIs(Hosted("fwd-large", _Forwarder).serve_app, _Forwarder)

    def test_asks_for_what_the_serve_app_asks_for(self):
        self.assertEqual(
            Hosted("fwd-large", _Forwarder).requirements(), cortexgrid.ModelRequirements()
        )

    def test_config_is_the_serve_app_s_for_the_model(self):
        self.assertEqual(
            Hosted("fwd-large", _Forwarder, region="us").config(),
            {"model": "fwd-large", "region": "us"},
        )

    def test_a_setting_the_serve_app_does_not_take_fails_at_construction(self):
        # Before anything reaches the registry, rather than on the first deploy.
        with self.assertRaises(TypeError):
            Hosted("fwd-large", _Forwarder, regoin="us")
