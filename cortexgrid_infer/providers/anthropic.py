"""Anthropic provider: a registry entry for a model whose weights aren't ours.

Anthropic models reach the cluster the same way HuggingFace ones do - imported
into the registry, deployed as a Serve app, reached through a client over its
URL. The only difference is that there is nothing to upload, so the import
registers the entry with no source and the serve app forwards to the API.
"""

from __future__ import annotations

import cortexgrid

from cortexgrid_infer.completion import ServedCompletingModel
from cortexgrid_infer.importing import ModelImport, split_model_id
from cortexgrid_infer.providers.anthropic_serve import (
    API_KEY_SECRET_PARAM,
    MODEL_PARAM,
    AnthropicDeployment,
)

# The cortexgrid secret the deployment reads its API key from. It is the secret's
# name that is stored on the registry entry, never the key itself: entries are
# readable by anyone who can see the model.
DEFAULT_API_KEY_SECRET = "ANTHROPIC_API_KEY"


class AnthropicImport(ModelImport):
    """What `cortexgrid.import_model` needs to register an Anthropic model.

    There are no weights, so it is imported with no source:

        imp = AnthropicImport("claude-sonnet-5")
        cortexgrid.import_model(
            None, imp.serve_app,
            family=imp.family, suffix=imp.suffix,
            requirements=imp.requirements(),
        )

    `model_id` is the name Anthropic knows the model by, and is what the
    deployment sends upstream; `api_key_secret` names the cortexgrid secret
    holding the key to send it with."""

    serve_app = AnthropicDeployment

    def __init__(
        self, model_id: str, api_key_secret: str = DEFAULT_API_KEY_SECRET
    ) -> None:
        self.model_id = model_id
        self.api_key_secret = api_key_secret
        self.family, self.suffix = split_model_id(model_id)

    def requirements(self) -> cortexgrid.ModelRequirements:
        """No hardware, and the two parameters the deployment needs to call out.

        A replica holds no weights and does no compute of its own, so it asks
        for nothing and is placed on any node, CPU-only included."""
        # `params` awaits the cortexgrid change that lets a registry entry carry
        # arbitrary values; until it lands this is the one thing here that does
        # not typecheck.
        return cortexgrid.ModelRequirements(  # type: ignore[call-arg]
            params={
                MODEL_PARAM: self.model_id,
                API_KEY_SECRET_PARAM: self.api_key_secret,
            }
        )

    def client(self, url: str) -> ServedCompletingModel:
        return ServedCompletingModel(url=url, model_id=self.model_id)
