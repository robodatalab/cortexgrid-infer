"""What every Gemini serve app shares: which model to call, and the key to call
it with.

A Gemini app loads no weights and needs no GPU, so there is nothing to import:
it is registered with `registry.Hosted`. What it needs instead comes from the
registry entry's `config`, set when the model was registered and editable on its
model card afterwards.
"""

from __future__ import annotations

import cortexgrid
from google import genai

from cortexgrid_infer.serve_apps.base import HostedModel

# Keys the registry entry must carry for the deployment to reach the API.
MODEL_PARAM = "model"
API_KEY_SECRET_PARAM = "api_key_secret"

# The cortexgrid secret the deployment reads its API key from. It is the secret's
# name that is stored on the registry entry, never the key itself: entries are
# readable by anyone who can see the model.
DEFAULT_API_KEY_SECRET = "GEMINI_API_KEY"


class GeminiModel(HostedModel):
    """Base of the serve apps that forward to the Gemini API, one per task."""

    @classmethod
    def config(
        cls, model_id: str, api_key_secret: str = DEFAULT_API_KEY_SECRET
    ) -> dict[str, str]:
        """`model_id` is the name Gemini knows the model by, and is what the
        deployment sends upstream; `api_key_secret` names the cortexgrid secret
        holding the key to send it with."""
        return {MODEL_PARAM: model_id, API_KEY_SECRET_PARAM: api_key_secret}

    def __init__(self, family: str, suffix: str, run_name: str) -> None:
        config = cortexgrid.model_config(family, suffix, run_name)
        missing = {MODEL_PARAM, API_KEY_SECRET_PARAM} - config.keys()
        if missing:
            raise RuntimeError(
                f"{family}/{suffix}/{run_name} is missing {sorted(missing)} from "
                "its config; set them on the model card"
            )
        self._config = config
        self._model = config[MODEL_PARAM]
        self._client = genai.Client(
            api_key=cortexgrid.get_secret(config[API_KEY_SECRET_PARAM])
        )
