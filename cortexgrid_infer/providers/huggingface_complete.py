"""HuggingFace causal-LM provider: what cortexgrid needs to serve one."""

from __future__ import annotations

from cortexgrid_infer.completion import ServedCompletingModel
from cortexgrid_infer.importing import HuggingFaceImport
from cortexgrid_infer.providers.huggingface_complete_serve import (
    MAX_INPUT_TOKENS_PARAM,
    HuggingFaceCompletingDeployment,
)


class HuggingFaceCompletingImport(HuggingFaceImport):
    """What `cortexgrid.import_model` needs to take a HuggingFace causal LM.

    The serve app loads the weights with `AutoModelForCausalLM` and streams from
    `/complete`. An instruct-tuned model emits tool calls as text in the form
    `ServedCompletingModel` parses, so the serve app forwards its tokens
    untouched and needs no encoding step.

    `max_input_tokens` is the longest prompt the deployment accepts; anything
    longer is truncated so the KV cache is never resized (see the serve app).
    Given here it seeds the model card, and can be changed on the card
    afterwards without re-importing."""

    serve_app = HuggingFaceCompletingDeployment

    def __init__(
        self,
        hf_id: str,
        token: str | None = None,
        ignore_patterns: list[str] | None = None,
        max_input_tokens: int | None = None,
    ) -> None:
        super().__init__(hf_id, token, ignore_patterns)
        if max_input_tokens is not None and max_input_tokens <= 0:
            raise ValueError(
                f"max_input_tokens must be a positive number of tokens, "
                f"got {max_input_tokens}"
            )
        self.max_input_tokens = max_input_tokens

    def config(self) -> dict[str, str]:
        """The cache length to pin, when the caller chose one.

        Reaches the card through `import_model(config=...)`. Saying nothing
        leaves the serve app on its default, and never overwrites a value
        already on the card."""
        if self.max_input_tokens is None:
            return {}
        return {MAX_INPUT_TOKENS_PARAM: str(self.max_input_tokens)}

    def client(self, url: str) -> ServedCompletingModel:
        return ServedCompletingModel(url=url, model_id=self.hf_id)
