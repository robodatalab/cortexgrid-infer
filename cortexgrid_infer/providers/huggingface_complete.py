"""HuggingFace causal-LM provider: what cortexgrid needs to serve one."""

from __future__ import annotations

from cortexgrid_infer.completion import ServedCompletingModel
from cortexgrid_infer.importing import HuggingFaceImport
from cortexgrid_infer.providers.huggingface_complete_serve import (
    HuggingFaceCompletingDeployment,
)


class HuggingFaceCompletingImport(HuggingFaceImport):
    """What `cortexgrid.import_model` needs to take a HuggingFace causal LM.

    The serve app loads the weights with `AutoModelForCausalLM` and streams from
    `/complete`. An instruct-tuned model emits tool calls as text in the form
    `ServedCompletingModel` parses, so the serve app forwards its tokens
    untouched and needs no encoding step."""

    serve_app = HuggingFaceCompletingDeployment

    def client(self, url: str) -> ServedCompletingModel:
        return ServedCompletingModel(url=url, model_id=self.hf_id)
