"""The wire protocol between a cortexgrid-served rewriting app and its client.

A rewriting serve app answers `POST /rewrite`: the request carries the text to
rewrite under `text`, plus whatever generation options the model takes
(`max_new_tokens`, `num_beams`, say) as further fields. The reply carries the
rewritten text under `text`.

Rewriting is one text in, one text out - correcting grammar, paraphrasing,
summarising, translating - with no conversation, tools or streaming, which is
what an encoder-decoder model does. One client, `ServedRewritingModel`, serves
every model of the task whoever made it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from cortexgrid_infer.core import RewritingModel


@dataclass
class ServedRewritingModel(RewritingModel):
    """The client of any rewriting serve app deployed at `url`."""

    url: str
    model_id: str

    @property
    def name(self) -> str:
        return self.model_id

    async def rewrite(self, text: str, **generation_options: Any) -> str:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{self.url}/rewrite", json={"text": text, **generation_options}
            )
            response.raise_for_status()
            return response.json()["text"]
