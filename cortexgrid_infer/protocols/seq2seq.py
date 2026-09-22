from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from cortexgrid_infer.core import DeployedModel


@dataclass
class ServedSeq2SeqModel(DeployedModel):
    url: str
    model_id: str

    @property
    def name(self) -> str:
        return self.model_id

    async def generate(self, text: str, **generation_options: Any) -> str:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{self.url}/generate", json={"text": text, **generation_options}
            )
            response.raise_for_status()
            return response.json()["text"]
