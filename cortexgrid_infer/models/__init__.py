"""The serve apps cortexgrid runs, one per task, each working with a model of
that task whichever source it came from."""

from cortexgrid_infer.models.anthropic import AnthropicText2Text
from cortexgrid_infer.models.base import HostedModel, LocalModel, Weights
from cortexgrid_infer.models.image2mesh import Image2Mesh
from cortexgrid_infer.models.text2image import Text2Image
from cortexgrid_infer.models.text2text import Text2Text

__all__ = [
    "LocalModel",
    "HostedModel",
    "Weights",
    "Text2Text",
    "Text2Image",
    "Image2Mesh",
    "AnthropicText2Text",
]
