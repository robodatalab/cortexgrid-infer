"""The serve apps cortexgrid runs, one per task, each working with a model of
that task whichever source it came from."""

from cortexgrid_infer.serve_apps.anthropic import AnthropicText2Text
from cortexgrid_infer.serve_apps.base import HostedModel, LocalModel, Weights
from cortexgrid_infer.serve_apps.gemini import GeminiText2Image, GeminiText2Text
from cortexgrid_infer.serve_apps.image2mesh import Image2Mesh
from cortexgrid_infer.serve_apps.text2image import Text2Image
from cortexgrid_infer.serve_apps.text2text import Text2Text
from cortexgrid_infer.serve_apps.text_rewriter import TextRewriter

__all__ = [
    "LocalModel",
    "HostedModel",
    "Weights",
    "Text2Text",
    "Text2Image",
    "Image2Mesh",
    "TextRewriter",
    "AnthropicText2Text",
    "GeminiText2Text",
    "GeminiText2Image",
]
