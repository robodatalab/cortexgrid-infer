"""Run inference on models the cortexgrid cluster serves.

cortexgrid owns a model's life - registry identity, weights, serve bundle,
hardware placement, deployment, jobs - and knows nothing about what kind of
model it holds. This library supplies the knowledge those calls need and
nothing else, split along the two things a model is:

- where its weights come from - an importer per source (`importers`), which
  names, fetches and sizes them;
- what runs it - a serve app per task (`serve_apps`), which loads the weights,
  answers the task's routes (`protocols`), and names the client that speaks
  them.

    imp = cortexgrid_infer.HuggingFaceImporter("Qwen/Qwen2.5-0.5B-Instruct", cortexgrid_infer.Text2Text)
    with imp:
        cortexgrid.import_model(
            imp.source, imp.serve_app,
            family=imp.family, suffix=imp.suffix,
            requirements=imp.requirements(),
        )
    deployment = cortexgrid.deploy_model(imp.family, imp.suffix, cortexgrid.IMPORTED, wait=True)
    model = imp.client(deployment.url)

A model hosted elsewhere has no weights and so no importer: its serve app
forwards to it, and a `Hosted` entry registers it.

See `examples/deploy_text_model` for the whole lifecycle, import job included.
"""

from cortexgrid import ModelDeployFailed

from cortexgrid_infer.core import (
    CompletionChunk,
    DeployedModel,
    CompletingModel,
    GeneratedImage,
    GeneratedMesh,
    GeneratingModel,
    MeshingModel,
    Message,
    Tool,
    ToolCall,
    ToolSpec,
    complete,
    generate,
    mesh,
)
from cortexgrid_infer.protocols import (
    ServedCompletingModel,
    ServedGeneratingModel,
    ServedMeshingModel,
)
from cortexgrid_infer.device import detect_device
from cortexgrid_infer.serve_apps import (
    AnthropicText2Text,
    GeminiText2Image,
    GeminiText2Text,
    HostedModel,
    Image2Mesh,
    LocalModel,
    Text2Image,
    Text2Text,
    Weights,
)
from cortexgrid_infer.registry import Hosted, ModelEntry, split_model_id
from cortexgrid_infer.importers import HuggingFaceImporter, Importer

__all__ = [
    "CompletionChunk",
    "DeployedModel",
    "CompletingModel",
    "GeneratedImage",
    "GeneratedMesh",
    "GeneratingModel",
    "MeshingModel",
    "Message",
    "Tool",
    "ToolCall",
    "ToolSpec",
    "complete",
    "generate",
    "mesh",
    "ModelDeployFailed",
    "detect_device",
    "ServedCompletingModel",
    "ServedGeneratingModel",
    "ServedMeshingModel",
    "LocalModel",
    "HostedModel",
    "Weights",
    "Text2Text",
    "Text2Image",
    "Image2Mesh",
    "AnthropicText2Text",
    "GeminiText2Text",
    "GeminiText2Image",
    "ModelEntry",
    "Hosted",
    "split_model_id",
    "Importer",
    "HuggingFaceImporter",
]
