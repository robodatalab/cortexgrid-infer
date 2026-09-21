"""Run inference on models the cortexgrid cluster serves.

cortexgrid owns a model's life - registry identity, weights, serve bundle,
hardware placement, deployment, jobs - and knows nothing about what kind of
model it holds. This library supplies the per-family knowledge those calls need
and nothing else: an importer that names, fetches, sizes and serves one family
of models, and a client that speaks the deployed app's routes.

    imp = cortexgrid_infer.HuggingFaceCompletingImport("Qwen/Qwen2.5-0.5B-Instruct")
    with imp:
        cortexgrid.import_model(
            imp.source, imp.serve_app,
            family=imp.family, suffix=imp.suffix,
            requirements=imp.requirements(),
        )
    deployment = cortexgrid.deploy_model(imp.family, imp.suffix, cortexgrid.IMPORTED, wait=True)
    model = imp.client(deployment.url)

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
from cortexgrid_infer.completion import ServedCompletingModel
from cortexgrid_infer.device import detect_device
from cortexgrid_infer.importing import HuggingFaceImport, ModelImport, split_model_id
from cortexgrid_infer.providers.anthropic import AnthropicImport
from cortexgrid_infer.providers.huggingface_complete import HuggingFaceCompletingImport
from cortexgrid_infer.providers.huggingface_image import (
    HuggingFaceImageImport,
    HuggingFaceImageModel,
)
from cortexgrid_infer.meshing import ServedMeshingModel
from cortexgrid_infer.meshing_serve import MeshingDeployment
from cortexgrid_infer.providers.huggingface_mesh import HuggingFaceMeshImport

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
    "split_model_id",
    "ModelImport",
    "HuggingFaceImport",
    "HuggingFaceCompletingImport",
    "HuggingFaceImageImport",
    "HuggingFaceMeshImport",
    "AnthropicImport",
    "ServedCompletingModel",
    "HuggingFaceImageModel",
    "ServedMeshingModel",
    "MeshingDeployment",
]
