"""The wire protocols between a serve app and its client, one per task, each with
the client that speaks it."""

from cortexgrid_infer.protocols.completion import ServedCompletingModel
from cortexgrid_infer.protocols.imaging import ServedGeneratingModel
from cortexgrid_infer.protocols.meshing import ServedMeshingModel

__all__ = [
    "ServedCompletingModel",
    "ServedGeneratingModel",
    "ServedMeshingModel",
]
