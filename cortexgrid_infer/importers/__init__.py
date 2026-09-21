"""Where a model's weights come from: one importer per source, each taking any
model its source holds into the cortexgrid registry."""

from cortexgrid_infer.importers.base import Importer
from cortexgrid_infer.importers.huggingface import HuggingFaceImporter

__all__ = [
    "Importer",
    "HuggingFaceImporter",
]
