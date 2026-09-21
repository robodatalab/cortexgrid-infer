"""What `cortexgrid.import_model` and `cortexgrid.register_model` need to file one
model in the registry and serve it, but cannot work out for themselves.

cortexgrid stores whatever weights it is handed under whatever key it is given,
and serves them with whatever class it was told to bundle. It deliberately knows
nothing about where a model came from or what shape it is. A `ModelEntry`
supplies exactly that missing knowledge - the registry identity, the serve app,
the hardware it needs, its config, and the client - and drives no lifecycle of
its own: importing, deploying and deleting stay the caller's calls against the
cortexgrid SDK.

A model with weights is an importer's (`cortexgrid_infer.importers`) and goes
through `import_model`. A model hosted elsewhere is a `Hosted` entry and goes
through `register_model`:

    entry = Hosted("claude-sonnet-5", AnthropicText2Text)
    cortexgrid.register_model(
        entry.serve_app,
        family=entry.family, suffix=entry.suffix,
        requirements=entry.requirements(), config=entry.config(),
    )
"""

from __future__ import annotations

import abc

import cortexgrid

from cortexgrid_infer.core import DeployedModel
from cortexgrid_infer.models.base import HostedModel


def split_model_id(model_id: str) -> tuple[str, str]:
    """Map a model id to a cortexgrid (family, suffix).

    Strips the org (anything before the first '/'), which an HF repo id has and
    a hosted model's name does not. Splits the remainder on the last '-': the
    part before becomes family, the part after becomes suffix. If there is no
    '-', suffix defaults to 'base'.
    """
    name = model_id.split("/", 1)[-1]
    if "-" not in name:
        return name, "base"
    family, _, suffix = name.rpartition("-")
    return family, suffix


class ModelEntry(abc.ABC):
    """One model as the registry files it and cortexgrid serves it."""

    # The id the model is known by where it came from; the registry key and the
    # client's name derive from it.
    model_id: str

    # The class cortexgrid bundles and instantiates on the cluster.
    serve_app: type

    # The registry key this model is filed under, with cortexgrid.IMPORTED.
    family: str
    suffix: str

    @abc.abstractmethod
    def requirements(self) -> cortexgrid.ModelRequirements:
        """What one replica of this model needs to be placed and to run."""

    def config(self) -> dict[str, str]:
        """Settings the serve app reads with `cortexgrid.model_config`.

        For what is neither weights nor hardware - which model a provider should
        be asked for, the name of a secret to read. Empty for a model whose
        bundled code already knows everything it needs."""
        return {}

    @abc.abstractmethod
    def client(self, url: str) -> DeployedModel:
        """Return a client for this model's serve app, deployed at `url`."""


class Hosted(ModelEntry):
    """A model hosted elsewhere, served by a `HostedModel` app that forwards to it.

    There are no weights, so it is registered rather than imported. `settings`
    are the serve app's own - see its `config` - and are checked here, so a
    misspelt one fails before anything reaches the registry."""

    serve_app: type[HostedModel]

    def __init__(self, model_id: str, serve_app: type[HostedModel], **settings: str) -> None:
        self.model_id = model_id
        self.serve_app = serve_app
        self.family, self.suffix = split_model_id(model_id)
        self._config = serve_app.config(model_id, **settings)

    def requirements(self) -> cortexgrid.ModelRequirements:
        return self.serve_app.requirements()

    def config(self) -> dict[str, str]:
        return dict(self._config)

    def client(self, url: str) -> DeployedModel:
        return self.serve_app.client(url, self.model_id)
