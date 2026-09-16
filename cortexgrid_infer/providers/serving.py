"""Bring a registry-ready model's cortexgrid Serve app to RUNNING, whatever an
earlier deploy left behind. Shared by the cluster-backed providers."""

from __future__ import annotations

import cortexgrid


# Phases of an app that is running, still coming up, or recovering. It is waited
# on rather than deployed again: re-PUTting the spec while Ray builds the app
# cancels the build and starts it over.
_SERVABLE_PHASES = ("running", "not_started", "deploying", "unhealthy")


def ensure_serving(
    family: str, suffix: str, run_name: str, timeout: float | None
) -> str:
    """Return the URL of the model's Serve app once it is RUNNING.

    An app that is running or on its way there is waited on. Anything else - no
    app, a failed one, or one being deleted - is deployed afresh by
    `cortexgrid.deploy_model`, which clears a failed app first. Raises
    `cortexgrid.ModelDeployFailed` if the app fails to deploy, and TimeoutError
    if it is not RUNNING within `timeout` seconds (None waits indefinitely)."""
    status = cortexgrid.model_serving_status(family, suffix, run_name)
    if status.phase in _SERVABLE_PHASES and status.url is not None:
        cortexgrid.wait_for_model_serving(family, suffix, run_name, timeout=timeout)
        return status.url
    deployment = cortexgrid.deploy_model(
        family=family, suffix=suffix, run_name=run_name, wait=True, timeout=timeout
    )
    return deployment.url
