"""Tests for cortexgrid_infer.providers.serving."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from cortexgrid import ModelDeployFailed

from cortexgrid_infer.providers.serving import ensure_serving


_URL = "http://h/r/FLUX.2-klein-base/4B/run-1"


@mock.patch("cortexgrid_infer.providers.serving.cortexgrid.deploy_model")
@mock.patch("cortexgrid_infer.providers.serving.cortexgrid.wait_for_model_serving")
@mock.patch("cortexgrid_infer.providers.serving.cortexgrid.model_serving_status")
class TestEnsureServing(unittest.TestCase):
    def test_waits_on_an_app_running_or_coming_up(
        self,
        mock_status: mock.Mock,
        mock_wait: mock.Mock,
        mock_deploy: mock.Mock,
    ):
        for phase in ("running", "not_started", "deploying", "unhealthy"):
            with self.subTest(phase=phase):
                mock_status.return_value = SimpleNamespace(phase=phase, url=_URL)
                mock_wait.reset_mock()

                url = ensure_serving("FLUX.2-klein-base", "4B", "run-1", 60.0)

                self.assertEqual(url, _URL)
                mock_wait.assert_called_once_with(
                    "FLUX.2-klein-base", "4B", "run-1", timeout=60.0
                )
        mock_deploy.assert_not_called()

    def test_deploys_afresh_without_a_servable_app(
        self,
        mock_status: mock.Mock,
        mock_wait: mock.Mock,
        mock_deploy: mock.Mock,
    ):
        mock_deploy.return_value = SimpleNamespace(url=_URL)
        for phase, status_url in (
            ("not_deployed", None),
            ("failed", _URL),
            ("deleting", _URL),
        ):
            with self.subTest(phase=phase):
                mock_status.return_value = SimpleNamespace(phase=phase, url=status_url)
                mock_deploy.reset_mock()

                url = ensure_serving("FLUX.2-klein-base", "4B", "run-1", 60.0)

                self.assertEqual(url, _URL)
                mock_deploy.assert_called_once_with(
                    family="FLUX.2-klein-base",
                    suffix="4B",
                    run_name="run-1",
                    wait=True,
                    timeout=60.0,
                )
        mock_wait.assert_not_called()

    def test_app_failing_while_waited_on_raises(
        self,
        mock_status: mock.Mock,
        mock_wait: mock.Mock,
        mock_deploy: mock.Mock,
    ):
        mock_status.return_value = SimpleNamespace(phase="deploying", url=_URL)
        mock_wait.side_effect = ModelDeployFailed("DEPLOY_FAILED: replica died")

        with self.assertRaises(ModelDeployFailed):
            ensure_serving("FLUX.2-klein-base", "4B", "run-1", None)

        mock_deploy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
