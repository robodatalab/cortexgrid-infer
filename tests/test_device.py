"""Tests for model_gateway.device."""

import unittest

import torch

from model_gateway.device import detect_device


class TestDetectDevice(unittest.TestCase):
    def test_returns_valid_device(self):
        device = detect_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ("cuda", "mps", "cpu"))
