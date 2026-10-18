"""光谱数据管线和光谱 CNN 的集成测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from cnnlab.config import load_config
from cnnlab.datasets import build_datasets
from cnnlab.model import build_model
from cnnlab.spectra import read_dark_avg, read_scan_txt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "[original data]"


class SpectralPipelineTest(unittest.TestCase):
    def test_txt_and_dark_shapes(self) -> None:
        txt_path = next((RAW_ROOT / "2-辛酮").glob("808-309_*.txt"))
        wavelengths, scans = read_scan_txt(txt_path)
        dark_wavelengths, dark_values = read_dark_avg(RAW_ROOT / "dark_avg.xlsx")
        self.assertEqual(scans.shape[0], 50)
        self.assertEqual(scans.shape[1], wavelengths.size)
        self.assertEqual(dark_wavelengths.size, dark_values.size)
        self.assertGreater(dark_wavelengths.size, 500)
        self.assertTrue(np.isfinite(scans).all())

    def test_spectral_bundle_and_model_forward(self) -> None:
        config = load_config("configs/spectral_cnn.yaml")
        bundle = build_datasets(config, seed=42)
        self.assertEqual(bundle.input_shape, (1, 23, 592))
        self.assertEqual(bundle.num_classes, 51)
        self.assertEqual(bundle.train_size, 1785)
        self.assertEqual(bundle.val_size, 408)
        self.assertEqual(bundle.test_size, 357)
        model = build_model(config, bundle.input_shape, bundle.num_classes)
        inputs = torch.randn(2, *bundle.input_shape)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2, 51))


if __name__ == "__main__":
    unittest.main()

