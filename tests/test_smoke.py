"""不依赖数据集下载的快速回归测试。

运行方式：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from cnnlab.config import ConfigError, load_config
from cnnlab.datasets import build_datasets
from cnnlab.engine import build_optimizer, build_scheduler
from cnnlab.layers import BuildContext, build_layers
from cnnlab.losses import build_loss
from cnnlab.model import build_model


class ConfigAndModelSmokeTest(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = load_config("configs/default.yaml")
        self.assertIn("dataset", config)
        self.assertIn("model", config)
        self.assertGreaterEqual(len(config["model"]["layers"]), 1)

    def test_unknown_override_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            load_config("configs/smoke_test.yaml", overrides=["train.epoch=1"])

    def test_synthetic_bundle_shapes(self) -> None:
        config = load_config("configs/smoke_test.yaml")
        bundle = build_datasets(config, seed=7)
        self.assertEqual(bundle.input_shape, (3, 32, 32))
        self.assertEqual(bundle.num_classes, 10)
        self.assertEqual(len(bundle.train), 512)
        self.assertEqual(len(bundle.val), 128)

    def test_dynamic_cnn_forward_and_backward(self) -> None:
        config = load_config("configs/smoke_test.yaml")
        bundle = build_datasets(config, seed=11)
        model = build_model(config, bundle.input_shape, bundle.num_classes)
        inputs = torch.rand(8, *bundle.input_shape)
        targets = torch.randint(0, bundle.num_classes, (8,))
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (8, bundle.num_classes))
        criterion, _ = build_loss(config["loss"], bundle.num_classes, list(range(bundle.num_classes)), torch.device("cpu"))
        loss = criterion(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_residual_config_builds(self) -> None:
        config = load_config(
            "configs/cifar10_resnet.yaml",
            overrides=[
                "dataset.name=synthetic",
                "dataset.download=false",
                "dataset.num_workers=0",
                "dataset.synthetic.train_samples=32",
                "dataset.synthetic.val_samples=16",
                "dataset.synthetic.test_samples=16",
            ],
        )
        bundle = build_datasets(config, seed=3)
        model = build_model(config, bundle.input_shape, bundle.num_classes)
        logits = model(torch.rand(2, *bundle.input_shape))
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_activation_and_normalization_variants(self) -> None:
        context = BuildContext(num_classes=10, defaults={})
        activations = [
            "relu",
            "relu6",
            "leaky_relu",
            "gelu",
            "silu",
            "mish",
            "tanh",
            "sigmoid",
            "hardswish",
            "prelu",
            "log_softmax",
        ]
        for activation in activations:
            blocks, shape, _ = build_layers(
                [
                    {
                        "type": "conv2d",
                        "out_channels": 4,
                        "kernel_size": 3,
                        "padding": "same",
                        "activation": activation,
                        "norm": "batchnorm2d",
                    }
                ],
                (3, 8, 8),
                context,
                prefix=f"test_{activation}",
            )
            output = nn.Sequential(*blocks)(torch.randn(2, 3, 8, 8))
            self.assertEqual(tuple(output.shape), (2, *shape))

        for norm in ["instancenorm2d", "groupnorm"]:
            norm_config = {"type": norm}
            if norm == "groupnorm":
                norm_config["num_groups"] = 2
            blocks, shape, _ = build_layers(
                [
                    {
                        "type": "conv2d",
                        "out_channels": 4,
                        "kernel_size": 3,
                        "padding": "same",
                        "norm": norm_config,
                    }
                ],
                (3, 8, 8),
                context,
                prefix=f"test_{norm}",
            )
            output = nn.Sequential(*blocks)(torch.randn(2, 3, 8, 8))
            self.assertEqual(tuple(output.shape), (2, *shape))

    def test_optimizer_and_scheduler_variants(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        optimizer = build_optimizer(
            model,
            {
                "name": "adamw",
                "lr": 0.001,
                "weight_decay": 0.01,
                "param_groups": [{"name": "bias", "patterns": [r"\.bias$"], "weight_decay": 0.0}],
                "params": {},
            },
        )
        # 让优化器先完成一次 step，避免 PyTorch 对 scheduler.step() 顺序发出警告。
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for scheduler_name in [
            "none",
            "step",
            "multi_step",
            "cosine",
            "cosine_restarts",
            "plateau",
            "linear",
            "exponential",
            "one_cycle",
        ]:
            scheduler = build_scheduler(
                optimizer,
                {"name": scheduler_name, "params": {}},
                epochs=4,
                steps_per_epoch=3,
            )
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(0.5)
                else:
                    scheduler.step()

    def test_linear_residual_auto_projection(self) -> None:
        context = BuildContext(num_classes=10, defaults={})
        blocks, shape, _ = build_layers(
            [
                {
                    "type": "residual",
                    "projection": "auto",
                    "activation": "relu",
                    "layers": [
                        {"type": "linear", "out_features": 8},
                        {"type": "linear", "out_features": 8},
                    ],
                }
            ],
            (4,),
            context,
            prefix="mlp_residual",
        )
        output = nn.Sequential(*blocks)(torch.randn(3, 4))
        self.assertEqual(shape, (8,))
        self.assertEqual(tuple(output.shape), (3, 8))

    def test_standalone_dropout_is_active(self) -> None:
        """p=1 的独立 dropout 层在训练模式下应把输入全部置零。"""

        context = BuildContext(num_classes=10, defaults={})
        blocks, _, _ = build_layers(
            [{"type": "dropout", "p": 1.0}],
            (4,),
            context,
            prefix="dropout_test",
        )
        blocks[0].train()
        output = blocks[0](torch.ones(2, 4))
        self.assertTrue(torch.all(output == 0))


if __name__ == "__main__":
    unittest.main()

