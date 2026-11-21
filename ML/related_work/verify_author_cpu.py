"""Check bounded ATENet attention values and gradients against author code."""

import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import torch

from author_cpu_runtime import bounded_attention, restore_epoch, save_epoch
from run_cpu import ROOT as AUTHOR_ROOT

ROOT = AUTHOR_ROOT / "ATENet/upstream"


def main():
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("ate_author", ROOT / "models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    attention = module.multiTimeAttention(8, nhidden=6, embed_time=16, num_heads=2)
    query = torch.randn(3, 2, 21, 8)
    key = torch.randn(3, 2, 19, 8)
    value = torch.randn(3, 1, 19, 8)
    mask = torch.randint(0, 2, (3, 1, 19, 8))
    mask[0] = 0
    outputs, gradients = [], []
    for method in (attention.attention, bounded_attention.__get__(attention)):
        inputs = [tensor.clone().requires_grad_() for tensor in (query, key, value)]
        predicted, _ = method(*inputs, mask=mask)
        outputs.append(predicted.detach())
        predicted.square().sum().backward()
        gradients.append([tensor.grad.clone() for tensor in inputs])
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-6, atol=2e-6)
    for old, new in zip(*gradients):
        torch.testing.assert_close(old, new, rtol=3e-5, atol=3e-5)

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    inputs = torch.randn(5, 2)
    for _ in range(2):
        optimizer.zero_grad()
        model(inputs).square().sum().backward()
        optimizer.step()
        scheduler.step()
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "epoch.pt"
        save_epoch(path, epoch=1, models={"model": model}, optimizer=optimizer, scheduler=scheduler, metadata={})
        expected = torch.rand(4), np.random.rand(4)
        expected_model = model(inputs).detach().clone()
        with torch.no_grad():
            model.weight.add_(1)
        state = restore_epoch(path, models={"model": model}, optimizer=optimizer, scheduler=scheduler)
        torch.testing.assert_close(model(inputs), expected_model)
        torch.testing.assert_close(torch.rand(4), expected[0])
        np.testing.assert_array_equal(np.random.rand(4), expected[1])
        assert state["epoch"] == 1
    print("PASS author attention values, gradients and all-missing masks; epoch checkpoint and RNG restoration.")


if __name__ == "__main__":
    main()
