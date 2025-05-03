import torch
import pytest

from src.normalization import GroupNorm, BatchNorm, LayerNorm, InstanceNorm
from src.utils import set_seed


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "shape, num_groups, affine",
    [((64, 6, 32, 32), 2, False), ((128, 8, 1024), 4, True)],
)
def test_group_norm(shape: tuple[int, ...], num_groups: int, affine: bool) -> None:
    for seed in range(10):
        scale = torch.randint(1, 10, (1,), dtype=torch.float64)
        bias = torch.randint(-10, 10, (1,), dtype=torch.float64)
        inputs = torch.rand(shape, dtype=torch.float64) * scale + bias

        set_seed(seed)
        model = GroupNorm(num_groups, shape[1], affine=affine, eps=0, dtype=torch.double)

        set_seed(seed)
        model_torch = torch.nn.GroupNorm(num_groups, shape[1], affine=affine, eps=0, dtype=torch.double)

        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        assert outputs.shape == outputs_torch.shape, f"Shape mismatch: {outputs.shape} vs {outputs_torch.shape}"
        assert torch.allclose(outputs, outputs_torch, atol=1e-3), "GroupNorm outputs mismatch"


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "shape, num_groups, affine",
    [((32, 6, 16, 16), 1, False), ((16, 12, 8, 8), 3, True)],
)
def test_batch_norm(shape: tuple[int, ...], num_groups: int, affine: bool) -> None:
    for seed in range(10):
        scale = torch.randint(1, 10, (1,), dtype=torch.float64)
        bias = torch.randint(-10, 10, (1,), dtype=torch.float64)
        inputs = torch.rand(shape, dtype=torch.float64) * scale + bias

        set_seed(seed)
        model = BatchNorm(num_groups, shape[1], eps=1e-5, affine=affine)

        set_seed(seed)
        model_torch = torch.nn.BatchNorm2d(shape[1], eps=1e-5, affine=affine).to(dtype=torch.double)

        if affine:
            with torch.no_grad():
                model_torch.weight.copy_(model.weight)
                model_torch.bias.copy_(model.bias)

        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        assert outputs.shape == outputs_torch.shape, f"Shape mismatch: {outputs.shape} vs {outputs_torch.shape}"
        assert torch.allclose(outputs, outputs_torch, atol=1e-3), "BatchNorm outputs mismatch"


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "shape, affine",
    [((16, 8, 32, 32), False), ((8, 6, 16, 16), True)],
)
def test_layer_norm(shape: tuple[int, ...], affine: bool) -> None:
    for seed in range(10):
        scale = torch.randint(1, 10, (1,), dtype=torch.float64)
        bias = torch.randint(-10, 10, (1,), dtype=torch.float64)
        x = torch.rand(shape, dtype=torch.float64) * scale + bias

        set_seed(seed)
        model = LayerNorm(shape[1], eps=1e-5)
        model_torch = torch.nn.LayerNorm(shape[1:], eps=1e-5, elementwise_affine=False).to(dtype=torch.double)

        y = model(x)
        y_torch = model_torch(x)

        assert y.shape == y_torch.shape
        assert torch.allclose(y, y_torch, atol=1e-3), "LayerNorm outputs mismatch"


@pytest.mark.order(4)
@pytest.mark.parametrize(
    "shape, affine",
    [((8, 3, 32, 32), False), ((16, 6, 8, 8), True)],
)
def test_instance_norm(shape: tuple[int, ...], affine: bool) -> None:
    for seed in range(10):
        scale = torch.randint(1, 10, (1,), dtype=torch.float64)
        bias = torch.randint(-10, 10, (1,), dtype=torch.float64)
        x = torch.rand(shape, dtype=torch.float64) * scale + bias

        set_seed(seed)
        model = InstanceNorm(shape[1], eps=1e-5, affine=affine)
        model_torch = torch.nn.InstanceNorm2d(shape[1], eps=1e-5, affine=affine).to(dtype=torch.double)

        if affine:
            with torch.no_grad():
                model_torch.weight.copy_(model.weight)
                model_torch.bias.copy_(model.bias)

        y = model(x)
        y_torch = model_torch(x)

        assert y.shape == y_torch.shape
        assert torch.allclose(y, y_torch, atol=1e-3), "InstanceNorm outputs mismatch"
