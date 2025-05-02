# test/test_normalization.py

import torch
import pytest

# own modules
from src.normalization import GroupNorm, BatchNorm


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "shape, num_groups, affine",
    [((4, 6, 8, 8), 2, False), ((2, 8, 4, 4), 4, True)]
)
def test_group_norm_forward_matches_torch(shape, num_groups, affine):
    """
    Tests whether the custom GroupNorm forward matches torch.nn.GroupNorm.
    """
    for seed in range(5):
        torch.manual_seed(seed)
        scale = torch.randint(1, 10, (1,)).double()
        bias = torch.randint(-10, 10, (1,)).double()
        x = torch.rand(shape, dtype=torch.double) * scale + bias

        custom_gn = GroupNorm(num_groups, shape[1], affine=affine, eps=1e-5, dtype=torch.double)
        torch_gn = torch.nn.GroupNorm(num_groups, shape[1], affine=affine, eps=1e-5).to(dtype=torch.double)

        if affine:
            with torch.no_grad():
                torch_gn.weight.copy_(custom_gn.weight)
                torch_gn.bias.copy_(custom_gn.bias)

        y_custom = custom_gn(x)
        y_torch = torch_gn(x)

        assert y_custom.shape == y_torch.shape
        assert torch.allclose(y_custom, y_torch, atol=1e-3), "Mismatch in GroupNorm output"


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "shape, num_groups",
    [((2, 4, 8, 8), 1), ((3, 6, 16, 16), 3)]
)
def test_batch_norm_forward_normalization(shape, num_groups):
    """
    Test custom BatchNorm normalizes to zero mean and unit variance per sample.
    """
    for seed in range(5):
        torch.manual_seed(seed)
        x = torch.randn(shape, dtype=torch.float64)

        model = BatchNorm(num_groups, shape[1], eps=1e-5)
        y = model(x)

        # reshape to [batch, -1] to compute per-sample mean and std
        y_flat = y.view(y.shape[0], -1)

        mean = y_flat.mean(dim=1)
        std = y_flat.std(dim=1)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), f"Non-zero mean: {mean}"
        assert torch.allclose(std, torch.ones_like(std), atol=1e-4), f"Non-unit std: {std}"


@pytest.mark.order(3)
def test_group_norm_affine_parameters():
    """
    Test that the affine parameters are properly initialized.
    """
    num_channels = 8
    model = GroupNorm(num_groups=2, num_channels=num_channels, affine=True)

    assert model.weight.shape == (num_channels,)
    assert model.bias.shape == (num_channels,)
    assert torch.allclose(model.weight, torch.ones_like(model.weight))
    assert torch.allclose(model.bias, torch.zeros_like(model.bias))
