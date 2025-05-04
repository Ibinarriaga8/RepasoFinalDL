# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.pooling import unfold_max_pool_2d, fold_max_pool_2d, MaxPool2d
from src.utils import set_seed


@pytest.mark.parametrize(
    "shape, kernel_size", [((64, 3, 32, 32), 4), ((128, 2, 64, 64), 3)]
)
def test_unfold_max_pool_2d(shape: tuple[int, ...], kernel_size: int) -> None:
    """
    This function is the test for the unfold_max_pool_2d.

    Args:
        shape: shape of the input tensor.
        kernel_size: kernel size for the unfold.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(shape)

    # unfold inputs
    inputs_unfolded: torch.Tensor = unfold_max_pool_2d(inputs, kernel_size, 1, 0)

    # check dimensions
    assert inputs_unfolded.shape[:2] == (
        shape[0] * shape[1],
        kernel_size**2,
    ), "Incorrect shape of unfold"

    # check values
    assert (
        inputs[0, 0, :kernel_size, :kernel_size].reshape(-1) != inputs_unfolded[0, :, 0]
    ).sum().item() == 0, "Incorrect values of unfold"
    assert (
        inputs[0, 1, :kernel_size, :kernel_size].reshape(-1) != inputs_unfolded[1, :, 0]
    ).sum().item() == 0, "Incorrect values of unfold"
    assert (
        inputs[0, 0, :kernel_size, 1 : (kernel_size + 1)].reshape(-1)
        != inputs_unfolded[0, :, 1]
    ).sum().item() == 0, "Incorrect values of unfold"

    return None


@pytest.mark.order(6)
@pytest.mark.parametrize(
    "shape,, kernel_size, stride, padding",
    [((64, 3, 32, 32), 3, 1, 0), ((128, 6, 64, 64), 7, 1, 0)],
)
def test_fold_max_pool_2d(
    shape: tuple[int, ...], kernel_size: int, stride: int, padding: int
) -> None:
    """
    This function is the test for the fold_max_pool_2d.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(shape)

    # compute fold version
    inputs_folded: torch.Tensor = fold_max_pool_2d(
        unfold_max_pool_2d(inputs, kernel_size, 1, 0),
        shape[2],
        inputs.shape[0],
        kernel_size,
        stride,
        padding,
    )

    # compute check tensor
    input_ones = torch.ones(inputs.shape, dtype=inputs.dtype)
    divisor = fold_max_pool_2d(
        unfold_max_pool_2d(input_ones, kernel_size, 1, 0),
        shape[2],
        inputs.shape[0],
        kernel_size,
        stride,
        padding,
    )
    check_tensor: torch.Tensor = divisor * inputs

    # check dimensions
    assert inputs_folded.shape == shape, "Incorrect shape of fold"

    # check values
    assert torch.allclose(
        inputs_folded, check_tensor, atol=1e-5
    ), "Incorrect values of fold"

    return None


@pytest.mark.order(7)
@pytest.mark.parametrize(
    "shape, kernel_size", [((64, 3, 32, 32), 4), ((128, 2, 64, 64), 3)]
)
def test_max_pool_forward(shape: tuple[int, ...], kernel_size: int) -> None:
    """
    This function is the test for the forward of the MaxPool2d.

    Args:
        shape: shape of the input tensor.
        kernel_size: kernel size to use.
    """

    # loop with different seeds
    for seed in range(10):
        # define inputs
        set_seed(seed)
        inputs: torch.Tensor = torch.rand(shape)

        # define models
        model = MaxPool2d(kernel_size, stride=1)
        model_torch = torch.nn.MaxPool2d(kernel_size, stride=1)

        # compute outputs
        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        # check output size
        assert (
            outputs.shape == outputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(outputs, outputs_torch, atol=1e-10), "Incorrect outputs"

    return None


@pytest.mark.order(8)
@pytest.mark.parametrize(
    "shape, kernel_size", [((64, 3, 32, 32), 4), ((128, 2, 64, 64), 3)]
)
def test_max_pool_backward(shape: tuple[int, ...], kernel_size: int) -> None:
    """
    This function is the test for the backward of the MaxPool2d.

    Args:
        shape: shape of the input tensor.
        kernel_size: kernel size to use.
    """

    # loop with different seeds
    for seed in range(10):
        # set seed
        set_seed(seed)

        # define inputs
        inputs: torch.Tensor = torch.rand(shape)
        inputs.requires_grad_(True)

        # define models
        model = MaxPool2d(kernel_size, stride=1)
        model_torch = torch.nn.MaxPool2d(kernel_size, stride=1)

        # compute backward of our maxpool
        outputs = model(inputs)
        if inputs.grad is not None:
            inputs.grad.zero_()
        outputs.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs: torch.Tensor = inputs.grad.clone()

        # compute backward of pytorch maxpool
        outputs_torch = model_torch(inputs)
        inputs.grad.zero_()
        outputs_torch.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs_torch: torch.Tensor = inputs.grad.clone()

        # check output size
        assert (
            grad_inputs.shape == grad_inputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(
            grad_inputs, grad_inputs_torch, atol=1e-10
        ), "Incorrect outputs"

    return None
