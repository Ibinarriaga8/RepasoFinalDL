"""
This module contains the code to test optimization algorithms.
"""

# Standard libraries
import copy

# 3pps
import torch
import pytest

# Own modules
from src.optimization import SGD, SGDMomentum, SGDNesterov, Adam, Adafactor

@pytest.fixture
def artifacts() -> tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
    torch.manual_seed(0)  # asegura reproducibilidad
    inputs = torch.rand(32, 10)
    targets = torch.rand(32, 1)
    model = torch.nn.Sequential(torch.nn.Linear(10, 1))
    return inputs, targets, model

@pytest.mark.order(1)
@pytest.mark.parametrize("lr, weight_decay", [(1e-3, 0), (1e-4, 1e-2)])
def test_sgd(
    lr: float,
    weight_decay: float,
    artifacts: tuple[torch.Tensor, torch.Tensor, torch.nn.Module],
) -> None:
    """
    This function is the test for the SGD algorithm.

    Args:
        lr: learning rate.
        weight_decay: weight decay rate.
    """

    # Get artifacts
    inputs: torch.Tensor
    targets: torch.Tensor
    model_original: torch.nn.Module
    inputs, targets, model_original = artifacts

    # clone model
    model1: torch.nn.Module = copy.deepcopy(model_original)
    model2: torch.nn.Module = copy.deepcopy(model_original)

    # define loss and lr
    loss = torch.nn.L1Loss()

    # define optimizers
    optimizer1: torch.optim.Optimizer = torch.optim.SGD(
        model1.parameters(), lr=lr, weight_decay=weight_decay
    )
    optimizer2: torch.optim.Optimizer = SGD(
        model2.parameters(), lr=lr, weight_decay=weight_decay
    )

    # optimize first model
    for _ in range(10):
        outputs: torch.Tensor = model1(inputs)
        loss_value: torch.Tensor = loss(outputs, targets)
        optimizer1.zero_grad()
        loss_value.backward()
        optimizer1.step()

    # optimize second model
    for _ in range(10):
        # optimize second model
        outputs = model2(inputs)
        loss_value = loss(outputs, targets)
        optimizer2.zero_grad()
        loss_value.backward()
        optimizer2.step()

    # check parameters of both models
    for parameter1, parameter2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(
            parameter1.data, parameter2.data
        ), "Incorrect return of the algorithm"

    return None


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "lr, momentum, weight_decay", [(1e-3, 0.9, 0), (1e-4, 0, 1e-2)]
)
def test_sgd_momentum(
    lr: float,
    momentum: float,
    weight_decay: float,
    artifacts: tuple[torch.Tensor, torch.Tensor, torch.nn.Module],
) -> None:
    """
    This function is the test for the SGD algorithm with momentum.

    Args:
        lr: learning rate.
        momentum: momentum rate.
        weight_decay: weight decay rate.
    """

    # Get artifacts
    inputs: torch.Tensor
    targets: torch.Tensor
    model_original: torch.nn.Module
    inputs, targets, model_original = artifacts

    # clone model
    model1: torch.nn.Module = copy.deepcopy(model_original)
    model2: torch.nn.Module = copy.deepcopy(model_original)

    # define loss and lr
    loss = torch.nn.L1Loss()

    # define optimizers
    optimizer1: torch.optim.Optimizer = torch.optim.SGD(
        model1.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    optimizer2: torch.optim.Optimizer = SGDMomentum(
        model2.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # optimize first model
    for _ in range(10):
        outputs: torch.Tensor = model1(inputs)
        loss_value: torch.Tensor = loss(outputs, targets)
        optimizer1.zero_grad()
        loss_value.backward()
        optimizer1.step()

    # optimize second model
    for _ in range(10):
        # optimize second model
        outputs = model2(inputs)
        loss_value = loss(outputs, targets)
        optimizer2.zero_grad()
        loss_value.backward()
        optimizer2.step()

    # check parameters of both models
    for parameter1, parameter2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(
            parameter1.data, parameter2.data
        ), "Incorrect return of the algorithm"

    return None


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "lr, momentum, weight_decay", [(1e-3, 0.9, 0), (1e-4, 0.5, 1e-2)]
)
def test_sgd_nesterov(
    lr: float,
    momentum: float,
    weight_decay: float,
    artifacts: tuple[torch.Tensor, torch.Tensor, torch.nn.Module],
) -> None:
    """
    This function is the test for the SGD algorithm with nesterov.
    momentum.

    Args:
        lr: learning rate.
        momentum: momentum rate.
        weight_decay: weight decay rate.
    """

    # Get artifacts
    inputs: torch.Tensor
    targets: torch.Tensor
    model_original: torch.nn.Module
    inputs, targets, model_original = artifacts

    # clone model
    model1: torch.nn.Module = copy.deepcopy(model_original)
    model2: torch.nn.Module = copy.deepcopy(model_original)

    # define loss and lr
    loss = torch.nn.L1Loss()

    # define optimizers
    optimizer1: torch.optim.Optimizer = torch.optim.SGD(
        model1.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    optimizer2: torch.optim.Optimizer = SGDNesterov(
        model2.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # optimize first model
    for _ in range(10):
        outputs: torch.Tensor = model1(inputs)
        loss_value: torch.Tensor = loss(outputs, targets)
        optimizer1.zero_grad()
        loss_value.backward()
        optimizer1.step()

    # optimize second model
    for _ in range(10):
        # optimize second model
        outputs = model2(inputs)
        loss_value = loss(outputs, targets)
        optimizer2.zero_grad()
        loss_value.backward()
        optimizer2.step()

    # check parameters of both models
    for parameter1, parameter2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(
            parameter1.data, parameter2.data
        ), "Incorrect return of the algorithm"

    return None


@pytest.mark.order(4)
@pytest.mark.parametrize(
    "lr, betas, weight_decay", [(1e-3, (0.9, 0.999), 0), (1e-4, (0.5, 0.4), 1e-2)]
)
def test_adam(
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
    artifacts: tuple[torch.Tensor, torch.Tensor, torch.nn.Module],
) -> None:
    """
    This function is the test for the SGD algorithm with nesterov
    momentum.

    Args:
        lr: learning rate.
        betas: betas parameters.
        weight_decay: weight decay rate.
    """

    # Get artifacts
    inputs: torch.Tensor
    targets: torch.Tensor
    model_original: torch.nn.Module
    inputs, targets, model_original = artifacts

    # clone model
    model1: torch.nn.Module = copy.deepcopy(model_original)
    model2: torch.nn.Module = copy.deepcopy(model_original)

    # define loss and lr
    loss = torch.nn.L1Loss()

    # define optimizers
    optimizer1: torch.optim.Optimizer = torch.optim.Adam(
        model1.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )
    optimizer2: torch.optim.Optimizer = Adam(
        model2.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )

    # optimize first model
    for _ in range(10):
        outputs: torch.Tensor = model1(inputs)
        loss_value: torch.Tensor = loss(outputs, targets)
        optimizer1.zero_grad()
        loss_value.backward()
        optimizer1.step()

    # optimize second model
    for _ in range(10):
        # optimize second model
        outputs = model2(inputs)
        loss_value = loss(outputs, targets)
        optimizer2.zero_grad()
        loss_value.backward()
        optimizer2.step()

    # check parameters of both models
    for parameter1, parameter2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(
            parameter1.data, parameter2.data
        ), "Incorrect return of the algorithm"

    return None



@pytest.mark.order(5)
@pytest.mark.parametrize("lr, eps, weight_decay", [(1e-2, 1e-30, 0), (1e-3, 1e-8, 1e-2)])
def test_adafactor(
    lr: float,
    eps: float,
    weight_decay: float,
    artifacts: tuple[torch.Tensor, torch.Tensor, torch.nn.Module],
) -> None:
    """
    This function is the test for the Adafactor optimizer.

    Args:
        lr: learning rate.
        eps: epsilon value for numerical stability.
        weight_decay: weight decay rate.
    """

    # Get test artifacts
    inputs, targets, model_original = artifacts

    # Clone models
    model1 = copy.deepcopy(model_original)
    model2 = copy.deepcopy(model_original)

    # Define loss
    loss = torch.nn.L1Loss()

    # Define optimizers
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    optimizer2 = Adafactor(model2.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)

    # Optimize reference model (Adam)
    for _ in range(10):
        outputs = model1(inputs)
        loss_value = loss(outputs, targets)
        optimizer1.zero_grad()
        loss_value.backward()
        optimizer1.step()

    # Optimize custom Adafactor
    for _ in range(10):
        outputs = model2(inputs)
        loss_value = loss(outputs, targets)
        optimizer2.zero_grad()
        loss_value.backward()
        optimizer2.step()

    # Compare final weights
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(param1.data, param2.data, atol=1e-2), "Incorrect Adafactor behavior"

    return None
