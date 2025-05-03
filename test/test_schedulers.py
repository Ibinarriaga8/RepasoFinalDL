
# Standard libraries
import copy

# 3pps
import torch
from torch.optim.lr_scheduler import LRScheduler
import pytest

# own modules
from src.schedulers import *
from src.utils import set_seed

# set seed
set_seed(42)


def test_steplr() -> None:
    # define model
    model: torch.nn.Module = torch.nn.Sequential(
        torch.nn.Linear(30, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )
    model_torch: torch.nn.Module = copy.deepcopy(model)

    # define inputs and targets
    inputs: torch.Tensor = torch.rand(64, 30)
    inputs_torch: torch.Tensor = inputs.clone().detach()
    targets: torch.Tensor = torch.rand(64, 1)

    # define loss and lr
    loss: torch.nn.Module = torch.nn.L1Loss()
    lr: float = 1e-3

    # define optimizers
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_torch: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define schedulers
    scheduler: LRScheduler = StepLR(optimizer, step_size=50, gamma=0.2)
    scheduler_torch: LRScheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_torch, 50, gamma=0.2
    )

    # iter over epochs loop
    for epoch in range(110):
        # compute outputs for both models
        outputs: torch.Tensor = model(inputs)
        outputs_torch: torch.Tensor = model_torch(inputs_torch)

        # compute loss and optimize
        loss_value: torch.Tensor = loss(outputs, targets)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # compute loss and optimize torch model
        loss_value = loss(outputs_torch, targets)
        optimizer_torch.zero_grad()
        loss_value.backward()
        optimizer_torch.step()

        # compute steps
        scheduler.step()
        scheduler_torch.step()

        # get lr and compare them
        lr = optimizer.param_groups[0]["lr"]
        lr_torch: float = optimizer_torch.param_groups[0]["lr"]
        print(lr)
        assert lr == lr_torch, (
            f"Incorrect step of scheduler, expected {lr_torch} in {epoch} epoch, "
            f"and got {lr}"
        )

    return None

def test_multisteplr() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )
    model_torch = copy.deepcopy(model)

    inputs = torch.rand(32, 20)
    targets = torch.rand(32, 1)

    loss = torch.nn.MSELoss()
    lr = 1e-2

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)

    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    scheduler_torch = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_torch, milestones=[10, 20, 30], gamma=0.5
    )

    for epoch in range(35):
        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        loss_value = loss(outputs, targets)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        loss_value = loss(outputs_torch, targets)
        optimizer_torch.zero_grad()
        loss_value.backward()
        optimizer_torch.step()

        scheduler.step()
        scheduler_torch.step()

        lr = optimizer.param_groups[0]["lr"]
        lr_torch = optimizer_torch.param_groups[0]["lr"]
        print(lr)
        assert lr == pytest.approx(lr_torch, rel=1e-6), (
            f"Incorrect step of MultiStepLR, expected {lr_torch} in {epoch} epoch, "
            f"and got {lr}"
        )


def test_cosineannealinglr() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(50, 20), torch.nn.ReLU(), torch.nn.Linear(20, 1)
    )
    model_torch = copy.deepcopy(model)

    inputs = torch.rand(16, 50)
    targets = torch.rand(16, 1)

    loss = torch.nn.L1Loss()
    lr = 1e-2

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_torch = torch.optim.Adam(model_torch.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001)
    scheduler_torch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_torch, T_max=30, eta_min=0.001
    )

    for epoch in range(35):
        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        loss_value = loss(outputs, targets)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        loss_value = loss(outputs_torch, targets)
        optimizer_torch.zero_grad()
        loss_value.backward()
        optimizer_torch.step()

        scheduler.step()
        scheduler_torch.step()

        lr = optimizer.param_groups[0]["lr"]
        lr_torch = optimizer_torch.param_groups[0]["lr"]
        print(lr)
        assert lr == pytest.approx(lr_torch, rel=1e-6), (
            f"Incorrect step of CosineAnnealingLR, expected {lr_torch} in {epoch} epoch, "
            f"and got {lr}"
        )


def test_cycliclr() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)
    )
    model_torch = copy.deepcopy(model)

    inputs = torch.rand(64, 10)
    targets = torch.rand(64, 1)

    loss = torch.nn.MSELoss()
    base_lr, max_lr = 1e-4, 1e-2

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    optimizer_torch = torch.optim.Adam(model_torch.parameters(), lr=base_lr)

    scheduler = CyclicLR(
        optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=10, mode="triangular"
    )
    scheduler_torch = torch.optim.lr_scheduler.CyclicLR(
        optimizer_torch, base_lr=base_lr, max_lr=max_lr, step_size_up=10, mode="triangular"
    )

    for epoch in range(30):
        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        loss_value = loss(outputs, targets)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        loss_value = loss(outputs_torch, targets)
        optimizer_torch.zero_grad()
        loss_value.backward()
        optimizer_torch.step()

        scheduler.step()
        scheduler_torch.step()

        lr = optimizer.param_groups[0]["lr"]
        lr_torch = optimizer_torch.param_groups[0]["lr"]
        print(lr)
        assert lr == pytest.approx(lr_torch, rel=1e-6), (
            f"Incorrect step of CyclicLR, expected {lr_torch} in {epoch} epoch, "
            f"and got {lr}"
        )
