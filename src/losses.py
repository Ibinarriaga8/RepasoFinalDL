# Standard libraries
from typing import Any

# 3pps
import torch
import math


class HuberLossFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the HuberLoss.
    """

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, targets: torch.Tensor, delta: float
    ) -> torch.Tensor:
        """
        This is the forward method of the HuberLoss.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO

    


    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """
        This method is the backward of the HuberLoss.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [*], same as the grad_output.
            None.
            None.
        """

        # TODO


class HuberLoss(torch.nn.Module):
    """
    This is the class that represents the HuberLoss Layer.
    """

    def __init__(self, delta: float = 1.0):
        """
        This method is the constructor of the HuberLoss layer.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.delta = delta

        # Set function
        self.fn = HuberLossFunction.apply

        return None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(inputs, targets, self.delta)
    