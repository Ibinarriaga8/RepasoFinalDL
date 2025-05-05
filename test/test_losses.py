import torch
import pytest
from src.losses import HuberLoss  # Asegúrate de que la ruta es correcta



@pytest.mark.parametrize("delta", [0.5, 1.0])
class TestHuberLoss:
    """
    This class implements the tests for the HuberLoss.
    """

    @pytest.mark.order(2)
    @torch.no_grad()
    def test_forward(self, delta: float, inputs_linear: torch.Tensor) -> None:
        """
        This method is the test for the forward pass of the HuberLoss.

        Args:
            delta: delta parameter of the loss.
            inputs_linear: inputs from the fixture.

        Returns:
            None.
        """

        # Get inputs and targets
        batch_size: int = inputs_linear.shape[0] // 2
        inputs: torch.Tensor = inputs_linear[:batch_size]
        targets: torch.Tensor = inputs_linear[batch_size:]

        # Define modules
        loss: torch.nn.Module = HuberLoss(delta)
        loss_torch: torch.nn.Module = HuberLoss(delta)

        # Compute outputs
        outputs: torch.Tensor = loss(inputs, targets)
        outputs_torch: torch.Tensor = loss_torch(inputs, targets)

        # Check shape
        assert (
            outputs.shape == outputs_torch.shape
        ), f"Incorrect shape, expected {outputs_torch.shape} and got {outputs.shape}"

        # Check outputs values
        assert torch.allclose(outputs, outputs_torch), "Incorrect forward values"

        # Set some elements at delta
        inputs[:] = delta

        # Compute outputs
        outputs = loss(inputs, targets)
        outputs_torch = loss_torch(inputs, targets)

        # Check outputs value at delta value
        assert torch.allclose(
            outputs, outputs_torch
        ), "Incorrect forward values at delta"

        return None

    @pytest.mark.order(3)
    def test_backward(self, delta: float, inputs_linear: torch.Tensor) -> None:
        """
        This method is the test for the backward pass of HuberLoss.

        Args:
            delta: delta parameter of the loss.
            inputs_linear: inputs from the fixture.

        Raises:
            RuntimeError: Error in gradient computation.

        Returns:
            None.
        """

        # Get inputs and targets
        batch_size: int = inputs_linear.shape[0] // 2
        inputs: torch.Tensor = inputs_linear[:batch_size].requires_grad_(True)
        targets: torch.Tensor = inputs_linear[batch_size:]

        # Define modules
        loss: torch.nn.Module = HuberLoss(delta)
        loss_torch: torch.nn.Module = HuberLoss(delta)

        for i in range(2):
            # Set elements to delta in second iteration
            if i == 1:
                with torch.no_grad():
                    inputs[:] = delta

            # Compute outputs
            outputs: torch.Tensor = loss(inputs, targets)
            outputs.sum().backward()
            if inputs.grad is None:
                raise RuntimeError("Error in gradient computation")
            gradients: torch.Tensor = inputs.grad.clone()
            inputs.grad.zero_()

            # Compute torch gradients
            outputs = loss_torch(inputs, targets)
            outputs.sum().backward()
            if inputs.grad is None:
                raise RuntimeError("Error in gradient computation")
            gradients_torch: torch.Tensor = inputs.grad.clone()
            inputs.grad.zero_()

            # Check gradients values
            if i == 1:
                assert torch.allclose(
                    gradients, gradients_torch
                ), "Incorrect backward values at delta"
            else:
                assert torch.allclose(
                    gradients, gradients_torch
                ), "Incorrect backward values"

        return None
