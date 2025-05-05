# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
from typing import Optional, Any


def unfold_max_pool_2d(
    inputs: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    """
    This function computes the unfold needed for the MaxPool2d.
    Since the maxpool only computes the max over single channel
    and not over all the channels, we need that the second dimension of
    our unfold tensors are data from only channel. For that, we will
    include the channels into another dimension that will
    not be affected by the consequently operations.

    Args:
        inputs: inputs tensor. Dimensions: [batch, channels, height,
            width].
        kernel_size: size of the kernel to use. In this case the
            kernel will be symmetric, that is why only an integer is
            accepted.
        stride: stride to use in the maxpool operation. As in the case
            of the kernel size, the stride willm be symmetric.
        padding: padding to use in the maxpool operation. As in the
            case of the kernel.

    Returns:
        inputs unfolded. Dimensions: [batch * channels,
            kernel size * kernel size, number of windows].
    """

    # TODO
    b,c,h,w = inputs.shape
    inputs = inputs.view(b*c,1,h,w)

    unfolded_inputs = F.unfold(
        inputs,
        kernel_size=kernel_size,
        stride=stride, 
        padding=padding
    )

    return unfolded_inputs
    

def fold_max_pool_2d(
    inputs: torch.Tensor,
    output_size: int,
    batch_size: int,
    kernel_size,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    This function computes the fold needed for the MaxPool2d.
    Since the maxpool only comute sthe max over single channel
    and not over all the channels, we need that the second dimension of
    our unfold tensors are data from only channel. To do that, we
    this fold version recovers the channel dimensions before executing 
    the fold operation.

    Args:
        inputs: inputs unfolded. Dimensions: [batch * channels,
            kernel size * kernel size, number of windows].

        output_size: output size for the fold, i.e., the height and
            the width.
        batch_size: batch size
        stride: stride to use in the maxpool operation. As in the case
            of the kernel size, the stride willm be symmetric.
        padding: padding to use in the maxpool operation. As in the
            case of the kernel.

    Returns:
        inputs folded. Dimensions: [batch, channels, height, width].
    """

    # TODO
    bc, _,_ = inputs.shape
    x= inputs.view(batch_size, bc//batch_size*kernel_size*kernel_size,-1)
    outputs = F.fold(
        x,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding

    )
    return outputs



class MaxPool2dFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the MaxPool2d.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the MaxPool2d.

        Args:
            ctx: context for saving elements for the backward.
            inputs: inputs for the model. Dimensions: [batch,
                channels, height, width].

        Returns:
            output of the layer. Dimensions:
                [batch, channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """

        # TODO
        # Unfold inputs
        b, ci, hi, wi = inputs.shape
        ho = (hi+2*padding-kernel_size)//stride + 1
        unfolded_inputs = unfold_max_pool_2d(inputs, kernel_size=kernel_size, stride=stride, padding=padding) # bc x (cikhkw x howo)

        # Compute max over unfolded inputs
        unfolded_outputs, max_indices = unfolded_inputs.max(dim =1)
        outputs = unfolded_outputs.view(b, ci, ho, ho)
        ctx.save_for_backward(max_indices, unfolded_inputs, inputs)
        ctx.kernel_size = kernel_size
        ctx.padding = padding
        ctx.stride = stride

        return outputs


    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        This method is the backward of the MaxPool2d.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions:
                [batch, channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]

        Returns:
            inputs gradients dimensions: [batch, channels,
                height, width].
            None value.
            None value.
            None value.
        """

        # TODO
        
        max_indices, unfolded_inputs, inputs = ctx.saved_tensors
        b, c, ho, wo = grad_outputs.shape
        b, c, hi, wi = inputs.shape
        kernel_size, padding, stride = ctx.kernel_size, ctx.padding, ctx.stride

        # Unfold
        grad_outputs_unfolded = grad_outputs.view(b*c, -1)
        
        # d/dx max
        grad_inputs_unfold = torch.zeros_like(unfolded_inputs)
        grad_inputs_unfold.scatter_(1, max_indices.unsqueeze(1), grad_outputs_unfolded.unsqueeze(1)) # dim = 1(kxk)


        # Fold
        grad_inputs = fold_max_pool_2d(grad_inputs_unfold,
                                       output_size=hi,
                                       batch_size=b,
                                       kernel_size=kernel_size,
                                       padding=padding, 
                                       stride=stride)
        
        return (grad_inputs,
                None,
                None,
                None)


class MaxPool2d(torch.nn.Module):
    """
    This is the class that represents the MaxPool2d Layer.
    """

    kernel_size: int
    stride: int

    def __init__(
        self, kernel_size: int, stride: Optional[int], padding: int = 0
    ) -> None:
        """
        This method is the constructor of the MaxPool2d layer.
        """

        # call super class constructor
        super().__init__()

        # set attributes value
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

        # save function
        self.fn = MaxPool2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, channels,
                output channels, height, width].

        Returns:
            outputs tensor. Dimensions: [batch, channels,
                height - kernel size + 1, width - kernel size + 1].
        """

        return self.fn(inputs, self.kernel_size, self.stride, self.padding)



"""
Avg Pooling Layer
"""


class AvgPool2dFunction(torch.autograd.Function):
    """
    Custom implementation of AvgPool2d forward and backward.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
        """
        This method implements the forward pass of AvgPool2d.

        Args:
            ctx: context for saving tensors for the backward pass.
            inputs: input tensor. Dimensions: [batch, channels, height, width].
            kernel_size: pooling kernel size.
            stride: stride used for pooling.
            padding: padding applied to inputs.

        Returns:
            Output tensor. Dimensions: [batch, channels, pooled height, pooled width].
        """
        # TODO
        pass

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """
        This method implements the backward pass of AvgPool2d.

        Args:
            ctx: context with saved tensors.
            grad_outputs: gradient of the loss w.r.t. output. Dimensions:
                [batch, channels, pooled height, pooled width].

        Returns:
            Gradient with respect to input tensor.
            None (for kernel_size).
            None (for stride).
            None (for padding).
        """
        # TODO
        pass

class AvgPool2d(torch.nn.Module):
    """
    This class represents the AvgPool2d Layer.
    It wraps the custom autograd function AvgPool2dFunction.
    """

    kernel_size: int
    stride: int

    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: int = 0) -> None:
        """
        Constructor of the AvgPool2d layer.

        Args:
            kernel_size: size of the pooling window.
            stride: stride of the pooling operation.
            padding: zero-padding added to both sides.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.fn = AvgPool2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the AvgPool2d layer.

        Args:
            inputs: input tensor. Dimensions: [batch, channels, height, width].

        Returns:
            Pooled output tensor.
        """
        return self.fn(inputs, self.kernel_size, self.stride, self.padding)