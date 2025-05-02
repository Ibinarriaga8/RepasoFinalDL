import torch
import torch.nn.functional as F
import math
from typing import Any

# own modules
from utils import get_dropout_random_indexes


"""
LINEAR LAYER
====================================
"""

class LinearFunction(torch.autograd.Function):
    """
    This class implements the forward and backward of the Linear layer.
    """

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """
        This method is the forward pass of the Linear layer.

        Args:
            ctx: Contex for saving elements for the backward.
            inputs: Inputs tensor. Dimensions:
                [batch, input dimension].
            weight: weights tensor.
                Dimensions: [output dimension, input dimension].
            bias: Bias tensor. Dimensions: [output dimension].

        Returns:
            Outputs tensor. Dimensions: [batch, output dimension].
        """

        # TODO
        result = torch.matmul(inputs, weight.T) + bias #broadcasting with batch dimension
        ctx.save_for_backward(inputs, weight)
        return result


    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward for the Linear layer.
        Computes inputs, weights and bias gradients through backpropagation with grad_output.

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients.
                Dimensions: [batch, output dimension].

        Returns:
            Inputs gradients. Dimensions: [batch, input dimension].
            Weights gradients. Dimensions: [output dimension,
                input dimension].
            Bias gradients. Dimension: [output dimension].
        """

        # TODO

        inputs, weight = ctx.saved_tensors

        input_gradient = torch.matmul(grad_output, weight) 
        weight_gradient = torch.matmul(grad_output.T, inputs)
        bias_gradient = grad_output.sum(dim = 0) #suma por batch, por columnas (dim = 0) 
        return input_gradient, weight_gradient, bias_gradient


class Linear(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.

    Attributes:
        weight: Weight torch parameter. Dimensions: [output dimension,
            input dimension].
        bias: Bias torch parameter. Dimensions: [output dimension].
        fn: Autograd function.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Linear layer.
        The attributes must be named the same as the parameters of the
        linear layer in pytorch. The parameters should be initialized

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_dim))

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = LinearFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, input dim].

        Returns:
            Outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(inputs, self.weight, self.bias)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None


"""
CONVOLUTION LAYER
====================================
"""


class Conv2dFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv2d
    layer.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dilation: int,
        padding: int,
        stride: int,
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.

        Args:
            ctx: context for saving elements for the backward.
            inputs: inputs for the model. Dimensions: [batch,
                input channels, sequence length].
            weight: weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size].
            bias: bias of the layer. Dimensions: [output channels].

        Returns:
            output of the layer. Dimensions:
                [batch, output channels,
                (sequence length + 2*padding - kernel size) /
                stride + 1]
        """

        co, ci, k = weight.shape
        b, ci, hi = inputs.shape
        ho = (hi + 2*padding - k) // stride + 1
        unfolded_inputs = unfold1d(inputs, kernel_size=k, dilation=dilation, padding=padding, stride=stride)
        unfolded_kernel = weight.view(co, k*ci)

        output = torch.matmul(unfolded_kernel, unfolded_inputs) + bias.view(co, 1)
        
        ctx.save_for_backward(
            inputs,
            unfolded_inputs, 
            weight, 
            unfolded_kernel
        )
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.stride = stride

        return output
    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        This is the backward of the layer.

        Args:
            ctx: contex for loading elements needed in the backward.
            grad_output: outputs gradients. Dimensions:
                [batch, output channels,
                (sequence length + 2*padding - kernel size) /
                stride + 1]

        Returns:
            gradient of the inputs. Dimensions: [batch,
                input channels, sequence length].
            gradient of the weights. Dimensions: [output channels,
                input channels, kernel size].
            gradient of the bias. Dimensions: [output channels].
            None.
            None.
            None.
        """

        # TODO 
        inputs, unfolded_inputs, weight, unfolded_kernel = ctx.saved_tensors
        padding = ctx.padding
        dilation = ctx.dilation
        stride = ctx.stride

        batch, co, ho = grad_output.shape
        batch, ci, hi = inputs.shape
        co,ci,k = weight.shape
        
        # Inputs gradient
        unfolded_grad_inputs = torch.matmul(grad_output.transpose(1,2), unfolded_kernel) #bxhoxcik
        grad_inputs = fold1d(unfolded_grad_inputs.transpose(1,2),
                             output_size=hi,
                             kernel_size=k,
                             dilation=dilation,
                             padding=padding,
                             stride=stride
                             )

        # Weight gradient
        unfolded_grad_weights = torch.bmm(grad_output, unfolded_inputs.transpose(1,2))
        grad_weights = unfolded_grad_weights.sum(dim = 0).view(co, ci, k)

        # Bias gradient
        grad_bias=grad_output.sum(dim = (0, 2))

        return (
            grad_inputs,
            grad_weights,
            grad_bias,
            None,
            None,
            None
        )


class Conv2d(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.

    Attributes:
        weight: Weight pytorch parameter. Dimensions: [output channels,
            input channels, kernel size, kernel size].
        bias: Bias torch parameter. Dimensions: [output channels].
        padding: Padding parameter.
        stride: Stride parameter.
        fn: Autograd function.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """
        This method is the constructor of the Linear layer. Follow the
        pytorch convention.

        Args:
            input_channels: Input dimension.
            output_channels: Output dimension.
            kernel_size: Kernel size to use in the convolution.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size, kernel_size)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_channels))
        self.padding = padding
        self.stride = stride

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = Conv2dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, input channels,
                output channels, height, width].

        Returns:
            outputs tensor. Dimensions: [batch, output channels,
                height - kernel size + 1, width - kernel size + 1].
        """

        return self.fn(inputs, self.weight, self.bias, self.padding, self.stride)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None
    


# CONV 1D

"""
## Unfold 1D (0.5 points)

Since unfold function of PyTorch is only able to work with 4D tensors (images), you will have to code this version for sequences. 
No nn function or for-loops are allowed except the unfold and fold function of PyTorch.

Hint: The easier way to complete this is to treat the tensors as if they are 4D matrices (images) instead of 3D (sequences).
"""


def unfold1d(
    inputs: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    padding: int = 0,
    stride: int = 1,
) -> torch.Tensor:
    """
    This operation computes the unfold operation for 1d convolutions.

    Args:
        inputs: input tensor. Dimensions: [batch, input channels,
            input sequence length].
        kernel_size: kernel size of the unfold operation.
        dilation: dilation of the unfold operation. Defaults to 1.
        padding: padding of the unfold operation. Defaults to 0.
        stride: stride of the unfold operation. Defaults to 1.

    Returns:
        outputs tensor. Dimensions: [batch,
            input channels * kernel size, number of windows].
    """

    # TODO
    conv_inputs = inputs.unsqueeze(2)
    output = F.unfold(conv_inputs, 
             kernel_size=(1,kernel_size), # Tb .unsqueeze(3), (kernel_size, 1)
             dilation= dilation,
             padding = padding,
             stride = stride)
    return output


"""
## Fold 1D (0.5 points)
Since fold function of PyTorch is only able to work with 4D tensors (images), you will have to code this version for sequences. 
It is needed to complete the unfold1d function before being able to pass the test. 
No nn function or for-loops are allowed EXCEPT Pytorch fold & unfold 

Hint: The easier way to complete this is to treat the tensors as if they are 4D matrices (images) instead of 3D (sequences).
"""

def fold1d(
    inputs: torch.Tensor,
    output_size: int,
    kernel_size: int,
    dilation: int = 1,
    padding: int = 0,
    stride: int = 1,
) -> torch.Tensor:
    """
    This operation computes the fold operation for 1d convolutions.

    Args:
        inputs: input tensor. Dimensions: [batch,
            output channels * kernel size, number of windows].
        output_size: output sequence length.
        kernel_size: kernel size to use in the fold operation.
        dilation: dilation to use in the fold operation.
        stride: stride to use in the fold operation.

    Returns:
        output tensor. Dimensions: [batch, output channels,
            output sequence length].
    """

    # TODO
    folded_inputs = F.fold(
        inputs, 
        output_size = (1, output_size),
        kernel_size=(1, kernel_size),
        dilation=dilation,
        padding=padding,
        stride = stride
    )

    output = folded_inputs.squeeze()
    return output


class Conv1dFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv1d
    layer.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dilation: int,
        padding: int,
        stride: int,
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.

        Args:
            ctx: context for saving elements for the backward.
            inputs: inputs for the model. Dimensions: [batch,
                input channels, sequence length].
            weight: weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size].
            bias: bias of the layer. Dimensions: [output channels].

        Returns:
            output of the layer. Dimensions:
                [batch, output channels,
                (sequence length + 2*padding - kernel size) /
                stride + 1]
        """

        # TODO
        co, ci, k = weight.shape
        b, ci, hi = inputs.shape
        ho = (hi + 2*padding - k) // stride + 1
        unfolded_inputs = unfold1d(inputs, kernel_size=k, dilation=dilation, padding=padding, stride=stride)
        unfolded_kernel = weight.view(co, k*ci)

        output = torch.matmul(unfolded_kernel, unfolded_inputs) + bias.view(co, 1)
        
        ctx.save_for_backward(
            inputs,
            unfolded_inputs, 
            weight, 
            unfolded_kernel
        )
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.stride = stride

        return output

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        This is the backward of the layer.

        Args:
            ctx: contex for loading elements needed in the backward.
            grad_output: outputs gradients. Dimensions:
                [batch, output channels,
                (sequence length + 2*padding - kernel size) /
                stride + 1]

        Returns:
            gradient of the inputs. Dimensions: [batch,
                input channels, sequence length].
            gradient of the weights. Dimensions: [output channels,
                input channels, kernel size].
            gradient of the bias. Dimensions: [output channels].
            None.
            None.
            None.
        """

        # TODO 
        inputs, unfolded_inputs, weight, unfolded_kernel = ctx.saved_tensors
        padding = ctx.padding
        dilation = ctx.dilation
        stride = ctx.stride

        batch, co, ho = grad_output.shape
        batch, ci, hi = inputs.shape
        co,ci,k = weight.shape
        
        # Inputs gradient
        unfolded_grad_inputs = torch.matmul(grad_output.transpose(1,2), unfolded_kernel) #bxhoxcik
        grad_inputs = unfolded_grad_inputs.view(batch, ci, k)

        # Weight gradient
        grad_weights = torch.bmm(grad_output, unfolded_inputs.transpose(1,2))
        grad_weights.sum(dim = 0).view(co, ci, k)

        # Bias gradient
        grad_bias=grad_output.sum(dim = (0, 2))

        return (
            grad_inputs,
            grad_weights,
            grad_bias,
            None,
            None,
            None
        )


class Conv1d(torch.nn.Module):
    """
    This is the class that represents the Conv1d Layer.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """
        This method is the constructor of the Linear layer. Follow the
        pytorch convention.

        Args:
            input_channels: input dimension.
            output_channels: output dimension.
            kernel_size: kernel size to use in the convolution.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_channels))
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = Conv1dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input channels,
                output channels, sequence length].

        Returns:
            outputs tensor. Dimensions: [batch, output channels,
                sequence length - kernel size + 1].
        """

        return self.fn(
            inputs, self.weight, self.bias, self.dilation, self.padding, self.stride
        )

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None
    
class Dropout(torch.nn.Module):
    """
    This the Dropout class.

    Attr:
        p: probability of the dropout.
        inplace: indicates if the operation is done in-place.
            Defaults to False.
    """

    def __init__(self, p: float, inplace: bool = False) -> None:
        """
        This function is the constructor of the Dropout class.

        Args:
            p: probability of the dropout.
            inplace: if the operation is done in place.
                Defaults to False.
        """

        # TODO
        super().__init__()
        self._p = p
        self._inplace = inplace

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forwward pass.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs. Dimensions: [*], same as inputs tensor.
        """

        # TODO
        if self.training: # .train() mode 
            dropout_mask = get_dropout_random_indexes(shape = inputs.shape, p = self._p)
            dropout_mask = (dropout_mask==0)/(1-self._p) #rescale by a factor of p/(1-p)
            
            if self._inplace:
                return inputs.mul_(dropout_mask) # inplace operation mul_ (trailing underscore)
            
            else:
                outputs = inputs.clone()
                return outputs*dropout_mask
        else: # .eval() mode; no dropout
            return inputs