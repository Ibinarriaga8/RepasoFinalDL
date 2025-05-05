import torch
import torch.nn.functional as F
import math
from typing import Any, Optional

# own modules
from src.utils import get_dropout_random_indexes


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
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        padding: int,
        stride: int,
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Inputs for the model. Dimensions: [batch,
                input channels, height, width].
            weight: Weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size, kernel size].
            bias: Bias of the layer. Dimensions: [output channels].
            padding: padding parameter.
            stride: stride parameter.

        Returns:
            Output of the layer. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """

        # TODO

        #Unfolding the input tensor

        #Unfolding the input tensor
        b, _, hi, wi = inputs.shape
        co, ci, kh, kw = weight.shape
        
        ho, wo = (hi+2*padding-kh) // stride + 1, (wi+2*padding-kh) // stride + 1
        # UNFOLD
        unfolded_inputs = F.unfold(inputs, kernel_size= (kh, kw), padding=padding, stride = stride) # (b, cikhkw, howo)
        unfolded_kernel = weight.view(co, ci*kh*kw)

        # Compute convolution
        unfolded_outputs = torch.matmul(unfolded_kernel, unfolded_inputs) + bias.view(co, 1) # (co, howo)

        #FOLD (co, howo) -> (co, ho, wo)
        outputs = F.fold(unfolded_outputs, (ho, wo), padding=padding, stride=stride)

        # Save for backward
        ctx.save_for_backward(inputs, weight, bias, unfolded_inputs, unfolded_kernel)
        ctx.stride = stride
        ctx.padding = padding

        return outputs


    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        This is the backward of the layer.

        Args:
            ctx: Context for loading elements needed in the backward.
            grad_output: Outputs gradients. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
                #2*padding to make it symmetric; add 2*padding to height and width	

        Returns:
            Inputs gradients. Dimensions: [batch, input channels,
                height, width].
            Weight gradients. Dimensions: [output channels,
                input channels, kernel size, kernel size].
            Bias gradients. Dimensions: [output channels].
            None.
            None.
        """

        # TODO
        inputs, weight, bias, unfolded_inputs, unfolded_kernel = ctx.saved_tensors
        batch, ci, hi, wi = inputs.shape
        batch, co, ho, wo = grad_output.shape
        co, ci, kh, kw = weight.shape
        padding = ctx.padding
        stride = ctx.stride 

        # Bias gradient
        bias_gradient = grad_output.sum(dim = (0,2,3)) # dims [b, co, ho, wo] sum over 0,2,3 -> bias applied over co

        # grad_output_unfolded
        grad_output_unfolded = grad_output.view(batch, co, ho*wo)

        # GRAD INPUTS
        inputs_grad_unfolded = torch.matmul(grad_output_unfolded.transpose(1,2), unfolded_kernel)
        
        grad_inputs = F.fold(inputs_grad_unfolded.transpose(1,2), output_size = (hi, wi), kernel_size=(kh, kw),
                             padding = padding, stride = stride)

        #GRAD WEIGHT
        weight_grad_unfolded = torch.bmm(grad_output_unfolded, unfolded_inputs.transpose(1,2))
        grad_weight = weight_grad_unfolded.sum(0).view(co, ci, kh, kw)

        return (
            grad_inputs,
            grad_weight,
            bias_gradient,
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
    

"""
This module contains the code for the Conv2d.
"""

# 3pps


class InterConv2d(torch.nn.Module):
    """
    This is the class to implement the Conv2d.

    Attributes:
        weight: weight tensor. Dimensions: [output channels, 
            input channels, kernel size, kernel size].
        bias: bias tensor. Dimensions: [output channels].
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """
        This method is the constructor of Conv2d.

        Args:
            in_channels: input channels to the layer.
            out_channels: output channels to the layer.
            kernel_size: kernel size for the layer.

        Returns:
            None.
        """
        
        # Call super class
        super().__init__()
        
        # Set attributes
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Define parameters
        self.weight = torch.nn.Parameter(
            torch.rand(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = torch.nn.Parameter(
            torch.rand(out_channels)
        )

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is thf forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, input channels, 
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, output channels, 
                (height - kernel size + 1), (width - kernel size + 1)].
        """

        # TODO

        b, ci, hi, wi = inputs.shape
        out_h = (hi - self.kernel_size + 1)
        out_w = (wi - self.kernel_size + 1)    
        co = self.out_channels

        outputs = torch.zeros(b, co, out_h, out_w)

        for ho in range(out_h):
            for wo in range(out_w):

                patch = inputs[:,:, ho : ho + self.kernel_size, wo : wo + self.kernel_size] # (batch, ci, hi, wi)
                patch = patch.unsqueeze(1)

                outputs[:,:,ho,wo] = (
                    patch * self.weight.unsqueeze(0) 
                ).sum(dim=(2,3,4)) + self.bias # (b, co, ci, hi, wi) -> (b,co,ho,wo)
        
        return outputs

        


    

"""
Dropout Layer
"""


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
        




"""
Recurrent Layers
"""




class RNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the RNN.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the forward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: first hidden state. Dimensions: [1, batch,
                hidden size].
            weight_ih: weight for the inputs.
                Dimensions: [hidden size, input size].
            weight_hh: weight for the inputs.
                Dimensions: [hidden size, hidden size].
            bias_ih: bias for the inputs.
                Dimensions: [hidden size].
            bias_hh: bias for the inputs.
                Dimensions: [hidden size].


        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        # TODO
        b, sequence, input_size = inputs.shape
        h_prev = h0
        _,_, hidden_size = h0.shape

        output = torch.zeros(b, sequence, hidden_size, dtype = inputs.dtype)

        for ts in range(sequence):
            x_t = inputs[:,ts,:]    
            
            z_inputs = x_t @ weight_ih.T + bias_ih
            z_h = h_prev @ weight_hh.T + bias_hh
            
            h = z_h + z_inputs 
            h *= (h>=0).float() # aply relu
            
            h_prev = h
            output[:,ts,:] = h

        ctx.save_for_backward(
            h0,
            output, 
            inputs,
            weight_hh,
            weight_ih
        )
        return output, h_prev



    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor, grad_hn: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This method is the backward of the RNN.

        Args:
            ctx: context for loading elements from the forward.

            grad_output: outputs gradients. Dimensions: [batch, sequence, hidden size].
            grad_hn: final hidden state gradients. Dimensions: [1, batch, hs].


        Returns:
            inputs gradients. Dimensions: [batch, sequence,
                input size].
            h0 gradients state. Dimensions: [1, batch,
                hidden size].
            weight_ih gradient. Dimensions: [hidden size,
                input size].
            weight_hh gradients. Dimensions: [hidden size,
                hidden size].
            bias_ih gradients. Dimensions: [hidden size].
            bias_hh gradients. Dimensions: [hidden size].
        """

        # TODO
        h0, output, inputs, weight_hh, weight_ih = ctx.saved_tensors
        b, sequence, hidden_size = output.shape
        b, sequence, input_size = inputs.shape

        grad_ho = grad_hn.clone()
        
        # Initialize all gradients

        grad_inputs = torch.zeros_like(inputs)
        grad_weight_ih = torch.zeros_like(weight_ih)
        grad_weight_hh = torch.zeros_like(weight_hh)
        grad_bias_ih =torch.zeros(hidden_size)
        grad_bias_hh = torch.zeros(hidden_size)

        for ts in reversed(range(sequence)):
            
            dh = grad_output[:,ts,:] + grad_ho
            grad_relu = (output[:,ts,:]>0).float()

            grad_z = dh*grad_relu
            grad_z = grad_z.squeeze()
            # Inputs grad:
            grad_inputs[:,ts,:] = grad_z @ weight_ih
            grad_weight_ih += torch.matmul(grad_z.T, inputs[:,ts,:])
            grad_bias_ih += grad_z.sum(0)

            # Hidden grad
            h_prev = output[:,ts-1,:] if ts > 0 else h0.squeeze(0)
            grad_weight_hh += grad_z.T @ h_prev
            grad_bias_hh += grad_z.sum(0)
            grad_ho = grad_z @ weight_hh

        return(
            grad_inputs,
            grad_ho.unsqueeze(0),
            grad_weight_ih,
            grad_weight_hh,
            grad_bias_ih,
            grad_bias_hh,
        )



        
        
class RNN(torch.nn.Module):
    """
    This is the class that represents the RNN Layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        """
        This method is the constructor of the RNN layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.hidden_size = hidden_size
        self.weight_ih: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, input_dim)
        )
        self.weight_hh: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_ih: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))
        self.bias_hh: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))

        # init parameters corectly
        self.reset_parameters()

        self.fn = RNNFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, sequence,
                input size].
            h0: initial hidden state.

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        return self.fn(
            inputs, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh
        )

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

        return None






"""
Embedding Layer

Development of backward pass of the Embeddings layer. No functions from nn package can be used.
Only 1 for-loop can be used.

"""


class EmbeddingFuncion(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Embedding layer.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        padding_idx: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the Embedding layer.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        # compute embeddings
        outputs: torch.Tensor = weight[inputs, :]

        # save tensors for the backward
        ctx.save_for_backward(inputs, weight, torch.tensor(padding_idx))

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        """
        This method is the backward of the Embedding layer.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions:
                [batch, output dim].

        Returns:
            None value.
            inputs gradients. Dimensions: [batch].
            None value.
        """

        # TODO

        inputs, weight, padding_idx = ctx.saved_tensors
        batch = inputs.shape[0]
        grad_inputs = torch.zeros_like(weight)
        
        for b in range(batch): 
            
            word_idx = inputs[b]

            if word_idx != padding_idx:
                grad_inputs[word_idx] += grad_outputs[b]

        return None, grad_inputs, None



class Embedding(torch.nn.Module):
    """
    This is the class that represents the Embedding Layer.
    """

    padding_idx: int

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ) -> None:
        """
        This method is the constructor of the Embedding layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )

        # init parameters corectly
        self.reset_parameters()

        # set padding idx
        self.padding_idx = padding_idx if padding_idx is not None else -1

        self.fn = EmbeddingFuncion.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(inputs, self.weight, self.padding_idx)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)





"""
LSTM 
"""



class LSTMFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of the LSTM.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LSTM.

        Args:
            ctx: context for saving elements for backward.
            inputs: input tensor. [batch, sequence, input size]
            h0: initial hidden state. [1, batch, hidden size]
            c0: initial cell state. [1, batch, hidden size]
            weight_ih: weights for input. [4 * hidden size, input size]
            weight_hh: weights for hidden state. [4 * hidden size, hidden size]
            bias_ih: bias for input. [4 * hidden size]
            bias_hh: bias for hidden state. [4 * hidden size]

        Returns:
            outputs: [batch, sequence, hidden size]
            hn: final hidden state. [1, batch, hidden size]
            cn: final cell state. [1, batch, hidden size]
        """

        # TODO
        b, sequence, input_size = inputs.shape
        _,_, hidden_size = h0.shape
        
        h_prev = h0
        c_prev = c0 


        # Input-forget-cell-output order:

        wi_i, wi_f ,wi_c , wi_o = weight_ih.view(4, hidden_size, input_size)
        bii, bif, bic, bio = bias_ih.view(4, hidden_size)

        wh_i, wh_f ,wh_c , wh_o = weight_hh.view(4, hidden_size, hidden_size)
        bhi, bhf, bhc, bho = bias_hh.view(4, hidden_size)

        
        outputs = torch.zeros(b, sequence, hidden_size, dtype = inputs.dtype)

        for ts in range(sequence):
            x = inputs[:,ts,:]

            # Forget gate
            z_f = (x @ wi_f.T + bif) + (h_prev @ wh_f.T + bhf)
            forget_gate = torch.sigmoid(z_f)

            # Input gate:
            z_i = (x @ wi_i.T + bii) + (h_prev @ wh_i.T + bhi)
            input_gate = torch.sigmoid(z_i)

            # Cell candidates:
            z_c = (x @ wi_c.T + bic) + (h_prev @ wh_c.T + bhc)
            cell_candidate = torch.tanh(z_c)

            # Output gate:
            z_o = (x @ wi_o.T + bio) + (h_prev @ wh_o.T + bho)
            output_gate = torch.sigmoid(z_o)

            # Compute next states:
            c = forget_gate * c_prev + input_gate * cell_candidate
            h = output_gate * (torch.tanh(c))

            h_prev = h
            c_prev = c

            outputs[:,ts,:] = h
        
        return outputs, h_prev, c_prev


class LSTM(torch.nn.Module):
    """
    Custom LSTM layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.weight_ih = torch.nn.Parameter(torch.empty(4 * hidden_size, input_dim))
        self.weight_hh = torch.nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.empty(4 * hidden_size))

        self.reset_parameters()
        self.fn = LSTMFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.fn(inputs, h0, c0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)



"""
GRU: Gated Recurrent Unit
"""




class GRUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of the GRU.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GRU.

        Args:
            ctx: context for saving elements for backward.
            inputs: input tensor. [batch, sequence, input size]
            h0: initial hidden state. [1, batch, hidden size]
            weight_ih: weights for input. [3 * hidden size, input size]
            weight_hh: weights for hidden state. [3 * hidden size, hidden size]
            bias_ih: bias for input. [3 * hidden size]
            bias_hh: bias for hidden state. [3 * hidden size]

        Returns:
            outputs: [batch, sequence, hidden size]
            hn: final hidden state. [1, batch, hidden size]
        """
        # TODO: implement the forward pass

        # Reset-Update-n
        _, b, hidden_size = h0.shape
        b, sequence, input_size = inputs.shape
        h_prev = h0
        outputs = torch.zeros(b, sequence, hidden_size, dtype=inputs.dtype)

        wir, wiz, win = weight_ih.view(3, hidden_size, input_size)
        bir, biz, bin = bias_ih.view(3, hidden_size)

        whr, whz, whn = weight_hh.view(3, hidden_size, hidden_size)
        bhr, bhz, bhn = bias_hh.view(3, hidden_size)

        for ts in range(sequence):

            x = inputs[:,ts,:]

            # Reset gate:
            r_t = torch.sigmoid(
                (x @ wir.T + bir) + (h_prev @ whr.T + bhr)
            )

            # Update gate:
            z_t = torch.sigmoid(
                (x @ wiz.T + biz) + (h_prev @ whz.T + bhz)
            )

            n_t = torch.tanh(
                (x @ win.T + bin) + r_t * (h_prev @ whn.T + bhn)
            )

            # Compute next state
            h = (1-z_t)*n_t + z_t*h_prev
            
            h_prev = h
            outputs[:,ts,:] = h
        
        return outputs, h_prev
        


class GRU(torch.nn.Module):
    """
    Custom GRU layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.weight_ih = torch.nn.Parameter(torch.empty(3 * hidden_size, input_dim))
        self.weight_hh = torch.nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.empty(3 * hidden_size))

        self.reset_parameters()
        self.fn = GRUFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fn(inputs, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)
