# Standard libraries
from typing import Any

# 3pps
import torch
import math





class ReLUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the ReLU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
        result = inputs.clone()
        result*= result>0
        ctx.save_for_backward(inputs)
        return result
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method is the backward of the relu.
        Defines how to compute gradient in backpropagation

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
        """

        # TODO
        result, = ctx.saved_tensors
        diff = result > 0 # bool tensor 
        return grad_output * diff


class ReLU(torch.nn.Module):
    """
    This is the class that represents the ReLU Layer.
    """

    def __init__(self):
        """
        This method is the constructor of the ReLU layer.
        """

        # call super class constructor
        super().__init__()

        self.fn = ReLUFunction.apply # define layer function

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: Inputs tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(inputs)



class LeakyReLUFunctional(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the LeakyReLU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, negative_slope: float) -> torch.Tensor:
        """
        This is the forward method of the LeakyReLU.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
        outputs = inputs.clone()
        outputs[outputs < 0] = negative_slope*inputs[outputs < 0] # LeakyReLU
        ctx.save_for_backward(inputs) 
        ctx.negative_slope = negative_slope
        return outputs

        

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method is the backward of the LeakyReLU.
        Defines how to compute gradient in backpropagation

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
        """

        # TODO
        inputs = ctx.saved_tensors[0]
        negative_slope = ctx.negative_slope
        
        grad_inputs = inputs.clone()
        grad_inputs[inputs >= 0] = 1
        grad_inputs[inputs<0] = negative_slope

        return grad_output * grad_inputs, None




class LeakyReLU(torch.nn.Module):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return LeakyReLUFunctional.apply(inputs, self.alpha)




class PreLUFunctional(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the PReLU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight: float) -> torch.Tensor:
        """
        This is the forward method of the PReLU.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].
            weight: Weight tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
        outputs = inputs.clone()
        outputs[outputs < 0] = weight*outputs[outputs < 0]
        ctx.save_for_backward(inputs)
        ctx.weight = weight
        
        return outputs
        
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method is the backward of the PReLU.
        Defines how to compute gradient in backpropagation

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
            Weights gradients. Dimensions: [*].
        """

        # TODO
        inputs = ctx.saved_tensors[0]
        weight = ctx.weight
        grad_inputs = inputs.clone()
        grad_inputs[grad_inputs >= 0] = 1
        grad_inputs[grad_inputs < 0] = weight

        grad_weight = inputs.clone()
        grad_weight = (grad_weight < 0) * inputs

        return grad_output*grad_inputs, grad_output*grad_weight

    
class PReLU(torch.nn.Module):
    def __init__(self, init_weight=0.25):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([init_weight], dtype=torch.float32))


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return PreLUFunctional.apply(inputs, self.weight)



class ELUFunctional(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the ELU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        This is the forward method of the ELU.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].
            alpha: Alpha parameter. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO

        outputs = inputs.clone()
        outputs[inputs<=0] = alpha*(torch.exp(outputs[inputs<0]) - 1)
        ctx.save_for_backward(inputs)
        ctx.alpha = alpha
        return outputs

    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method is the backward of the ELU.
        Defines how to compute gradient in backpropagation

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
            None.
        """
        # TODO

        inputs = ctx.saved_tensors[0]
        alpha = ctx.alpha

        grad_inputs = inputs.clone()
        grad_inputs[inputs>0] = 1
        grad_inputs[inputs<=0] = alpha * torch.exp(inputs[inputs<=0])

        return grad_output*grad_inputs, None
    

class ELU(torch.nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return ELUFunctional.apply(inputs, self.alpha)



class SELUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the SELU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, alpha: float, scale: float) -> torch.Tensor:
        """
        This is the forward method of the SELU.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].
            alpha: Alpha parameter. Dimensions: [*].
            scale: Scale parameter. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
           
        outputs = inputs.clone()
        outputs[inputs>0] *= scale
        outputs[inputs<=0] = scale*alpha*(torch.exp(inputs[inputs<=0])-1)
        ctx.save_for_backward(inputs)
        ctx.scale =scale
        ctx.alpha = alpha
        return outputs

    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method is the backward of the SELU.
        Defines how to compute gradient in backpropagation

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
            None.
            None.
        """
        # TODO
        inputs = ctx.saved_tensors[0]
        alpha, scale = ctx.alpha, ctx.scale

        grad_inputs = inputs.clone()
        grad_inputs[inputs>0] = scale
        grad_inputs[inputs<=0] = scale*alpha*torch.exp(inputs[inputs<=0])

        return grad_output*grad_inputs, None, None


class SELU(torch.nn.Module):
    """
    Class for the implementation of the forward and backward pass of
    the SELU.
    """

    def __init__(self, alpha: float = 1.67326, scale: float = 1.0507):
        super().__init__()
        self.alpha = alpha
        self.scale = scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return SELUFunction.apply(inputs, self.alpha, self.scale)


class HardShrinkFunctional(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the HardShrink.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, lambd: float) -> torch.Tensor:
        """
        This is the forward method of the HardShrink.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].
            lambd: Lambda parameter. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
        outputs = inputs.clone()
        outputs[(-lambd < inputs) & (inputs<lambd)] = 0
        ctx.save_for_backward(inputs)
        ctx.lambd = lambd
        return outputs
        
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method is the backward of the HardShrink.
        Defines how to compute gradient in backpropagation

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
            None.
        """
        # TODO
        inputs = ctx.saved_tensors[0]
        lambd = ctx.lambd

        inputs_grad = (inputs>lambd) | (inputs<-lambd) # .to(torch.float32) # boolean tensor
        
        return grad_output*inputs_grad, None
    
class HardShrink(torch.nn.Module):
    """
    Class for the implementation of the forward and backward pass of
    the HardShrink.
    """

    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HardShrinkFunctional.apply(inputs, self.lambd)



class HardTanhFunctional(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the HardTanh.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """
        This is the forward method of the HardTanh.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].
            min_val: Minimum value. Dimensions: [*].
            max_val: Maximum value. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO

        outputs = inputs.clone()
        outputs[inputs>max_val] = max_val
        outputs[inputs<min_val] = min_val
        mask = (min_val<inputs)&(inputs<max_val)
        
        ctx.save_for_backward(inputs, mask)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method is the backward of the HardTanh.
        Defines how to compute gradient in backpropagation

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
            None.
            None.
        """
        # TODO
        inputs, mask = ctx.saved_tensors

        grad_inputs = mask
        return grad_output*grad_inputs, None, None
    

class HardTanh(torch.nn.Module):
    """
    Class for the implementation of the forward and backward pass of
    the HardTanh.
    """

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HardTanhFunctional.apply(inputs, self.min_val, self.max_val)



class HardSwishFunctional(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the HardSwish.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward method of the HardSwish.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
        outputs = inputs.clone()
        outputs[inputs<= -3] = 0
        mask = (-3<inputs) & (inputs<3)
        outputs[mask] = 1/6*inputs[mask]*(inputs[mask]+3)

        ctx.save_for_backward(inputs, mask) 
        return outputs
        
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method is the backward of the HardSwish.
        Defines how to compute gradient in backpropagation

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
            None.
            None.
        """
        # TODO
        inputs, mask = ctx.saved_tensors
        
        grad_inputs = inputs.clone()
        grad_inputs[mask] = inputs[mask]*1/3 + 1/2
        grad_inputs[inputs<=-3] = 0
        grad_inputs[inputs>=3] = 1

        return grad_output*grad_inputs


class HardSwish(torch.nn.Module):
    """
    Class for the implementation of the forward and backward pass of
    the HardSwish.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HardSwishFunctional.apply(inputs)
    


class MaxoutFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Maxout.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weights_first: torch.Tensor,
        bias_first: torch.Tensor,
        weights_second: torch.Tensor,
        bias_second: torch.Tensor,
    ) -> torch.Tensor:
        """
        This is the forward method of the MaxFunction.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        # TODO

        # compute forward
        z1 = inputs @ weights_first.T + bias_first
        z2 = inputs @ weights_second.T + bias_second

        mask = z1>=z2
        outputs = z1.clone()
        
        outputs[mask] = z1[mask]
        outputs[~mask] = z2[~mask]

        ctx.save_for_backward(inputs, weights_first, bias_first, weights_second, bias_second, z1, z2)

        return outputs






    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward of the Maxout.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions:
                [batch, output dim].

        Returns:
            inputs gradients. Dimensions: [batch, input dim].
            gradients for the first weights. Dimensions:
                [output dim, input dim].
            gradients for the first bias. Dimensions: [output dim].
            gradient for the second weights. Dimensions: [output dim,
                input dim].
            gradient for the second bias. Dimensions: [output dim].
        """

        # TODO
        # load tensors from the forward
        inputs, w1, b1, w2, b2, z1, z2 = ctx.saved_tensors

        mask = z1 >= z2
        z1_grad = grad_output*mask.float() # *: para que se mantengan las dimensiones de grad_output
        # Si aplicamos la máscara sólo, de las 64 x 20, se conservarian solo 623 (de 64x20)
        z2_grad = grad_output*(~mask).float()

        weights1_grad = z1_grad.T@ inputs
        weights2_grad =z2_grad.T @ inputs
        b1_grad = z1_grad.sum(dim = 0)
        b2_grad = z2_grad.sum(dim = 0)

        inputs_grad_z1 = z1_grad @ w1
        inputs_grad_z2 = z2_grad @ w2
        inputs_grad = inputs_grad_z1 + inputs_grad_z2


        return inputs_grad, weights1_grad, b1_grad, weights2_grad, b2_grad             


class HardSigmoidFunctional(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the HardSigmoid.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward method of the HardSigmoid.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
        outputs = inputs.clone()
        outputs[inputs <= -3] = 0
        outputs[inputs >= 3] = 1

        mask = (inputs >-3) & (inputs<3)
        outputs[mask] = 1/2 + inputs[mask]*1/6
        
        ctx.save_for_backward(inputs, mask)
        return outputs


    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method is the backward of the HardSigmoid.
        Defines how to compute gradient in backpropagation

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
        """

        # TODO
        inputs, mask = ctx.saved_tensors

        grad_inputs = grad_output.clone()
        grad_inputs[mask] *=1/6
        grad_inputs[~mask] = 0

        return grad_inputs


class HardSigmoid(torch.nn.Module):
    """
    Class for the implementation of the forward and backward pass of
    the HardSigmoid.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HardSigmoidFunctional.apply(inputs)
        

        

class Maxout(torch.nn.Module):
    """
    This is the class that represents the Maxout Layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Maxout layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weights_first: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias_first: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim)
        )
        self.weights_second: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias_second: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim)
        )

        # init parameters corectly
        self.reset_parameters()

        self.fn = MaxoutFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(
            inputs,
            self.weights_first,
            self.bias_first,
            self.weights_second,
            self.bias_second,
        )

    @torch.no_grad()
    def set_parameters(
        self,
        weights_first: torch.Tensor,
        bias_first: torch.Tensor,
        weights_second: torch.Tensor,
        bias_second: torch.Tensor,
    ) -> None:
        """
        This function is to set the parameters of the model.

        Args:
            weights_first: weights for the first branch.
            bias_first: bias for the first branch.
            weights_second: weights for the second branch.
            bias_second: bias for the second branch.
        """

        # set attributes
        self.weights_first = torch.nn.Parameter(weights_first)
        self.bias_first = torch.nn.Parameter(bias_first)
        self.weights_second = torch.nn.Parameter(weights_second)
        self.bias_second = torch.nn.Parameter(bias_second)

        return None

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weights_first, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weights_second, a=math.sqrt(5))
        if self.bias_first is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights_first)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_first, -bound, bound)
            torch.nn.init.uniform_(self.bias_second, -bound, bound)

        return None
