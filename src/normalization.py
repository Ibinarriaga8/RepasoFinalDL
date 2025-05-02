"""
This module contains the code to implement the GroupNorm.
"""

# 3pps
import torch



class GroupNorm(torch.nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        This is the constructor of the GroupNorm class.

        Args:
            num_groups: number of groups to use.
            num_channels: number of channels to use.
            eps: epsilon to avoid overflow. Defaults to 1e-5.
            affine: Indicator to perform affine transformation.
                Defaults to True.
        """

        # call super class constructor
        super().__init__()

        # save attributes
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

        # create parameters if it is an affine transformation
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.empty(num_channels, dtype=dtype))

        self.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass of the module.

        Args:
            inputs: input tensor. Dimensions: [batch, channels, *].

        Returns:
            outputs tensor. Dimensions: [batch, channels, *].
        """

        # TODO
        x = inputs.clone()
        b,c,h,w = inputs.shape

        x = x.view(b, self.num_groups, c//self.num_groups, h, w)
        mean_x = x.mean(dim = (2, 3,4), keepdim = True)
        var_x = x.var(dim = (2,3,4), unbiased = False, keepdim = True)

        outputs = (x-mean_x)/(var_x + self.eps).sqrt()
        outputs = outputs.view(b, c, h, w)
        return outputs

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)



class BatchNorm(torch.nn.Module):
    """
    This class implements the GroupNorm layer of torch.

    Attributes:
        num_groups: Number of groups.
        num_channels: Number of channels.
        eps: epsilon to avoid division by 0.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        """
        This method is the constructor of GroupNorm class.

        Args:
            num_groups: Number of groups.
            num_channels: Number of channels.
            eps: epsilon to avoid division by 0. Defaults to 1e-5.

        Returns:
            None.
        """
        
        # Call super class constructor
        super().__init__()

        # Set attributes
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps



        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, channels,
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, channels, height,
                width].
        """

        # TODO
        
        x = inputs.clone()
        mean_x = x.mean(dim = (1,2,3), keepdim = True) # reduce (1,2,3) dim
        var_x = x.var(dim = (1,2,3), keepdim = True)

        outputs = (inputs - mean_x)/(var_x + self.eps).sqrt()
        return outputs

