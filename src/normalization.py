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
        super().__init__()

        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

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
        x = inputs.clone()
        b, c = inputs.shape[:2]
        x = x.view(b, self.num_groups, c // self.num_groups, -1)
        mean_x = x.mean(dim=(2, 3), keepdim=True)
        var_x = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        norm = (x - mean_x) / (var_x + self.eps).sqrt()
        norm = norm.view(inputs.shape)

        if self.affine:
            w = self.weight.unsqueeze(0).unsqueeze(-1)
            b = self.bias.unsqueeze(0).unsqueeze(-1)
            return (w * norm + b).view(inputs.shape)
        else:
            return norm

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

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        """
        This method is the constructor of GroupNorm class.

        Args:
            num_groups: Number of groups.
            num_channels: Number of channels.
            eps: epsilon to avoid division by 0. Defaults to 1e-5.
            affine: Indicator to perform affine transformation. Defaults to True.
        """
        super().__init__()

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels))
            self.bias = torch.nn.Parameter(torch.empty(num_channels))
            self.reset_parameters()

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
        x = inputs.clone()
        mean_x = x.mean(dim=(0, 2, 3), keepdim=True)
        var_x = x.var(dim=(0, 2, 3), keepdim=True, unbiased = False)

        norm = (x - mean_x) / (var_x + self.eps).sqrt()

        if self.affine:
            w = self.weight.unsqueeze(-1).unsqueeze(-1)
            b = self.bias.unsqueeze(-1).unsqueeze(-1)
            return w * norm + b
        else:
            return norm

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)


class LayerNorm(torch.nn.Module):
    """
    This class implements the LayerNorm layer of torch.

    Attributes:
        num_features: Number of features.
        eps: epsilon to avoid division by 0.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        """
        This method is the constructor of LayerNorm class.

        Args:
            num_features: Number of features.
            eps: epsilon to avoid division by 0. Defaults to 1e-5.
            affine: Indicator to perform affine transformation. Defaults to True.

        Returns:
            None.
        """
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features))
            self.bias = torch.nn.Parameter(torch.empty(num_features))
            self.reset_parameters()

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
        x = inputs.clone()
        mean_x = x.mean(dim=(1, 2, 3), keepdim=True)
        var_x = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)

        norm = (x - mean_x) / (var_x + self.eps).sqrt()

        if self.affine:
            w = self.weight.unsqueeze(-1).unsqueeze(-1)
            b = self.bias.unsqueeze(-1).unsqueeze(-1)
            return w * norm + b
        else:
            return norm

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)


class InstanceNorm(torch.nn.Module):
    """
    This class implements the InstanceNorm layer of torch.

    Attributes:
        num_features: Number of features.
        eps: epsilon to avoid division by 0.
        affine: Indicator to perform affine transformation.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        """
        This method is the constructor of InstanceNorm class.

        Args:
            num_features: Number of features.
            eps: epsilon to avoid division by 0. Defaults to 1e-5.
            affine: Indicator to perform affine transformation. Defaults to True.

        Returns:
            None.
        """
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features))
            self.bias = torch.nn.Parameter(torch.empty(num_features))
            self.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, channels, height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, channels, height, width].
        """
        x = inputs.clone()
        mean_x = x.mean(dim=(2, 3), keepdim=True)
        var_x = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        norm = (x - mean_x) / (var_x + self.eps).sqrt()

        if self.affine:
            w = self.weight.unsqueeze(-1).unsqueeze(-1)
            b = self.bias.unsqueeze(-1).unsqueeze(-1)
            return w * norm + b
        else:
            return norm

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
