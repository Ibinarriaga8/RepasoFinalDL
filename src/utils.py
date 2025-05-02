# deep learning libraries
import torch
import numpy as np


# other libraries
import os
import math
import random
import torch.nn.functional as F

@torch.no_grad()
def parameters_to_double(model: torch.nn.Module) -> None:
    """
    This function transforms the model parameters to double.

    Args:
        model: pytorch model.
    """

    # TODO
    for parameter in model.parameters():
        parameter.data = parameter.data.double()
        
    return None

def get_dropout_random_indexes(shape: torch.Size, p: float) -> torch.Tensor:
    """
    This function get the indexes to put elements at zero for the
    dropout layer. It ensures the elements are selected following the
    same implementation than the pytorch layer.

    Args:
        shape: shape of the inputs to put it at zero. Dimensions: [*].
        p: probability of the dropout.

    Returns:
        indexes to put elements at zero in dropout layer.
            Dimensions: shape.
    """

    # get inputs indexes
    inputs: torch.Tensor = torch.ones(shape)

    # get indexes
    indexes: torch.Tensor = F.dropout(inputs, p)
    indexes = (indexes == 0).int()

    return indexes
