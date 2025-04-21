# deep learning libraries
import torch
import numpy as np


# other libraries
import os
import math
import random


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