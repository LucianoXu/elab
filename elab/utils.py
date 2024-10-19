import torch
from torch import nn

def get_parameter_size(model: nn.Module) -> int:
    """
    Get the total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)