import torch
from torch import nn

def get_parameter_size(model: nn.Module) -> int:
    """
    Get the total number of parameters in the model.

    Args:
        model (nn.Module): The PyTorch model instance.

    Returns:
        int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Get the norm of the gradients of the model.

    Args:
        model (nn.Module): The PyTorch model instance.
        norm_type (float, optional): The type of the used norm. Defaults to 2.0.

    Returns:
        float: The norm of the gradients of the model.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm