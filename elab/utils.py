from typing import Optional
import torch
from torch import nn
from tqdm import tqdm

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


def set_adamw_params(
        optimizer: torch.optim.adamw.AdamW, 
        r: Optional[float] = None, 
        betas: Optional[float] = None, 
        eps: Optional[float] = None, 
        weight_decay: Optional[float] = None) -> torch.optim.adamw.AdamW:
    """
    Set the parameters of the AdamW optimizer. In-place operation.
    None values will not change the corresponding parameter.

    Args:
        optimizer (torch.optim.adamw.AdamW): The AdamW optimizer instance.
        r (float, optional): The learning rate. Defaults to None.
        betas (float, optional): The betas. Defaults to None.
        eps (float, optional): The epsilon. Defaults to None.
        weight_decay (float, optional): The weight decay. Defaults to None.

    Returns:
        torch.optim.adamw.AdamW: The AdamW optimizer instance.
    """

    if r is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = r

    if betas is not None:
        for param_group in optimizer.param_groups:
            param_group['betas'] = betas

    if eps is not None:
        for param_group in optimizer.param_groups:
            param_group['eps'] = eps

    if weight_decay is not None:
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay

    return optimizer


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def generate(
        model: nn.Module, 
        prompt_ids: list[int],
        max_len: int,
        EOT_id: Optional[int] = None,
        T: float = 0.6,
        p_threshold: float = 0.95,
        include_prompt: bool = True,
        show_progress: bool = True) -> list[int]:
    '''
    Generate a sequence of tokens from the model.

    Args:
        model(nn.Module): The PyTorch model instance with a forward method.

        prompt_ids(list[int]): The prompt token ids.

        max_len(int, optional): The maximum length of the generated sequence. If None, the generation will continue until the end-of-text token is generated. Defaults to None.

        EOT_id(int, optional): The end-of-text token id. If None, the generation will continue until the maximum length is reached. Defaults to None.

        T(float, optional): The temperature for sampling. Defaults to 0.6.

        p_threshold(float, optional): The probability threshold for top-p sampling. Defaults to 0.95.

        include_prompt(bool, optional): Whether to include the prompt in the generated sequence. Defaults to True.

        show_progress(bool, optional): Whether to show the progress bar. Defaults to True.

    Returns:
        list[int]: The generated token ids. Depending on `include_prompt`, the prompt may or may not be included in the generated sequence.
    '''
    device = model.parameters().__next__().device

    inputs = torch.tensor([prompt_ids], dtype=torch.long, device = device)
    generated = torch.tensor([[]], dtype=torch.long, device = device)

    if show_progress:
        it = tqdm(range(len(prompt_ids), max_len))
    else:
        it = range(len(prompt_ids), max_len)

    for i in it:
        output = model.forward(inputs)
        logits = output[:, -1, :]
        logits = logits / T
        probs = torch.softmax(logits, dim=-1)
        next_token = sample_top_p(probs, p_threshold)
        inputs = torch.cat([inputs, next_token], dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)
        if next_token == EOT_id:
            break
    
    res = generated[0].tolist()
    if include_prompt:
        res = prompt_ids + res
    return res
        
def generate_batched(
        model: nn.Module, 
        prompt_ids: list[list[int]],
        max_len: int,
        EOT_id: Optional[int] = None,
        T: float = 0.6,
        p_threshold: float = 0.95,
        include_prompt: bool = True,
        show_progress: bool = True) -> list[list[int]]:

    '''
    Generate sequences of tokens from the model. This function is similar to `generate`, but it generates multiple sequences in parallel.

    The only different argument from `generate` is `prompt_ids`, which is a list of prompts. Each element in the list is a prompt.

    Args:
        prompt_ids(list[list[int]]): The prompt token ids. Each element in the list is a prompt. The length of each prompt can be different.

    Returns:
        list[list[int]]: The generated token ids. Depending on `include_prompt`, the prompt may or may not be included in the generated sequence. The length of each can be different.
    '''
    
    device = model.parameters().__next__().device

    # the unfinished prompts. the index is the index of the prompt in the prompt_ids
    unfinished_prompt_ids = {i: prompt for i, prompt in enumerate(prompt_ids)}

    if include_prompt:
        res = prompt_ids.copy()
    else:
        res = [[] for _ in range(len(prompt_ids))]

    # start from the shortest prompt
    min_len = min(len(prompt) for prompt in prompt_ids)

    if show_progress:
        it = tqdm(range(min_len, max_len))
    else:
        it = range(min_len, max_len)

    for current_len in it:
        if len(unfinished_prompt_ids) == 0:
            break

        # collect the current batch of prompt ids

        # the current batch of prompt ids
        current_batch_ids = []
        # the indices of each prompt in the current batch
        current_batch_index = []

        for i in unfinished_prompt_ids:
            if len(unfinished_prompt_ids[i]) == current_len:
                current_batch_ids.append(unfinished_prompt_ids[i])
                current_batch_index.append(i)

        # inference and sample the next tokens
        inputs = torch.tensor(current_batch_ids, dtype=torch.long, device = device)
        output = model.forward(inputs)
        logits = output[:, -1, :]
        logits = logits / T
        probs = torch.softmax(logits, dim=-1)
        next_tokens = sample_top_p(probs, p_threshold)

        # update the prompts
        for i in range(len(current_batch_index)):
            next_token = next_tokens[i][0].item()
            res[current_batch_index[i]].append(next_token)
            if next_token == EOT_id:
                del unfinished_prompt_ids[current_batch_index[i]]

    return res