
from typing import Type, Tuple

import torch
from torch import nn
from torch import optim

from .config import ELabConfig

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# TODO: I can preseve the tokenizer in the ELab class.

class ELab:
    def print(self, *args, **kwargs):
        if self.verbose and (not self._ddp or self.rank == 0):
            print("[ELab] ", *args, **kwargs)

    def __init__(self, 
                 config: ELabConfig | str, 
                 model: nn.Module, 
                 optimizer: optim.Optimizer,
                 
                 # Distributed training parameters
                 ddp: bool = False,
                 rank: int = 0, world_size:int = 1,
                 backend: str = "nccl",
                 master_addr: str = "localhost",
                 master_port: int = 12355,
                 verbose = True):

        self._ddp = ddp
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self.verbose = verbose

        self.print("Instantiating ELab ...")

        if isinstance(config, str):
            config = ELabConfig.from_path(config)
        self.config = config


        # configurate the device and ddp settings
        if self._ddp:
            self.print("  - Using DDP.")
            os.environ['MASTER_ADDR'] = self.master_addr
            os.environ['MASTER_PORT'] = str(self.master_port)
            init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size)
            self.device = f"cuda:{self.rank}"
            torch.cuda.set_device(self.rank)
            
        else:    
            # Define the device
            device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.print("- Using device:", device_name)
            if (device_name == 'cuda'):
                self.print(f"  - CUDA version: {torch.version.cuda}")
                self.print(f"  - Device name: {torch.cuda.get_device_name(device_name.index)}") # type: ignore
                self.print(f"  - Device memory: {torch.cuda.get_device_properties(device_name.index).total_memory / 1024 ** 3} GB") # type: ignore
            self.device = torch.device(device_name)
        
        self.print(f"- checkpoint: {config.proj_path}")

        self.model_class = model.__class__
        self.model = model.to(self.device)
        if self._ddp:
            self.model = DDP(self.model, device_ids=[self.rank])
        self.optim_class = optimizer.__class__
        self.optimizer = optimizer

        self.print("  - Model Type: ", self.model_class)
        self.print("  - Optimizer Type: ", self.optim_class)


        checkpoint_path = config.get_checkpoint_file_path()
        if checkpoint_path is not None:
            self.print(f"  Loading checkpoint from {checkpoint_path} ...")

            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=self.device)

            # try to recover the states
            if "model_state_dict" in checkpoint:
                module = self.model.module if self._ddp else self.model
                module.load_state_dict(checkpoint["model_state_dict"])
                self.print("  - Model state loaded.")
            else:
                self.print("  - No model state found.")
            
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.print("  - Optimizer state loaded.")
            else:
                self.print("  - No optimizer state found.")

            if "global_step" in checkpoint:
                self.global_step = checkpoint["global_step"]
                self.print("  - Global step: ", self.global_step)
            else:
                self.global_step = 0
                self.print("  - No global step found. Set to 0.")
                
            if "proc_tokens" in checkpoint:
                self.proc_tokens = checkpoint["proc_tokens"]
                self.print("  - Processed tokens: ", self.proc_tokens)
            else:
                self.proc_tokens = 0
                self.print("  - No processed tokens found. Set to 0.")

        else:
            self.print("  No checkpoint found.")
            self.global_step = 0
            self.proc_tokens = 0

        self.print("Instantiation complete.")


    @property
    def model_size(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
    

    def save_checkpoint(self, postfix: str | None = None):
        '''
        Save the checkpoint.
        If postfix is None, use the global_step as postfix.
        '''
        if self._ddp:
            return
        
        if postfix is None:
            postfix = f"{self.global_step:08d}"
        
        self.config.checkpoint_path.mkdir(parents=True, exist_ok=True)

        checkpoint_filename = self.config.checkpoint_path / f"{self.config['elab_name']}-{self.config['elab_version']}-{postfix}.pt"

        self.print(F"Saving ELab instance to {checkpoint_filename} ...")

        module = self.model.module if self._ddp else self.model

        torch.save({
            'model_state_dict': module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'proc_tokens': self.proc_tokens,
        }, checkpoint_filename)

        self.print("Checkpoint saved.")

    def add_proc_tokens(self, extra_tokens: int):
        '''
        Add additional processed tokens to the counter.
        Broadcast the tokens to all processes if DDP is enabled.
        '''
        if not self._ddp:
            self.proc_tokens += extra_tokens
        else:
            # Define the tensor to broadcast
            extra_tokens_broadcast = torch.tensor([extra_tokens]).to(self.device)

            # Broadcast the extra tokens to all processes in the group
            dist.broadcast(extra_tokens_broadcast, src=self.rank)
            dist.all_reduce(extra_tokens_broadcast, op=dist.ReduceOp.SUM)
            self.proc_tokens += extra_tokens_broadcast.item()

    def train(self):
        raise NotImplementedError("train() is not implemented.")
    

    def __del__(self):
        if self._ddp:
            destroy_process_group()


