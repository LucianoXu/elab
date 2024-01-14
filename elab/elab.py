
from typing import Type, Tuple

import torch
from torch import nn
from torch import optim

from .config import ELabConfig

# TODO: I can preseve the tokenizer in the ELab class.

class ELab:
    def __init__(self, 
                 config: ELabConfig | str, 
                 model: Type[nn.Module] | nn.Module, 
                 optimizer: Type[optim.Optimizer] | optim.Optimizer):

        print()
        print("Instantiating ELab ...")

        if isinstance(config, str):
            config = ELabConfig.from_path(config)
        self.config = config

        # Define the device
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print("- Using device:", device_name)
        if (device_name == 'cuda'):
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - Device name: {torch.cuda.get_device_name(device_name.index)}") # type: ignore
            print(f"  - Device memory: {torch.cuda.get_device_properties(device_name.index).total_memory / 1024 ** 3} GB") # type: ignore
        self.device = torch.device(device_name)
        
        print(f"- ELab checkpoint: {config.proj_path}")
        if isinstance(model, nn.Module):
            self.model_class = model.__class__
            self.model = model.to(self.device)
        else:
            self.model_class = model
            self.model = model(**config["model_args"]).to(self.device)

        if isinstance(optimizer, optim.Optimizer):
            self.optim_class = optimizer.__class__
            self.optimizer = optimizer
        else:
            self.optim_class = optimizer
            self.optimizer = optimizer(self.model.parameters(), **config["optim_args"])

        print("  - Model Type: ", self.model_class)
        print("  - Optimizer Type: ", self.optim_class)


        checkpoint_path = config.get_checkpoint_file_path()
        if checkpoint_path is not None:
            print(f"  Loading checkpoint from {checkpoint_path} ...")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # try to recover the states
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print("  - Model state loaded.")
            else:
                print("  - No model state found.")
            
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("  - Optimizer state loaded.")
            else:
                print("  - No optimizer state found.")

            if "global_step" in checkpoint:
                self.global_step = checkpoint["global_step"]
                print("  - Global step: ", self.global_step)
            else:
                self.global_step = 0
                print("  - No global step found. Set to 0.")
                
            if "proc_tokens" in checkpoint:
                self.proc_tokens = checkpoint["proc_tokens"]
                print("  - Processed tokens: ", self.proc_tokens)
            else:
                self.proc_tokens = 0
                print("  - No processed tokens found. Set to 0.")

        else:
            print("  No checkpoint found.")
            self.global_step = 0
            self.proc_tokens = 0

        print("Instantiation complete.")
        print()


    @property
    def model_size(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
    

    def save_checkpoint(self, postfix: str | None = None):
        '''
        Save the checkpoint.
        If postfix is None, use the global_step as postfix.
        '''
        if postfix is None:
            postfix = str(self.global_step)
        
        checkpoint_filename = self.config.checkpoint_path / f"{self.config['elab_name']}-{self.config['elab_version']}-{postfix}.pt"

        print(F"Saving ELab instance to {checkpoint_filename} ...", end="")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'proc_tokens': self.proc_tokens,
        }, checkpoint_filename)

        print("Checkpoint saved.")
        



