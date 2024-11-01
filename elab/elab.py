
from typing import Literal, Type, Tuple, Optional, Any

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from pathlib import Path

class ELab:
    '''
    The ELab class is a utility class for saving and loading PyTorch models and optimizers.

    Attributes:
        folder_path (Path): The folder path where the checkpoint files are stored.

        model (nn.Module): The PyTorch model instance.

        optimizer (Optimizer, optional): The PyTorch optimizer instance. If `None`, the optimizer will not be maintained.

        device (str): The device to load the model and optimizer to.

        states (dict): The states of the ELab object.
    '''
    def _print(self, *args, **kwargs):
        if self.verbose:
            print("[ELab] ", *args, **kwargs, flush=True)

    def __init__(self, 
                 folder_path: str|Path, 
                 ckpt_name: str|Literal['latest', 'none'],
                 model: nn.Module,
                 optimizer: Optional[Optimizer] = None,
                 default_states: Optional[dict[str, Any]] = None,
                 device: Optional[str] = None,
                 verbose: bool = True):
        '''
        Initialize the ELab object by specifying the folder path, the checkpoint to load, as well as the model and optimizer instances.

        Args:
            folder_path (str or Path): The folder path where the checkpoint files are stored.

            ckpt_name (str or Literal['latest', 'none']): If 'none', no checkpoint file will be loaded. Otherwise, the model and optimizer will be loaded from the specified checkpoint file. If `ckpt_name` is 'latest', the latest checkpoint file in the folder will be loaded.

            model (nn.Module): The PyTorch model instance.

            optimizer (Optimizer, optional): The PyTorch optimizer instance. If `None`, only the model is loaded. Defaults to `None`.

            default_states (dict, optional): The default states of the ELab object. Defaults to `None`.

            device (str, optional): The device to load the model and optimizer to. If `None`, the device will be inferred from the checkpoint. Defaults to `None`.

            verbose (bool): Whether to print messages. Defaults to `True`.
        '''
        
        self.verbose = verbose
        
        self._print("ELab initializing at", folder_path)
        self.folder_path: Path = Path(folder_path)

        self._print("Model: ", type(model))
        self.model = model

        self._print("Optimizer: ", type(optimizer))
        self.optimizer = optimizer

        self._print("Device: ", device)
        self.device = device

        self.states = default_states if default_states is not None else {}

        if ckpt_name != 'none':
            self.load(ckpt_name)
    
    def save(self, ckpt_name: str):
        '''
        Save the model, optimizer and states of this elab to the specified checkpoint file.

        Args:
            ckpt_name (str): The name of the checkpoint file.

        Returns:
            None
        '''
        
        self.folder_path.mkdir(parents=True, exist_ok=True)
        
        obj = {
            'model': self.model.state_dict(),
        }
        if self.optimizer is not None:
            obj['optimizer'] = self.optimizer.state_dict()

        obj['states'] = self.states

        self._print("Saving to", self.folder_path/ckpt_name, "...")
        torch.save(obj, self.folder_path/ckpt_name)
        self._print("done.")
        
        
    def load(self, ckpt_name: str|Literal['latest'] = 'latest'):
        '''
        Load the model, optimizer and states from the specified checkpoint file.

        Args:
            ckpt_name (str or Literal['latest']): The name of the checkpoint file. If 'latest', the latest checkpoint file in the folder will be loaded. Defaults to 'latest'.

        Returns:
            None

        Raises:
            FileNotFoundError: If no checkpoint file is found in the folder.
            ValueError: If the loading the optimizer is required, while the optimizer state is not found in the checkpoint.
        '''

        # calculate the source path
        if ckpt_name == 'latest':
            ckpt_files = list(self.folder_path.glob("*.pth"))
            if len(ckpt_files) == 0:
                raise FileNotFoundError(f"No checkpoint file found in {self.folder_path}.")
            ckpt_files.sort()
            source_path = ckpt_files[-1]

        else:
            source_path = self.folder_path/ckpt_name

        self._print("Loading from", source_path, "...")

        load_args = {}
        if self.device is not None:
            load_args['map_location'] = self.device

        obj = torch.load(source_path, weights_only=True, **load_args)

        self.model.load_state_dict(obj['model'], strict=True)

        if self.optimizer is not None:
            if 'optimizer' not in obj:
                raise ValueError("Optimizer state not found in the checkpoint.")
            self.optimizer.load_state_dict(obj['optimizer'])

        self.states = obj.get('states', {})

        self._print("done.")

