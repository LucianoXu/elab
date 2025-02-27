
from datetime import datetime
import os
from typing import Literal, Type, Tuple, Optional, Any

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from pathlib import Path
from .utils import get_parameter_size

CKPT_NAME = "ckpt.pth"

def is_elab_folder(folder_path: str|Path) -> bool:
    '''
    Check whether the folder is an ELab folder.

    Args:
        folder_path (str or Path): The folder path to check.

    Returns:
        bool: Whether the folder is an ELab folder.
    '''
    folder_path = Path(folder_path)

    if not folder_path.exists() or not folder_path.is_dir():
        return False
    
    if not (folder_path / CKPT_NAME).exists():
        return False
    
    return True

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
                 version_name: str|Literal['latest', 'none'],
                 data: dict[str, Any],
                 device: str|torch.device,
                 verbose: bool = True):
        '''
        Initialize the ELab object by specifying the folder path, the checkpoint to load, as well as the model and optimizer instances.
        The device is automatically set to the device of the model.

        Args:
            folder_path (str or Path): The folder path where the checkpoint files are stored.

            version_name (str or Literal['latest', 'none']): The name for the checkpoint subfolder. ELab will try to load the version if it already exists. The model and optimizer will be loaded from the specified version folder. If `ckpt_name` is 'latest', the latest checkpoint file in the folder will be loaded. If `ckpt_name` is 'none', no checkpoint file will be loaded.

            model (nn.Module): The PyTorch model instance.

            optimizer (Optimizer, optional): The PyTorch optimizer instance. If `None`, only the model is loaded. Defaults to `None`.

            default_states (dict, optional): The default states of the ELab object. Defaults to `None`.

            verbose (bool): Whether to print messages. Defaults to `True`.
        '''
        
        self.verbose = verbose
        
        self._print("ELab initializing at", folder_path)
        self.folder_path: Path = Path(folder_path)

        self.data = data
        self.device = device

        self.load(version_name)
    
    def save(self, version_name: Optional[str] = None) -> str:
        '''
        Save the model, optimizer and states of this elab to the specified checkpoint file.

        Args:
            version_name (str, optional): The name of the checkpoint subfolder.
            If None, will use the time as the version.

        Returns:
            str: the name for the saved version
        '''

        if version_name is None:
            # get the current time in the YYMMDDHHMMSS format
            version_name = datetime.now().strftime('%y%m%d%H%M%S')

        version_folder = self.folder_path / version_name
        
        version_folder.mkdir(parents=True, exist_ok=True)

        ckpt_name = version_folder / CKPT_NAME

        self._print("Saving to", ckpt_name, "...")

        obj = {}
        for key in self.data:
            if isinstance(self.data[key], nn.Module):
                obj[key] = self.data[key].state_dict()
            elif isinstance(self.data[key], Optimizer):
                obj[key] = self.data[key].state_dict()
            else:
                obj[key] = self.data[key]

        torch.save(obj, ckpt_name)

        self._print("done.")

        return version_name
        
        
    def load(self, version_name: str|Literal['latest', 'none']) -> str|Literal['none']:
        '''
        Load from the version folder. Only keys in self.data will be loaded.

        Returns:
            str|Literal['none']: the name for the loaded version.

        Raises:
            FileNotFoundError: If no checkpoint file is found in the folder.
            ValueError: If the loading the optimizer is required, while the optimizer state is not found in the checkpoint.
        '''

        if version_name == 'none':
            self._print("No version name specified. Skipping loading.")


        # calculate the source path
        if version_name == 'latest':
            version_folders = list(folder for folder in self.folder_path.glob("*") if is_elab_folder(folder))

            if len(version_folders) == 0:
                raise FileNotFoundError(f"Version name was set to 'latest', but no version folder found in {self.folder_path}.")
            version_folders.sort(key=lambda x: os.path.getctime(x))
            version_folder = version_folders[-1]
            version_name = version_folder.name

        version_folder = self.folder_path/version_name

        # check whether the version folder exists
        if not version_folder.exists():
            self._print(f"Version folder {version_folder} not found.")
            return 'none'
        
        else:
            source_path = version_folder / CKPT_NAME
            
            self._print("Loading from", source_path, "...")



            obj = torch.load(source_path, weights_only=True, map_location=self.device)

            for key in self.data:
                if isinstance(self.data[key], nn.Module):
                    self.data[key].load_state_dict(obj[key], strict=True)
                elif isinstance(self.data[key], Optimizer):
                    self.data[key].load_state_dict(obj[key])
                else:
                    self.data[key] = obj[key]


            self._print("done.")
            return version_name

