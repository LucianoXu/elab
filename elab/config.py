
from pathlib import Path
import json

class ELabConfig:

    def default_config(self):
        return {
                'elab_name'        : '<elab_name>',
                'elab_version'     : '<elab_version>',
                'elab_path'        : '.',
                'checkpoint'        : 'none',
                'corpus_path'       : "~/.cache/huggingface/datasets",

                'model_args'        : {},
                'optim_args'        : {},
            }
    
    @staticmethod
    def from_path(path: str | Path):
        result = ELabConfig("", "")
        if not isinstance(path, Path):
            path = Path(path)
        
        if not path.name.endswith("elab-config.json"):
            path = path / "elab-config.json"
        
        with open(path, 'r') as file:
            result.data = json.load(file)

        return result
    
    def __init__(self, 
                 elab_name: str,
                 elab_version: str,
                 elab_path: str = '.'):
        self.data = self.default_config()
        self.data['elab_name'] = elab_name
        self.data['elab_version'] = elab_version
        self.data['elab_path'] = elab_path

    
    def __repr__(self) -> str:
        return f"ELabConfig({self.data})"
    
    def __getitem__(self, key):
        return self.data[key]

    @property
    def proj_path(self) -> Path:
        return Path(self.data['elab_path']) / (self.data['elab_name'] + "-" + self.data['elab_version'])
    
    @property
    def checkpoint_path(self) -> Path:
        return self.proj_path / 'checkpoints'

    @property
    def logger_path(self) -> Path:
        return self.proj_path / 'runs'
    

    def get_checkpoint_file_path(self) -> str | None:
        '''
        Find the latest weights file in the weights folder
        The whole path is returned
        '''

        if self.data['checkpoint'] == 'none':
            return None

        elif self.data['checkpoint'] == 'latest':
            model_filename = f"{self.data['elab_name']}-{self.data['elab_version']}-*"
            weights_files = list(self.checkpoint_path.glob(model_filename))
            if len(weights_files) == 0:
                return None
            weights_files.sort()
            return str(weights_files[-1])
        else:
            model_filename = f"{self.data['elab_name']}-{self.data['elab_version']}-{self.data['checkpoint']}.pt"
            return str(self.checkpoint_path / model_filename)

    def save(self):
        self.proj_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logger_path.mkdir(parents=True, exist_ok=True)
        
        json_path = self.proj_path / 'elab-config.json'
        with open(json_path, 'w') as file:
            json.dump(self.data, file, indent=4)