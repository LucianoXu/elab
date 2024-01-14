# ELab

```
pip install -e .
```

## Documentation

### elab.ELabConfig
Constructor:
```python
ELabConfig.from_path(path: str | Path)
```
```python
ELabConfig(elab_name: str, elab_version: str, elab_path: str = '.')
```

Methods:
```python
ELabConfig.save(self)
```
Properties:
- `proj_path`
- `checkpoint_path`
- `logger_path`


### elab.Logger
Constructor:
```python
Logger(log_dir: str | Path)
```

Methods:
```python
Logger.log_scalar(self, name, scalar, step)
```
```python
Logger.log_scalars(self, group_name, scalar_dict, step)
```
```python
Logger.log_text(self, name, text, step)
```
```python
Logger.flush(self)
```

### elab.ELab
Constructor:
```python
ELab(config: ELabConfig, 
    model_class: Type[nn.Module] | nn.Module, 
    optim_class: Type[optim.Optimizer] | optim.Opimizer)
```
Methods:
```python
ELab.save_checkpoint(self, postfix: str | None = None)
```

Properties:
- `config`
- `device`
- `model_class`
- `optim_class`
- `model`
- `optimizer`
- `global_step`
- `proc_tokens`
- `model_size`