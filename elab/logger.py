import os
from torch.utils.tensorboard.writer import SummaryWriter

from pathlib import Path

class Logger:
    def __init__(self, log_dir : str | Path):
        self._log_dir = log_dir
        print('logging outputs to ', log_dir)
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, name, scalar, step):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step)

    def log_scalars(self, group_name, scalar_dict, step):
        """log a group of scalars"""
        self._summ_writer.add_scalars('{}'.format(group_name), scalar_dict, step)

    def log_text(self, name, text, step):
        self._summ_writer.add_text('{}'.format(name), text, step)

    def flush(self):
        self._summ_writer.flush()




