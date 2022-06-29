import os
import shutil
import numpy as np
import torch
from pathlib import Path

def save_checkpoint(state, save_path: str, is_best: bool = False, best_ckpt_path:str=None):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        best_ckpt_path(str): path to best checkpoint
    """

    save_dir = os.path.dirname(save_path)
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    # save checkpoint
    torch.save(state, save_path)
    # copy latest
    shutil.copyfile(save_path, os.path.join(save_dir, 'latest_model.ckpt'))
    if is_best:
        
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def load_checkpoint(ckpt_dir_or_file: str, cp_path, map_location=None, load_best=True):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file,cp_path, 'checkpoints', 'best_model.ckpt')
        else:
            ckpt_path = os.path.join(ckpt_dir_or_file, cp_path, 'checkpoints', 'latest_model.ckpt')

    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def ensure_dir(dir_name: str):
    """Creates folder if not exists.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

