"""
Fix all random seeds (Python, NumPy, PyTorch, and PyTorch's CUDA generator
if present) so that repeated baseline runs are comparable. Without this,
you cannot tell whether a change in Silhouette/DBI between two runs came
from your algorithm or from random initialization.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic where possible (may slow training slightly).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
