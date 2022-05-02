import random

import numpy as np
import torch


def set_seed(seed: int = 6, deterministic: bool = False):
    """
    This function controls sources of randomness to aid reproducibility.
    Args:
        seed: int to initialize random number generator (RNG)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and deterministic:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
