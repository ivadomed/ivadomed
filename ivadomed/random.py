import torch
import random
import numpy as np

def set_seed(seed: int = 6):
    """
    This function controls sources of randomness to aid reproducibility.
    Args:
        seed: int to initialize random number generator (RNG)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
