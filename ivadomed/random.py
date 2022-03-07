import torch
import random
import numpy as np

def set_seed(seed=6, rank=0):
    """
    Args:
        seed: int to initialize random number generator (RNG)
        rank: int specifying the GPU id for training
    Returns:
        seeded RNG in the specified libraries/packages.
    """
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
