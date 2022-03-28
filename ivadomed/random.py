import os
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
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
