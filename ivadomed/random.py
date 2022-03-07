import torch
import random
import numpy as np

def set_seed(seed=6, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
