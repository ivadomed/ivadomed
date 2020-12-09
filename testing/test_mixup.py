import ivadomed.mixup as imed_mixup
import torch
import pytest
import os


@pytest.mark.parametrize("debugging", [False, True])
@pytest.mark.parametrize("ofolder", ["test", "mixup_test"])
def test_mixup(debugging, ofolder):
    inp = [[[[0 for i in range(40)] for i in range(40)]]]
    targ = [[[[0 for i in range(40)] for i in range(40)]]]
    for i in range(10):
        for j in range(10):
            targ[0][0][i][j] = 1
    inp = torch.tensor(inp).float()
    targ = torch.tensor(targ).float()
    # just testing if mixup function run
    out = imed_mixup.mixup(inp, targ, alpha=0.5, debugging=debugging, ofolder=ofolder)
    os.remove(ofolder)

