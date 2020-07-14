import ivadomed.models as imed_model
import torch


# testing countception model
def test_countception():
    a = [[[0 for i in range(10)] for i in range(10)]]
    inp = torch.tensor(a)
    model = imed_model.Countception(in_channel=1,out_channel=1)
    model(inp)
