import ivadomed.models as imed_model
import torch
import torchvision

# testing countception model
def test_countception():
    a = [[[[0 for i in range(10)] for i in range(10)]]]
    inp = torch.tensor(a).float()
    model = imed_model.Countception(in_channel=1, out_channel=1)
    inf = model(inp)
    assert (type(inf) == torch.Tensor)


def test_model_3d_att():
    # verifying if 3d attention model can be created
    a = [[[[[0 for i in range(48)] for j in range(48)] for k in range(16)]]]
    inp = torch.tensor(a).float()
    model = imed_model.UNet3D(in_channel=1, out_channel=1, attention=True)
    inf = model(inp)
    assert(type(inf) == torch.Tensor)


def test_resnet():
    a = [[[[0 for i in range(100)] for i in range(100)]]]
    inp = torch.tensor(a).float()
    model = imed_model.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    inf = model(inp)
    assert (type(inf) == torch.Tensor)


def test_densenet():
    a = [[[[0 for i in range(100)] for i in range(100)]]]
    inp = torch.tensor(a).float()
    model = imed_model.DenseNet(32, (6, 12, 24, 16), 64)
    inf = model(inp)
    assert (type(inf) == torch.Tensor)


def test_filmed_unet():
    a = [[[[0 for i in range(100)] for i in range(100)]]]
    inp = torch.tensor(a).float()
    model = imed_model.FiLMedUnet()
    inf = model(inp)
    assert (type(inf) == torch.Tensor)



