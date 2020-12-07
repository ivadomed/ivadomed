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
    model = imed_model.Modified3DUNet(in_channel=1, out_channel=1, attention=True)
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


def test_film_generator():
    a = [[[[0 for i in range(64)] for i in range(64)]]]
    inp = torch.tensor(a).float()
    model = imed_model.FiLMgenerator(64, 1)
    inf = model(inp)
    assert (type(inf[0]) == torch.Tensor)
    assert (type(inf[1]) == torch.nn.parameter.Parameter)

def test_modified_encoder():
    # verifying changes in the Encoder class
    a = [[[[0 for i in range(322)]for i in range(322)]]]
    inp = torch.tensor(a).float()
    model = imed_model.Encoder(in_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1, n_metadata=None, film_layers=None,
                 is_2d=True, n_filters=64)
    inf = model(inp)
    assert(type(inf) == tuple)
    
def test_modified_decoder():
    # verifying if changes such as 2D attention and other modification works 
    a = [torch.zeros((1, 64, 322, 322)), torch.zeros((1, 128, 161, 161)),
         torch.zeros((1, 256, 80, 80)), torch.zeros((1, 512, 40, 40))] # by changing the depth, the length of the input feature list should be changed.
    inp = a # should not be transformed to torch tensor because "a" is the appended feature list from the skip connections.
    model = imed_model.Decoder(out_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1,
                 n_metadata=None, film_layers=None, hemis=False, final_activation="sigmoid", is_2d=True,
                 n_filters=64, attention=True)
    inf = model(inp)
    assert(type(inf) == torch.Tensor)
    
def test_unet_att2d():
    
    # verifying if the 2D attention works 
    a = [[[[0 for i in range(322)]for i in range(322)]]]
    inp = torch.tensor(a).float()
    model = imed_model.Unet(in_channel=1, out_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1, final_activation='sigmoid',
                 is_2d=True, n_filters=64)
    inf = model(inp)
    assert(type(inf) == torch.Tensor)



