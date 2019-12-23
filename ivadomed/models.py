import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F


class DownConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv2_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class Encoder(Module):
    """Encoding part of the U-Net model.
            It returns the features map for the skip connections

            see also::
            Ronneberger, O., et al (2015). U-Net: Convolutional
            Networks for Biomedical Image Segmentation
            ArXiv link: https://arxiv.org/abs/1505.04597

                """

    def __init__(self, in_channel=1, depth=3, n_metadata=None, film_layers=None, drop_rate=0.4, bn_momentum=0.1):
        super(Encoder, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        # first block
        self.down_path.append(DownConv(in_channel, 64, drop_rate, bn_momentum))
        self.down_path.append(FiLMlayer(n_metadata, 64) if film_layers[0] else None)
        self.down_path.append(nn.MaxPool2d(2))

        # other blocks
        in_channel = 64

        for i in range(depth - 1):
            self.down_path.append(DownConv(in_channel, in_channel * 2, drop_rate, bn_momentum))
            self.down_path.append(FiLMlayer(n_metadata, in_channel * 2) if film_layers[i + 1] else None)
            self.down_path.append(nn.MaxPool2d(2))
            in_channel = in_channel * 2

        # Bottom
        self.conv_bottom = DownConv(in_channel, in_channel, drop_rate, bn_momentum)
        self.film_bottom = FiLMlayer(n_metadata, in_channel) if film_layers[self.depth] else None

    def forward(self, x, context=None):
        features = []

        # First block
        x = self.down_path[0](x)
        if self.down_path[1]:
            x, w_film = self.down_path[1](x, context, None)
        features.append(x)
        x = self.down_path[2](x)

        # Down-sampling path (other blocks)
        for i in range(1, self.depth):
            x = self.down_path[i * 3](x)
            if self.down_path[i * 3 + 1]:
                x, w_film = self.down_path[i * 3 + 1](x, context, None if 'w_film' not in locals() else w_film)
            features.append(x)
            x = self.down_path[i * 3 + 2](x)

        # Bottom level
        x = self.conv_bottom(x)
        if self.film_bottom:
            x, w_film = self.film_bottom(x, context, None if 'w_film' not in locals() else w_film)
        features.append(x)

        return features, None if 'w_film' not in locals() else w_film


class Decoder(Module):
    """Encoding part of the U-Net model.
            It returns the features map for the skip connections

            see also::
            Ronneberger, O., et al (2015). U-Net: Convolutional
            Networks for Biomedical Image Segmentation
            ArXiv link: https://arxiv.org/abs/1505.04597

                """

    def __init__(self, out_channel=1, depth=3, n_metadata=None, film_layers=None, drop_rate=0.4, bn_momentum=0.1,
                 hemis=False):
        super(Decoder, self).__init__()
        self.depth = depth
        self.out_channel = out_channel
        # Up-Sampling path
        self.up_path = nn.ModuleList()
        if hemis:
            in_channel = 64 * 2 ** (self.depth + 1)
            self.depth += 1
        else:
            in_channel = 64 * 2 ** self.depth

        self.up_path.append(UpConv(in_channel, 64 * 2 ** (self.depth - 1), drop_rate, bn_momentum))
        self.up_path.append(FiLMlayer(n_metadata, 64 * 2 ** (self.depth - 1)) if film_layers[self.depth + 1] else None)

        for i in range(1, depth):
            in_channel //= 2
            self.up_path.append(
                UpConv(in_channel + 64 * 2 ** (self.depth - i - 1), 64 * 2 ** (self.depth - i - 1), drop_rate,
                       bn_momentum))
            self.up_path.append(FiLMlayer(n_metadata, 64 * 2 ** (self.depth - i - 1)) if film_layers[self.depth + i + 1]
                                else None)

        # Last Convolution
        self.last_conv = nn.Conv2d(in_channel // 2, out_channel, kernel_size=3, padding=1)
        self.last_film = FiLMlayer(n_metadata, 1) if film_layers[-1] else None

    def forward(self, features, context=None, w_film=None):
        x = features[-1]
        for i in reversed(range(self.depth)):
            x = self.up_path[-(i + 1) * 2](x, features[i])
            if self.up_path[-(i + 1) * 2 + 1]:
                x, w_film = self.up_path[-(i + 1) * 2 + 1](x, context, w_film)

        # Last convolution
        x = self.last_conv(x)
        if self.last_film:
            x, w_film = self.last_film(x, context, w_film)
        if self.out_channel > 1:
            preds = F.softmax(x, dim=1)
        else:
            preds = torch.sigmoid(x)
        return preds


class Unet(Module):
    """A reference U-Net model.

    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, n_metadata=None, film_layers=None, drop_rate=0.4,
                 bn_momentum=0.1, film_bool=False):
        super(Unet, self).__init__()

        # Verify if the length of boolean FiLM layers corresponds to the depth
        if film_bool and len(film_layers) != 2 * depth + 2:
            raise ValueError(f"The number of FiLM layers {len(film_layers)} entered does not correspond to the "
                             f"UNet depth. There should 2 * depth + 2 layers.")

        # If no metadata type is entered all layers should be to 0
        if not film_bool:
            film_layers = [0] * (2 * depth + 2)

        # Encoder path
        self.encoder = Encoder(in_channel, depth, n_metadata, film_layers, drop_rate, bn_momentum)

        # Decoder path
        self.decoder = Decoder(out_channel, depth, n_metadata, film_layers, drop_rate, bn_momentum)

    def forward(self, x, context=None):
        features, w_film = self.encoder(x, context)

        preds = self.decoder(features, context, w_film)

        return preds


class FiLMgenerator(Module):
    """The FiLM generator processes the conditioning information
    and produces parameters that describe how the target network should alter its computation.

    Here, the FiLM generator is a multi-layer perceptron.
    """

    def __init__(self, n_features, n_channels, n_hid=64):
        super(FiLMgenerator, self).__init__()
        self.linear1 = nn.Linear(n_features, n_hid)
        self.sig1 = nn.Sigmoid()
        self.linear2 = nn.Linear(n_hid, n_hid // 4)
        self.sig2 = nn.Sigmoid()
        self.linear3 = nn.Linear(n_hid // 4, n_channels * 2)
        self.sig3 = nn.Sigmoid()

    def forward(self, x, shared_weights=None):
        x = self.linear1(x)
        x = self.sig1(x)

        if shared_weights is not None:  # weight sharing
            self.linear2.weight = shared_weights

        x = self.linear2(x)
        x = self.sig2(x)
        x = self.linear3(x)

        out = self.sig3(x)
        return out, self.linear2.weight


class FiLMlayer(Module):
    """Applies Feature-wise Linear Modulation to the incoming data as described
    in the paper `FiLM: Visual Reasoning with a General Conditioning Layer`:
        https://arxiv.org/abs/1709.07871
    """

    def __init__(self, n_metadata, n_channels):
        super(FiLMlayer, self).__init__()

        self.batch_size = None
        self.height = None
        self.width = None
        self.feature_size = None
        self.generator = FiLMgenerator(n_metadata, n_channels)
        # Add the parameters gammas and betas to access them out of the class.
        self.gammas = None
        self.betas = None

    def forward(self, feature_maps, context, w_shared):
        _, self.feature_size, self.height, self.width = feature_maps.data.shape

        if torch.cuda.is_available():
            context = torch.Tensor(context).cuda()
        else:
            context = torch.Tensor(context)

        # Estimate the FiLM parameters using a FiLM generator from the contioning metadata
        film_params, new_w_shared = self.generator(context, w_shared)

        # FiLM applies a different affine transformation to each channel,
        # consistent accross spatial locations
        film_params = film_params.unsqueeze(-1).unsqueeze(-1)
        film_params = film_params.repeat(1, 1, self.height, self.width)

        self.gammas = film_params[:, :self.feature_size, :, :]
        self.betas = film_params[:, self.feature_size:, :, :]

        # Apply the linear modulation
        output = self.gammas * feature_maps + self.betas

        return output, new_w_shared


class HeMISUnet(Module):
    """A U-Net model inspired by HeMIS to deal with missing modalities.
        1) It has as many encoders as modalities but only one decoder.
        2) Skip connections are the concatenations of the means and var of all encoders skip connections

        Param:
        Modalities: list of all the possible modalities. ['T1', 'T2', 'T2S', 'F']

        see also::
        Havaei, M., Guizard, N., Chapados, N., Bengio, Y.:
        Hemis: Hetero-modal image segmentation.
        ArXiv link: https://arxiv.org/abs/1607.05194
        ---
        Reuben Dorent and Samuel Joutard and Marc Modat and Sébastien Ourselin and Tom Vercauteren
        Hetero-Modal Variational Encoder-Decoder for Joint Modality Completion and Segmentation
        ArXiv link: https://arxiv.org/abs/1907.11150
        """

    def __init__(self, modalities, depth=3, drop_rate=0.4, bn_momentum=0.1):
        super(HeMISUnet, self).__init__()

        self.depth = depth
        self.modalities = modalities

        # Encoder path
        self.Down = nn.ModuleDict(
            [['Down_{}'.format(Mod), Encoder(1, depth, drop_rate, bn_momentum)] for Mod in self.modalities])

        # Decoder path
        self.decoder = Decoder(1, depth, drop_rate, bn_momentum, hemis=True)

    def forward(self, x_mods):
        """"
            X is  list like X = [x_T1, x_T2, x_T2S, x_F]
        """

        features_mod = [[] for _ in range(self.depth + 1)]

        # Down-sampling
        for i, Mod in enumerate(self.modalities):
            features = self.Down['Down_{}'.format(Mod)](x_mods[i])
            for j in range(self.depth + 1):
                features_mod[j].append(features[j].unsqueeze(0))

        # Abstraction
        for j in range(self.depth + 1):
            features_mod[j] = torch.cat([torch.cat(features_mod[j], 0).mean(0).unsqueeze(0), \
                                         torch.cat(features_mod[j], 0).var(0).unsqueeze(0)], 0)

        # Up-sampling
        preds = self.decoder(features_mod)

        return preds

class UNet3D(nn.Module):
    """A reference of 3D U-Net model.
    Implementation origin :
    https://github.com/shiba24/3d-unet/blob/master/pytorch/model.py
    .. seealso::
        Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox
        and Olaf Ronneberger (2016). 3D U-Net: Learning Dense Volumetric
        Segmentation from Sparse Annotation
        ArXiv link: https://arxiv.org/pdf/1606.06650.pdf
    """
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.down_conv(self.in_channel, 32, bias=False, batchnorm=False)
        self.ec1 = self.down_conv(32, 64, bias=False, batchnorm=False)
        self.ec2 = self.down_conv(64, 64, bias=False, batchnorm=False)
        self.ec3 = self.down_conv(64, 128, bias=False, batchnorm=False)
        self.ec4 = self.down_conv(128, 128, bias=False, batchnorm=False)
        self.ec5 = self.down_conv(128, 256, bias=False, batchnorm=False)
        self.ec6 = self.down_conv(256, 256, bias=False, batchnorm=False)
        self.ec7 = self.down_conv(256, 512, bias=False, batchnorm=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.up_conv(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.down_conv(256 + 512, 256, bias=False)
        self.dc7 = self.down_conv(256, 256, bias=False)
        self.dc6 = self.up_conv(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.down_conv(128 + 256, 128, bias=False)
        self.dc4 = self.down_conv(128, 128, bias=False)
        self.dc3 = self.up_conv(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.down_conv(64 + 128, 64, bias=False)
        self.dc1 = self.down_conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.down_conv(64, n_classes, kernel_size=1, stride=1, padding=0, bias=False)


    def down_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU())
        return layer


    def up_conv(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.LeakyReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0