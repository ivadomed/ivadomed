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


class Unet(Module):
    """A reference U-Net model.

    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_channel=1, drop_rate=0.4, bn_momentum=0.1):
        super(Unet, self).__init__()

        #Downsampling path
        self.conv1 = DownConv(in_channel, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottom
        self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)

        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
        self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.mp1(x1)

        x3 = self.conv2(x2)
        x4 = self.mp2(x3)

        x5 = self.conv3(x4)
        x6 = self.mp3(x5)

        # Bottom
        x7 = self.conv4(x6)

        # Up-sampling
        x8 = self.up1(x7, x5)
        x9 = self.up2(x8, x3)
        x10 = self.up3(x9, x1)

        x11 = self.conv9(x10)
        preds = torch.sigmoid(x11)

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

        self.gammas = film_params[:, : self.feature_size, :, :]
        self.betas = film_params[:, self.feature_size :, :, :]

        # Apply the linear modulation
        output = self.gammas * feature_maps + self.betas

        return output, new_w_shared


class FiLMedUnet(Module):
    """A U-Net model, modulated using FiLM.

    A FiLM layer has been added after each convolution layer.
    """

    def __init__(self, n_metadata, film_bool=[1]*8, drop_rate=0.4, bn_momentum=0.1):
        super(FiLMedUnet, self).__init__()

        #Downsampling path
        self.conv1 = DownConv(1, 64, drop_rate, bn_momentum)
        self.film0 = FiLMlayer(n_metadata, 64) if film_bool[0] else None
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.film1 = FiLMlayer(n_metadata, 128) if film_bool[1] else None
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.film2 = FiLMlayer(n_metadata, 256) if film_bool[2] else None
        self.mp3 = nn.MaxPool2d(2)

        # Bottom
        self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)
        self.film3 = FiLMlayer(n_metadata, 256) if film_bool[3] else None

        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
        self.film4 = FiLMlayer(n_metadata, 256) if film_bool[4] else None
        self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
        self.film5 = FiLMlayer(n_metadata, 128) if film_bool[5] else None
        self.up3 = UpConv(192, 64, drop_rate, bn_momentum)
        self.film6 = FiLMlayer(n_metadata, 64) if film_bool[6] else None

        self.conv9 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.film7 = FiLMlayer(n_metadata, 1) if film_bool[7] else None

    def forward(self, x, context):
        x1 = self.conv1(x)
        if self.film0:
            x1, w_film = self.film0(x1, context, None)
        x2 = self.mp1(x1)

        x3 = self.conv2(x2)
        if self.film1:
            x3, w_film = self.film1(x3, context, None if 'w_film' not in locals() else w_film)
        x4 = self.mp2(x3)

        x5 = self.conv3(x4)
        if self.film2:
            x5, w_film = self.film2(x5, context, None if 'w_film' not in locals() else w_film)
        x6 = self.mp3(x5)

        # Bottom
        x7 = self.conv4(x6)
        if self.film3:
            x7, w_film = self.film3(x7, context, None if 'w_film' not in locals() else w_film)

        # Up-sampling
        x8 = self.up1(x7, x5)
        if self.film4:
            x8, w_film = self.film4(x8, context, None if 'w_film' not in locals() else w_film)
        x9 = self.up2(x8, x3)
        if self.film5:
            x9, w_film = self.film5(x9, context, None if 'w_film' not in locals() else w_film)
        x10 = self.up3(x9, x1)
        if self.film6:
            x10, w_film = self.film6(x10, context, None if 'w_film' not in locals() else w_film)

        x11 = self.conv9(x10)
        if self.film7:
            x11, w_film = self.film7(x11, context, None if 'w_film' not in locals() else w_film)
        preds = torch.sigmoid(x11)

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