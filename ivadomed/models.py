import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn import init


class DownConv(Module):
    """Two successive series of down convolution, batch normalization and drop out in 2D.
    Used in U-Net's encoder.

    Args:
        in_feat (int): Number of channels in the input image.
        out_feat (int): Number of channels in the output image.
        drop_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.

    Attributes:
        conv1 (Conv2d): First 2D down convolution with kernel size 3 and padding of 1.
        conv1_bn (BatchNorm2d): First 2D batch normalization.
        conv1_drop (Dropout2d): First 2D dropout.
        conv2 (Conv2d): Second 2D down convolution with kernel size 3 and padding of 1.
        conv2_bn (BatchNorm2d): Second 2D batch normalization.
        conv2_drop (Dropout2d): Second 2D dropout.
    """

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
    """2D down convolution.
    Used in U-Net's decoder.

    Args:
        in_feat (int): Number of channels in the input image.
        out_feat (int): Number of channels in the output image.
        drop_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.

    Attributes:
        downconv (DownConv): Down convolution.
    """
    
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = F.interpolate(x, size=y.size()[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class Encoder(Module):
    """Encoding part of the U-Net model.
    It returns the features map for the skip connections

    Args:
        in_channel (int): Number of channels in the input image.
        depth (int): Number of down convolutions minus bottom down convolution.
        drop_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        down_path (ModuleList): List of module operations done during encoding.
        conv_bottom (DownConv): Bottom down convolution.
        film_bottom (FiLMlayer): FiLM layer applied to bottom convolution.
    """

    def __init__(self, in_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1, n_metadata=None, film_layers=None):
        super(Encoder, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        # first block
        self.down_path.append(DownConv(in_channel, 64, drop_rate, bn_momentum))
        self.down_path.append(FiLMlayer(n_metadata, 64) if film_layers and film_layers[0] else None)
        self.down_path.append(nn.MaxPool2d(2))

        # other blocks
        in_channel = 64

        for i in range(depth - 1):
            self.down_path.append(DownConv(in_channel, in_channel * 2, drop_rate, bn_momentum))
            self.down_path.append(FiLMlayer(n_metadata, in_channel * 2) if film_layers and film_layers[i + 1] else None)
            self.down_path.append(nn.MaxPool2d(2))
            in_channel = in_channel * 2

        # Bottom
        self.conv_bottom = DownConv(in_channel, in_channel, drop_rate, bn_momentum)
        self.film_bottom = FiLMlayer(n_metadata, in_channel) if film_layers and film_layers[self.depth] else None

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
    """Decoding part of the U-Net model.

    Args:
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        drop_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        hemis (bool): Boolean indicating if HeMIS is on or not.

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        out_channel (int): Number of channels in the output image.
        up_path (ModuleList): List of module operations done during decoding.
        last_conv (Conv2d): Last convolution.
        last_film (FiLMlayer): FiLM layer applied to last convolution.
    """

    def __init__(self, out_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1,
                 n_metadata=None, film_layers=None, hemis=False):
        super(Decoder, self).__init__()
        self.depth = depth
        self.out_channel = out_channel
        # Up-Sampling path
        self.up_path = nn.ModuleList()
        if hemis:
            in_channel = 64 * 2 ** self.depth
            self.up_path.append(UpConv(in_channel * 2, 64 * 2 ** (self.depth - 1), drop_rate, bn_momentum))
            if film_layers and film_layers[self.depth + 1]:
                self.up_path.append(FiLMlayer(n_metadata, 64 * 2 ** (self.depth - 1)))
            else:
                self.up_path.append(None)
            # self.depth += 1
        else:
            in_channel = 64 * 2 ** self.depth

            self.up_path.append(UpConv(in_channel, 64 * 2 ** (self.depth - 1), drop_rate, bn_momentum))
            if film_layers and film_layers[self.depth + 1]:
                self.up_path.append(FiLMlayer(n_metadata, 64 * 2 ** (self.depth - 1)))
            else:
                self.up_path.append(None)

        for i in range(1, depth):
            in_channel //= 2

            self.up_path.append(
                UpConv(in_channel + 64 * 2 ** (self.depth - i - 1 + int(hemis)), 64 * 2 ** (self.depth - i - 1),
                       drop_rate,
                       bn_momentum))
            if film_layers and film_layers[self.depth + i + 1]:
                self.up_path.append(FiLMlayer(n_metadata, 64 * 2 ** (self.depth - i - 1)))
            else:
                self.up_path.append(None)

        # Last Convolution
        self.last_conv = nn.Conv2d(in_channel // 2, out_channel, kernel_size=3, padding=1)
        self.last_film = FiLMlayer(n_metadata, 1) if film_layers and film_layers[-1] else None

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
            # Remove background class
            preds = preds[:, 1:, ]
        else:
            preds = torch.sigmoid(x)
        return preds


class Unet(Module):
    """A reference U-Net model.

    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597

    Args:
        in_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        drop_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        **kwargs:

    Attributes:
        encoder (Encoder): U-Net encoder.
        decoder (Decoder): U-net decoder.
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1, **kwargs):
        super(Unet, self).__init__()

        # Encoder path
        self.encoder = Encoder(in_channel=in_channel, depth=depth, drop_rate=drop_rate, bn_momentum=bn_momentum)

        # Decoder path
        self.decoder = Decoder(out_channel=out_channel, depth=depth, drop_rate=drop_rate, bn_momentum=bn_momentum)

    def forward(self, x):
        features, _ = self.encoder(x)
        preds = self.decoder(features)

        return preds


class FiLMedUnet(Unet):
    """U-Net network containing FiLM layers to condition the model with another data type (i.e. not an image).

    Args:
        n_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        drop_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        **kwargs:

    Attributes:
        encoder (Encoder): U-Net encoder.
        decoder (Decoder): U-net decoder.
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, drop_rate=0.4,
                 bn_momentum=0.1, n_metadata=None, film_layers=None, **kwargs):
        super().__init__(in_channel=1, out_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1)

        # Verify if the length of boolean FiLM layers corresponds to the depth
        if film_layers:
            if len(film_layers) != 2 * depth + 2:
                raise ValueError("The number of FiLM layers {} entered does not correspond to the "
                                 "UNet depth. There should 2 * depth + 2 layers.".format(len(film_layers)))
        else:
            film_layers = [0] * (2 * depth + 2)
        # Encoder path
        self.encoder = Encoder(in_channel=in_channel, depth=depth, drop_rate=drop_rate, bn_momentum=bn_momentum,
                               n_metadata=n_metadata, film_layers=film_layers)
        # Decoder path
        self.decoder = Decoder(out_channel=out_channel, depth=depth, drop_rate=drop_rate, bn_momentum=bn_momentum,
                               n_metadata=n_metadata, film_layers=film_layers)

    def forward(self, x, context=None):
        features, w_film = self.encoder(x, context)
        preds = self.decoder(features, context, w_film)

        return preds


class FiLMgenerator(Module):
    """The FiLM generator processes the conditioning information
    and produces parameters that describe how the target network should alter its computation.

    Here, the FiLM generator is a multi-layer perceptron.

    Args:
        n_features (int): Number of input channels.
        n_channels (int): Number of output channels.
        n_hid (int): Number of hidden units in layer.

    Attributes:
        linear1 (Linear): Input linear layer.
        sig (Sigmoid): Sigmoid function.
        linear2 (Linear): Hidden linear layer.
        linear3 (Linear): Output linear layer.
    """

    def __init__(self, n_features, n_channels, n_hid=64):
        super(FiLMgenerator, self).__init__()
        self.linear1 = nn.Linear(n_features, n_hid)
        self.sig = nn.Sigmoid()
        self.linear2 = nn.Linear(n_hid, n_hid // 4)
        self.linear3 = nn.Linear(n_hid // 4, n_channels * 2)

    def forward(self, x, shared_weights=None):
        x = self.linear1(x)
        x = self.sig(x)

        if shared_weights is not None:  # weight sharing
            self.linear2.weight = shared_weights

        x = self.linear2(x)
        x = self.sig(x)
        x = self.linear3(x)

        out = self.sig(x)
        return out, self.linear2.weight


class FiLMlayer(Module):
    """Applies Feature-wise Linear Modulation to the incoming data
    .. seealso::
        Perez, Ethan, et al. "Film: Visual reasoning with a general conditioning layer."
        Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

    Args:
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        n_channels (int): Number of output channels.

    Attributes:
        batch_size (int): Batch size.
        height (int): Image height.
        width (int): Image width.
        feature_size (int): Number of features in data.
        generator (FiLMgenerator): FiLM network.
        gammas (float): Multiplicative term of the FiLM linear modulation.
        betas (float): Additive term of the FiLM linear modulation.
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
    """A U-Net model inspired by HeMIS to deal with missing contrasts.
        1) It has as many encoders as contrasts but only one decoder.
        2) Skip connections are the concatenations of the means and var of all encoders skip connections.

        Param:
        contrasts: list of all the possible contrasts. ['T1', 'T2', 'T2S', 'F']

    .. seealso::
        Havaei, M., Guizard, N., Chapados, N., Bengio, Y.:
        Hemis: Hetero-modal image segmentation.
        ArXiv link: https://arxiv.org/abs/1607.05194
        
        Reuben Dorent and Samuel Joutard and Marc Modat and Sébastien Ourselin and Tom Vercauteren
        Hetero-Modal Variational Encoder-Decoder for Joint Modality Completion and Segmentation
        ArXiv link: https://arxiv.org/abs/1907.11150

    Args:
        contrasts (list): List of contrasts.
        out_channel (int): Number of output channels.
        depth (int): Number of down convolutions minus bottom down convolution.
        drop_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        **kwargs:

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        contrasts (list): List of contrasts.
        Encoder_mod (ModuleDict): Contains encoder for each modality.
        decoder (Decoder): U-Net decoder.
    """

    def __init__(self, contrasts, out_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1, **kwargs):
        super(HeMISUnet, self).__init__()
        self.depth = depth
        self.contrasts = contrasts

        # Encoder path
        self.Encoder_mod = nn.ModuleDict(
            [['Encoder_{}'.format(Mod), Encoder(in_channel=1, depth=depth, drop_rate=drop_rate,
                                                bn_momentum=bn_momentum)] for Mod in self.contrasts])

        # Decoder path
        self.decoder = Decoder(out_channel=out_channel, depth=depth, drop_rate=drop_rate,
                               bn_momentum=bn_momentum, hemis=True)

    def forward(self, x_mods, indexes_mod):
        """
        X is list like X = [x_T1, x_T2, x_T2S, x_F]
        indexes_mod: list of arrays like [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]]
        N.B. len(list) = number of contrasts.
        len(list[i]) = Batch size
        """
        features_mod = [[] for _ in range(self.depth + 1)]

        # Down-sampling
        for i, Mod in enumerate(self.contrasts):
            features, _ = self.Encoder_mod['Encoder_{}'.format(Mod)](x_mods[i])

            for j in range(self.depth + 1):
                features_mod[j].append(features[j].unsqueeze(0))

        # Abstraction
        for j in range(self.depth + 1):
            features_cat = torch.cat(features_mod[j], 0).transpose(0, 1)

            features_mod[j] = torch.cat([torch.cat([features_cat[i][indexes_mod[i]].squeeze(1).mean(0),
                                                    features_cat[i][indexes_mod[i]].squeeze(1).var(0)], 0).unsqueeze(0)
                                         for i in range(len(indexes_mod))], 0)

        # Up-sampling
        preds = self.decoder(features_mod)

        return preds


class UNet3D(nn.Module):
    """Code from the following repository:
    https://github.com/pykao/Modified-3D-UNet-Pytorch
    The main differences with the original UNet resides in the use of LeakyReLU instead of ReLU, InstanceNormalisation
    instead of BatchNorm due to small batch size in 3D and the addition of segmentation layers in the decoder.

    If attention=True, attention gates are added in the decoder to help focus attention on important features for a
    given task. Code related to the attentions gates is inspired from:
    https://github.com/ozan-oktay/Attention-Gated-Networks

    Args:
        in_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        n_filters (int): Number of base filters in the U-Net.
        attention (bool): Boolean indicating whether the attention module is on or not.
        drop_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        **kwargs:

    Attributes:
        in_channels (int): Number of channels in the input image.
        n_classes (int): Number of channels in the output image.
        base_n_filter (int): Number of base filters in the U-Net.
        attention (bool): Boolean indicating whether the attention module is on or not.
        momentum (float): Batch normalization momentum.

    Note: All layers are defined as attributes and used in the forward method.
    """

    def __init__(self, in_channel, out_channel, n_filters=16, attention=False, drop_rate=0.6, bn_momentum=0.1,
                 **kwargs):
        super(UNet3D, self).__init__()
        self.in_channels = in_channel
        self.n_classes = out_channel
        self.base_n_filter = n_filters
        self.attention = attention
        self.momentum = bn_momentum

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=drop_rate)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(
            self.in_channels, self.base_n_filter,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv3d_c1_2 = nn.Conv3d(
            self.base_n_filter, self.base_n_filter,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.lrelu_conv_c1 = self.lrelu_conv(
            self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter, momentum=self.momentum)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(
            self.base_n_filter, self.base_n_filter * 2,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(
            self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2, momentum=self.momentum)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(
            self.base_n_filter * 2, self.base_n_filter * 4,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(
            self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4, momentum=self.momentum)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(
            self.base_n_filter * 4, self.base_n_filter * 8,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(
            self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8, momentum=self.momentum)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(
            self.base_n_filter * 8, self.base_n_filter * 16,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(
            self.base_n_filter * 16, self.base_n_filter * 16)

        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = \
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 16, self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(
            self.base_n_filter * 8, self.base_n_filter * 8,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8, momentum=self.momentum)

        # Attention UNet
        if self.attention:
            self.gating = UnetGridGatingSignal3(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=(1, 1, 1),
                                                is_batchnorm=True)

            # attention blocks
            self.attentionblock2 = GridAttentionBlockND(in_channels=self.base_n_filter * 2,
                                                        gating_channels=self.base_n_filter * 8,
                                                        inter_channels=self.base_n_filter * 2,
                                                        sub_sample_factor=(2, 2, 2),
                                                        )
            self.attentionblock3 = GridAttentionBlockND(in_channels=self.base_n_filter * 4,
                                                        gating_channels=self.base_n_filter * 8,
                                                        inter_channels=self.base_n_filter * 4,
                                                        sub_sample_factor=(2, 2, 2),
                                                        )
            self.attentionblock4 = GridAttentionBlockND(in_channels=self.base_n_filter * 8,
                                                        gating_channels=self.base_n_filter * 8,
                                                        inter_channels=self.base_n_filter * 8,
                                                        sub_sample_factor=(2, 2, 2),
                                                        )
            self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 16, momentum=self.momentum)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(
            self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(
            self.base_n_filter * 16, self.base_n_filter * 8,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = \
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 8, self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(
            self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(
            self.base_n_filter * 8, self.base_n_filter * 4,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = \
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 4, self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(
            self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(
            self.base_n_filter * 4, self.base_n_filter * 2,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = \
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 2, self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(
            self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(
            self.base_n_filter * 2, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        self.ds2_1x1_conv3d = nn.Conv3d(
            self.base_n_filter * 8, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.ds3_1x1_conv3d = nn.Conv3d(
            self.base_n_filter * 4, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False
        )

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out, momentum=self.momentum),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in, momentum=self.momentum),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out,
                      kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out,
                      kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in, momentum=self.momentum),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out, momentum=self.momentum),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5

        if self.attention:
            out = self.inorm3d_l0(out)
            out = self.lrelu(out)

            gating = self.gating(out)
            context_4, att4 = self.attentionblock4(context_4, gating)
            context_3, att3 = self.attentionblock3(context_3, gating)
            context_2, att2 = self.attentionblock2(context_2, gating)

        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([context_1, out], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(
            ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out
        if self.n_classes > 1:
            out = self.softmax(out)
            # Remove background class
            out = out[:, 1:, ]
        else:
            out = torch.sigmoid(seg_layer)
        return out


class GridAttentionBlockND(nn.Module):
    """Attention module to focus on important features passed through U-Net's decoder
    Specific to Attention UNet

    .. seealso::
        Oktay, Ozan, et al. "Attention u-net: Learning where to look for the pancreas."
        arXiv preprint arXiv:1804.03999 (2018).

    Args:
        in_channels (int): Number of channels in the input image.
        gating_channels (int): Number of channels in the gating step.
        inter_channels (int): Number of channels in the intermediate gating step.
        dimension (int): Value of 2 or 3 to indicating whether it is used in a 2D or 3D model.
        sub_sample_factor (tuple or list): Convolution kernel size.

    Attributes:
        in_channels (int): Number of channels in the input image.
        gating_channels (int): Number of channels in the gating step.
        inter_channels (int): Number of channels in the intermediate gating step.
        dimension (int): Value of 2 or 3 to indicating whether it is used in a 2D or 3D model.
        sub_sample_factor (tuple or list): Convolution kernel size.
        upsample_mode (str): 'bilinear' or 'trilinear' related to the use of 2D or 3D models.
        W (Sequential): Sequence of convolution and batch normalization layers.
        theta (Conv2d or Conv3d): Convolution layer for gating operation.
        phi (Conv2d or Conv3d): Convolution layer for gating operation.
        psi (Conv2d or Conv3d): Convolution layer for gating operation.

    """
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3,
                 sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Initialise weights
        for m in self.children():
            m.apply(weights_init_kaiming)

        # Define the operation
        self.operation_function = self._concatenation

    def forward(self, x, g):
        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


def weights_init_kaiming(m):
    """Initialize weights according to method describe here:
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class UnetGridGatingSignal3(nn.Module):
    """Operation to extract important features for a specific task using 1x1x1 convolution (Gating) which is used in the
    attention blocks.

    Args:
        in_size (int): Number of channels in the input image.
        out_size (int): Number of channels in the output image.
        kernel_size (tuple): Convolution kernel size.
        is_batchnorm (bool): Boolean indicating whether to apply batch normalization or not.

    Attributes:
        conv1 (Sequential): 3D convolution, batch normalization and ReLU activation.
    """

    def __init__(self, in_size, out_size, kernel_size=(1, 1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, stride=1, pad=0, activation=nn.LeakyReLU()):
        """
        Perform convolution, activation and batch normalization.
        Args:
            in_chan (int): number of channels on input
            out_chan (int): number of channel on output
            ksize (int): size of kernel for the 2d convolution
            stride (int): strides for 2d convolution
            pad (int): pad for nn.conv2d
            activation (nn.layers): activation layer. default Leaky ReLu
        """
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)
        self.activation = activation
        self.batch_norm = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv1(x)))


class SimpleBlock(nn.Module):
    def __init__(self, in_chan, out_chan_1x1, out_chan_3x3, activation=nn.LeakyReLU()):
        """
        Inception module with 3 convolutions with different kernel size with separate activation.
        The 3 outputs are then concatenated. Max pooling performed on concatenation.
        Args:
            in_chan (int): number of channel of input
            out_chan_1x1 (int): number of channel after first convolution block
            out_chan_3x3 (int): number of channel for the other convolution blocks
            activation (nn.layers): activation layer used in convolution block
        """
        super(SimpleBlock, self).__init__()
        self.conv1 = ConvBlock(in_chan, out_chan_1x1, ksize=3, pad=0, activation=activation)
        self.conv2 = ConvBlock(in_chan, out_chan_3x3, ksize=5, pad=1, activation=activation)
        self.conv3 = ConvBlock(in_chan, out_chan_3x3, ksize=9, pad=3, activation=activation)
        self.MP = nn.MaxPool2d(1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        output = torch.cat([conv1_out, conv2_out, conv3_out], 1)
        output = self.MP(output)
        return output


class Countception(Module):
    """Countception model.
    Fully convolutional model using inception module and used for keypoints detection.
    The inception model extracts several patches within each image. Every pixel is therefore processed by the 
    network several times, allowing to average multiple predictions and minimize false negatives.

    .. seealso::
        Cohen JP et al. "Count-ception: Counting by fully convolutional redundant counting."
        Proceedings of the IEEE International Conference on Computer Vision Workshops. 2017.

    Args:
        in_channel (int): number of channels on input image
        out_channel (int): number of channels on output image
        use_logits (bool): boolean to change output
        logits_per_output (int): number of outputs of final convolution which will multiplied by the number of channels
        name (str): model's name used for call in configuration file.
    """

    def __init__(self, in_channel=3, out_channel=1, use_logits=False, logits_per_output=12, name='CC', **kwargs):
        super(Countception, self).__init__()

        # params
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.activation = nn.LeakyReLU(0.01)
        self.final_activation = nn.LeakyReLU(0.3)
        self.patch_size = 40
        self.use_logits = use_logits
        self.logits_per_output = logits_per_output

        torch.LongTensor()

        self.conv1 = ConvBlock(self.in_channel, 64, ksize=3, pad=self.patch_size, activation=self.activation)
        self.simple1 = SimpleBlock(64, 16, 16, activation=self.activation)
        self.simple2 = SimpleBlock(48, 16, 32, activation=self.activation)
        self.conv2 = ConvBlock(80, 16, ksize=14, activation=self.activation)
        self.simple3 = SimpleBlock(16, 112, 48, activation=self.activation)
        self.simple4 = SimpleBlock(208, 64, 32, activation=self.activation)
        self.simple5 = SimpleBlock(128, 40, 40, activation=self.activation)
        self.simple6 = SimpleBlock(120, 32, 96, activation=self.activation)
        self.conv3 = ConvBlock(224, 32, ksize=20, activation=self.activation)
        self.conv4 = ConvBlock(32, 64, ksize=10, activation=self.activation)
        self.conv5 = ConvBlock(64, 32, ksize=9, activation=self.activation)

        if use_logits:
            self.conv6 = nn.ModuleList([ConvBlock(
                64, logits_per_output, ksize=1, activation=self.final_activation) for _ in range(out_channel)])
        else:
            self.conv6 = ConvBlock(32, self.out_channel, ksize=20, pad=1, activation=self.final_activation)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('leaky_relu', param=0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        net = self.conv1(x)  # 32
        net = self.simple1(net)
        net = self.simple2(net)
        net = self.conv2(net)
        net = self.simple3(net)
        net = self.simple4(net)
        net = self.simple5(net)
        net = self.simple6(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.conv5(net)

        if self.use_logits:
            net = [c(net) for c in self.conv6]
            [self._print(n) for n in net]
        else:
            net = self.conv6(net)

        return net


def set_model_for_retrain(model_path, retrain_fraction, map_location):
    """Set model for transfer learning.

    The first layers (defined by 1-retrain_fraction) are frozen (i.e. requires_grad=False).
    The weights of the last layers (defined by retrain_fraction) are reset.

    Args:
        model_path (str): Pretrained model path.
        retrain_fraction (float): Fraction of the model that will be retrained, between 0 and 1. If set to 0.3,
            then the 30% last fraction of the model will be re-initalised and retrained.
        map_location (str): Device.

    Returns:
        torch.Module: Model ready for retrain.
    """
    # Load pretrained model
    model = torch.load(model_path, map_location=map_location)
    # Get number of layers with learnt parameters
    layer_names = [name for name, layer in model.named_modules() if hasattr(layer, 'reset_parameters')]
    n_layers = len(layer_names)
    # Compute the number of these layers we want to freeze
    n_freeze = int(round(n_layers * (1 - retrain_fraction)))
    # Last frozen layer
    last_frozen_layer = layer_names[n_freeze]

    # Set freeze first layers
    for name, layer in model.named_parameters():
        if not name.startswith(last_frozen_layer):
            layer.requires_grad = False
        else:
            break

    # Reset weights of the last layers
    for name, layer in model.named_modules():
        if name in layer_names[n_freeze:]:
            layer.reset_parameters()

    return model


def get_model_filenames(folder_model):
    """Get trained model filenames from its folder path.

    This function checks if the folder_model exists and get trained model (.pt or .onnx) and its configuration file
    (.json) from it.
    Note: if the model exists as .onnx, then this function returns its onnx path instead of the .pt version.

    Args:
        folder_name (str): Path of the model folder.

    Returns:
        str, str: Paths of the model (.onnx) and its configuration file (.json).
    """
    if os.path.isdir(folder_model):
        prefix_model = os.path.basename(folder_model)
        # Check if model and model metadata exist. Verify if ONNX model exists, if not try to find .pt model
        fname_model = os.path.join(folder_model, prefix_model + '.onnx')
        if not os.path.isfile(fname_model):
            fname_model = os.path.join(folder_model, prefix_model + '.pt')
            if not os.path.exists(fname_model):
                print('Model file not found: {}'.format(fname_model))
                exit()
        fname_model_metadata = os.path.join(folder_model, prefix_model + '.json')
        if not os.path.isfile(fname_model_metadata):
            print('Model metadata file not found: {}'.format(fname_model_metadata))
            exit()
    else:
        print('Model folder not found: {}'.format(folder_model))
        exit()
    return fname_model, fname_model_metadata
