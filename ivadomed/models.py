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

    def __init__(self, in_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1):
        super(Encoder, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        # first block
        self.down_path.append(DownConv(in_channel, 64, drop_rate, bn_momentum))
        self.down_path.append(nn.MaxPool2d(2))

        # other blocks
        in_channel = 64

        for i in range(depth - 1):
            self.down_path.append(DownConv(in_channel, in_channel * 2, drop_rate, bn_momentum))
            self.down_path.append(nn.MaxPool2d(2))
            in_channel = in_channel * 2

        # Bottom
        self.conv_bottom = DownConv(in_channel, in_channel, drop_rate, bn_momentum)

    def forward(self, x):
        features = []
        # Down-sampling path
        for i in range(self.depth):
            x = self.down_path[i * 2](x)
            features.append(x)
            x = self.down_path[i * 2 + 1](x)

        # Bottom level
        features.append(self.conv_bottom(x))

        return features


class Decoder(Module):
    """Encoding part of the U-Net model.
            It returns the features map for the skip connections

            see also::
            Ronneberger, O., et al (2015). U-Net: Convolutional
            Networks for Biomedical Image Segmentation
            ArXiv link: https://arxiv.org/abs/1505.04597

                """

    def __init__(self, out_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1, hemis=False):
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

        for i in range(1, depth):
            in_channel //= 2
            self.up_path.append(
                UpConv(in_channel + 64 * 2 ** (self.depth - i - 1), 64 * 2 ** (self.depth - i - 1), drop_rate,
                       bn_momentum))

        # Last Convolution
        self.last_conv = nn.Conv2d(in_channel // 2, out_channel, kernel_size=3, padding=1)

    def forward(self, features):
        x = features[-1]
        for i in reversed(range(self.depth)):
            x = self.up_path[-i - 1](x, features[i])

        # Last convolution
        x = self.last_conv(x)
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

    def __init__(self, in_channel=1, out_channel=1, depth=3, drop_rate=0.4, bn_momentum=0.1):
        super(Unet, self).__init__()

        # Encoder path
        self.encoder = Encoder(in_channel, depth, drop_rate, bn_momentum)

        # Decoder path
        self.decoder = Decoder(out_channel, depth, drop_rate, bn_momentum)

    def forward(self, x):
        # Encoding part
        features = self.encoder(x)

        preds = self.decoder(features)

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


class FiLMEncoder(Module):
    def __init__(self, n_metadata, film_bool, depth, drop_rate=0.4, bn_momentum=0.1):
        super(FiLMEncoder, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()

        self.down_path.append(DownConv(1, 64, drop_rate, bn_momentum))
        self.down_path.append(FiLMlayer(n_metadata, 64) if film_bool[0] else None)
        self.down_path.append(nn.MaxPool2d(2))

        in_channel = 64
        # Encoder
        for i in range(depth - 1):
            self.down_path.append(DownConv(in_channel, in_channel * 2, drop_rate, bn_momentum))
            self.down_path.append(FiLMlayer(n_metadata, in_channel * 2) if film_bool[i + 1] else None)
            self.down_path.append(nn.MaxPool2d(2))
            in_channel = in_channel * 2

        # Bottom
        self.conv_bottom = DownConv(in_channel, in_channel, drop_rate, bn_momentum)
        self.film_bottom = FiLMlayer(n_metadata, in_channel) if film_bool[self.depth] else None

    def forward(self, x, context):
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

class FiLMDecoder(Module):
    def __init__(self, n_metadata, film_bool, depth=3, drop_rate=0.4, bn_momentum=0.1):
        super(FiLMDecoder, self).__init__()

        # Decoder
        self.depth = depth

        # Up-Sampling path
        self.up_path = nn.ModuleList()
        in_channel = 64 * 2 ** self.depth

        self.up_path.append(UpConv(in_channel, 64 * 2 ** (self.depth - 1), drop_rate, bn_momentum))
        self.up_path.append(FiLMlayer(n_metadata, 64 * 2 ** (self.depth - 1)) if film_bool[self.depth + 1] else None)

        for i in range(1, depth):
            in_channel //= 2
            self.up_path.append(
                UpConv(in_channel + 64 * 2 ** (self.depth - i - 1), 64 * 2 ** (self.depth - i - 1), drop_rate,
                       bn_momentum))
            self.up_path.append(FiLMlayer(n_metadata, 64 * 2 ** (self.depth - i - 1)) if film_bool[self.depth + i + 1]
                                else None)

        # Last Convolution
        self.last_conv = nn.Conv2d(in_channel // 2, 1, kernel_size=3, padding=1)
        self.last_film = FiLMlayer(n_metadata, 1) if film_bool[-1] else None

    def forward(self, features, context, w_film):
        x = features[-1]

        for i in reversed(range(self.depth)):
            x = self.up_path[-(i + 1) * 2](x, features[i])
            if self.up_path[-(i + 1) * 2 + 1]:
                x, w_film = self.up_path[-(i + 1) * 2 + 1](x, context, w_film)

        x = self.last_conv(x)
        if self.last_film:
            x, w_film = self.last_film(x, context, w_film)
        # Last convolution
        preds = torch.sigmoid(x)
        return preds


class FiLMedUnet(Module):
    """A U-Net model, modulated using FiLM.

    A FiLM layer has been added after each convolution layer.
    """

    def __init__(self, n_metadata, film_bool, depth=None, in_channel=1, out_channel=1, drop_rate=0.4, bn_momentum=0.1):
        super(FiLMedUnet, self).__init__()
        
        if len(film_bool) != 2 * depth + 2:
            raise ValueError(f"The number of FiLM layers {len(film_bool)} entered does not correspond to the depth UNet"
                             f"depth. There should 2 * depth + 2 layers.")

        # Encoder path
        self.encoder = FiLMEncoder(n_metadata, film_bool, depth, drop_rate, bn_momentum)

        # Decoder path
        self.decoder = FiLMDecoder(n_metadata, film_bool, depth, drop_rate, bn_momentum)
        # Downsampling path

    def forward(self, x, context):
        features, w_film = self.encoder(x, context)

        preds = self.decoder(features, context, w_film)

        return preds


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
