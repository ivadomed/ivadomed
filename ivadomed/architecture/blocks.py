import torch
from torch import nn as nn
from torch.nn import Module, functional as F

from ivadomed.architecture.layers_common import weights_init_kaiming


"""
Blocks are typically not suppose to stand along but be a part of a larger network architecture. 
"""

class DownConvBlock(Module):
    """Two successive series of down convolution, batch normalization and dropout in 2D.
    Used in U-Net's encoder.

    Args:
        in_feat (int): Number of channels in the input image.
        out_feat (int): Number of channels in the output image.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.

    Attributes:
        conv1 (Conv2d): First 2D down convolution with kernel size 3 and padding of 1.
        conv1_bn (BatchNorm2d): First 2D batch normalization.
        conv1_drop (Dropout2d): First 2D dropout.
        conv2 (Conv2d): Second 2D down convolution with kernel size 3 and padding of 1.
        conv2_bn (BatchNorm2d): Second 2D batch normalization.
        conv2_drop (Dropout2d): Second 2D dropout.
    """

    def __init__(self, in_feat, out_feat, dropout_rate=0.3, bn_momentum=0.1, is_2d=True):
        super(DownConvBlock, self).__init__()
        if is_2d:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            dropout = nn.Dropout2d
        else:
            conv = nn.Conv3d
            bn = nn.InstanceNorm3d
            dropout = nn.Dropout3d

        self.conv1 = conv(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = bn(out_feat, momentum=bn_momentum)
        self.conv1_drop = dropout(dropout_rate)

        self.conv2 = conv(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = bn(out_feat, momentum=bn_momentum)
        self.conv2_drop = dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConvBlock(Module):
    """2D down convolution.
    Used in U-Net's decoder.

    Args:
        in_feat (int): Number of channels in the input image.
        out_feat (int): Number of channels in the output image.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.

    Attributes:
        downconv (DownConvBlock): Down convolution.
    """

    def __init__(self, in_feat, out_feat, dropout_rate=0.3, bn_momentum=0.1, is_2d=True):
        super(UpConvBlock, self).__init__()
        self.is_2d = is_2d
        self.downconv = DownConvBlock(in_feat, out_feat, dropout_rate, bn_momentum, is_2d)

    def forward(self, x, y):
        # For retrocompatibility purposes
        if not hasattr(self, "is_2d"):
            self.is_2d = True
        mode = 'bilinear' if self.is_2d else 'trilinear'
        dims = -2 if self.is_2d else -3
        x = F.interpolate(x, size=y.size()[dims:], mode=mode, align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class EncoderBlock(Module):
    """Encoding part of the U-Net model.
    It returns the features map for the skip connections

    Args:
        in_channel (int): Number of channels in the input image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        down_path (ModuleList): List of module operations done during encoding.
        conv_bottom (DownConvBlock): Bottom down convolution.
        film_bottom (FiLMlayerBlock): FiLM layer applied to bottom convolution.
    """

    def __init__(self, in_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, n_metadata=None, film_layers=None,
                 is_2d=True, n_filters=64):
        super(EncoderBlock, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        # first block
        self.down_path.append(DownConvBlock(in_channel, n_filters, dropout_rate, bn_momentum, is_2d))
        self.down_path.append(FiLMlayerBlock(n_metadata, n_filters) if film_layers and film_layers[0] else None)
        max_pool = nn.MaxPool2d if is_2d else nn.MaxPool3d
        self.down_path.append(max_pool(2))

        for i in range(depth - 1):
            self.down_path.append(DownConvBlock(n_filters, n_filters * 2, dropout_rate, bn_momentum, is_2d))
            self.down_path.append(FiLMlayerBlock(n_metadata, n_filters * 2) if film_layers and film_layers[i + 1] else None)
            self.down_path.append(max_pool(2))
            n_filters = n_filters * 2

        # Bottom
        self.conv_bottom = DownConvBlock(n_filters, n_filters, dropout_rate, bn_momentum, is_2d)
        self.film_bottom = FiLMlayerBlock(n_metadata, n_filters) if film_layers and film_layers[self.depth] else None

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


class DecoderBlock(Module):
    """Decoding part of the U-Net model.

    Args:
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        hemis (bool): Boolean indicating if HeMIS is on or not.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        out_channel (int): Number of channels in the output image.
        up_path (ModuleList): List of module operations done during decoding.
        last_conv (Conv2d): Last convolution.
        last_film (FiLMlayerBlock): FiLM layer applied to last convolution.
        softmax (Softmax): Softmax layer that can be applied as last layer.
    """

    def __init__(self, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1,
                 n_metadata=None, film_layers=None, hemis=False, final_activation="sigmoid", is_2d=True, n_filters=64):
        super(DecoderBlock, self).__init__()
        self.depth = depth
        self.out_channel = out_channel
        self.final_activation = final_activation
        # Up-Sampling path
        self.up_path = nn.ModuleList()
        if hemis:
            in_channel = n_filters * 2 ** self.depth
            self.up_path.append(
                UpConvBlock(in_channel * 2, n_filters * 2 ** (self.depth - 1), dropout_rate, bn_momentum,
                            is_2d))
            if film_layers and film_layers[self.depth + 1]:
                self.up_path.append(FiLMlayerBlock(n_metadata, n_filters * 2 ** (self.depth - 1)))
            else:
                self.up_path.append(None)
            # self.depth += 1
        else:
            in_channel = n_filters * 2 ** self.depth

            self.up_path.append(
                UpConvBlock(in_channel, n_filters * 2 ** (self.depth - 1), dropout_rate, bn_momentum, is_2d))
            if film_layers and film_layers[self.depth + 1]:
                self.up_path.append(FiLMlayerBlock(n_metadata, n_filters * 2 ** (self.depth - 1)))
            else:
                self.up_path.append(None)

        for i in range(1, depth):
            in_channel //= 2

            self.up_path.append(
                UpConvBlock(in_channel + n_filters * 2 ** (self.depth - i - 1 + int(hemis)),
                            n_filters * 2 ** (self.depth - i - 1),
                            dropout_rate, bn_momentum, is_2d))
            if film_layers and film_layers[self.depth + i + 1]:
                self.up_path.append(FiLMlayerBlock(n_metadata, n_filters * 2 ** (self.depth - i - 1)))
            else:
                self.up_path.append(None)

        # Last Convolution
        conv = nn.Conv2d if is_2d else nn.Conv3d
        self.last_conv = conv(in_channel // 2, out_channel, kernel_size=3, padding=1)
        self.last_film = FiLMlayerBlock(n_metadata, self.out_channel) if film_layers and film_layers[-1] else None
        self.softmax = nn.Softmax(dim=1)

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

        if hasattr(self, "final_activation") and self.final_activation not in ["softmax", "relu", "sigmoid"]:
            raise ValueError("final_activation value has to be either softmax, relu, or sigmoid")
        elif hasattr(self, "final_activation") and self.final_activation == "softmax":
            preds = self.softmax(x)
        elif hasattr(self, "final_activation") and self.final_activation == "relu":
            preds = nn.ReLU()(x) / nn.ReLU()(x).max()
            # If nn.ReLU()(x).max()==0, then nn.ReLU()(x) will also ==0. So, here any zero division will always be 0/0.
            # For float values, 0/0=nan. So, we can handle zero division (without checking data!) by setting nans to 0.
            preds[torch.isnan(preds)] = 0.
            # If model multiclass
            if self.out_channel > 1:
                class_sum = preds.sum(dim=1).unsqueeze(1)
                # Avoid division by zero
                class_sum[class_sum == 0] = 1
                preds /= class_sum
        else:
            preds = torch.sigmoid(x)

        # If model multiclass
        if self.out_channel > 1:
            # Remove background class
            preds = preds[:, 1:, ]

        return preds


class ConvBlock(Module):
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


class SimpleBlock(Module):
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


class GridAttentionBlock(Module):
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
        super(GridAttentionBlock, self).__init__()

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
            raise NotImplementedError

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


class UnetGridGatingSignal3DBlock(Module):
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
        super(UnetGridGatingSignal3DBlock, self).__init__()

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


class FiLMlayerBlock(Module):
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
        super(FiLMlayerBlock, self).__init__()

        self.batch_size = None
        self.height = None
        self.width = None
        self.depth = None
        self.feature_size = None
        self.generator = FiLMgenerator(n_metadata, n_channels)
        # Add the parameters gammas and betas to access them out of the class.
        self.gammas = None
        self.betas = None

    def forward(self, feature_maps, context, w_shared):
        data_shape = feature_maps.data.shape
        if len(data_shape) == 4:
            _, self.feature_size, self.height, self.width = data_shape
        elif len(data_shape) == 5:
            _, self.feature_size, self.height, self.width, self.depth = data_shape
        else:
            raise ValueError("Data should be either 2D (tensor length: 4) or 3D (tensor length: 5), found shape: {}".format(data_shape))

        if torch.cuda.is_available():
            context = torch.Tensor(context).cuda()
        else:
            context = torch.Tensor(context)

        # Estimate the FiLM parameters using a FiLM generator from the contioning metadata
        film_params, new_w_shared = self.generator(context, w_shared)

        # FiLM applies a different affine transformation to each channel,
        # consistent accross spatial locations
        if len(data_shape) == 4:
            film_params = film_params.unsqueeze(-1).unsqueeze(-1)
            film_params = film_params.repeat(1, 1, self.height, self.width)
        else:
            film_params = film_params.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            film_params = film_params.repeat(1, 1, self.height, self.width, self.depth)

        self.gammas = film_params[:, :self.feature_size, ]
        self.betas = film_params[:, self.feature_size:, ]

        # Apply the linear modulation
        output = self.gammas * feature_maps + self.betas

        return output, new_w_shared


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
        if shared_weights is not None:  # weight sharing
            self.linear1.weight = shared_weights[0]
            self.linear2.weight = shared_weights[1]

        x = self.linear1(x)
        x = self.sig(x)
        x = self.linear2(x)
        x = self.sig(x)
        x = self.linear3(x)

        out = self.sig(x)
        return out, [self.linear1.weight, self.linear2.weight]