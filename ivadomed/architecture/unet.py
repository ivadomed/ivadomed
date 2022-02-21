import torch
from torch import nn as nn
from torch.nn import Module

from ivadomed.architecture.blocks import EncoderBlock, DecoderBlock, GridAttentionBlock, UnetGridGatingSignal3DBlock, \
    FiLMlayerBlock


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
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.
        **kwargs:

    Attributes:
        encoder (ivadomed.architecture.blocks.EncoderBlock): U-Net encoder.
        decoder (ivadomed.architecture.blocks.DecoderBlock): U-net decoder.
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, final_activation='sigmoid',
                 is_2d=True, n_filters=64, **kwargs):
        super(Unet, self).__init__()

        # Encoder path
        self.encoder = EncoderBlock(in_channel=in_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                                    is_2d=is_2d, n_filters=n_filters)

        # Decoder path
        self.decoder = DecoderBlock(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                                    final_activation=final_activation, is_2d=is_2d, n_filters=n_filters)

    def forward(self, x):
        features, _ = self.encoder(x)
        preds = self.decoder(features)

        return preds


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
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        **kwargs:

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        contrasts (list): List of contrasts.
        Encoder_mod (ModuleDict): Contains encoder for each modality.
        decoder (DecoderBlock): U-Net decoder.
    """

    def __init__(self, contrasts, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, **kwargs):
        super(HeMISUnet, self).__init__()
        self.depth = depth
        self.contrasts = contrasts

        # Encoder path
        self.Encoder_mod = nn.ModuleDict(
            [['Encoder_{}'.format(Mod), EncoderBlock(in_channel=1, depth=depth, dropout_rate=dropout_rate,
                                                     bn_momentum=bn_momentum)] for Mod in self.contrasts])

        # Decoder path
        self.decoder = DecoderBlock(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate,
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


class Modified3DUNet(nn.Module):
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
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".
        **kwargs:

    Attributes:
        in_channels (int): Number of channels in the input image.
        n_classes (int): Number of channels in the output image.
        base_n_filter (int): Number of base filters in the U-Net.
        attention (bool): Boolean indicating whether the attention module is on or not.
        momentum (float): Batch normalization momentum.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".

    Note: All layers are defined as attributes and used in the forward method.
    """

    def __init__(self, in_channel, out_channel, n_filters=16, attention=False, dropout_rate=0.3, bn_momentum=0.1,
                 final_activation="sigmoid", n_metadata=None, film_layers=None, **kwargs):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channel
        self.n_classes = out_channel
        self.base_n_filter = n_filters
        self.attention = attention
        self.momentum = bn_momentum
        self.final_activation = final_activation

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=dropout_rate)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(
            self.in_channels, self.base_n_filter,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.film_layer1 = FiLMlayerBlock(n_metadata, self.base_n_filter) if film_layers and film_layers[0] else None
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
        self.film_layer2 = FiLMlayerBlock(n_metadata, self.base_n_filter * 2) if film_layers and film_layers[1] else None
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(
            self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2, momentum=self.momentum)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(
            self.base_n_filter * 2, self.base_n_filter * 4,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.film_layer3 = FiLMlayerBlock(n_metadata, self.base_n_filter * 4) if film_layers and film_layers[2] else None
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(
            self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4, momentum=self.momentum)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(
            self.base_n_filter * 4, self.base_n_filter * 8,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.film_layer4 = FiLMlayerBlock(n_metadata, self.base_n_filter * 8) if film_layers and film_layers[3] else None
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

        if film_layers and film_layers[4]:
            self.norm_lrelu_0 = self.norm_lrelu(self.base_n_filter * 16)
            self.film_layer5 = FiLMlayerBlock(n_metadata, self.base_n_filter * 16)
            self.upscale_conv_norm_lrelu_0 = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                          self.base_n_filter * 8)
        else:
            self.norm_lrelu_upscale_conv_norm_lrelu_l0 = \
                self.norm_lrelu_upscale_conv_norm_lrelu(
                    self.base_n_filter * 16, self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(
            self.base_n_filter * 8, self.base_n_filter * 8,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.film_layer6 = FiLMlayerBlock(n_metadata, self.base_n_filter * 8) if film_layers and film_layers[5] else None
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8, momentum=self.momentum)

        # Attention UNet
        if self.attention:
            self.gating = UnetGridGatingSignal3DBlock(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=(1, 1, 1),
                                                      is_batchnorm=True)

            # attention blocks
            self.attentionblock2 = GridAttentionBlock(in_channels=self.base_n_filter * 2,
                                                      gating_channels=self.base_n_filter * 8,
                                                      inter_channels=self.base_n_filter * 2,
                                                      sub_sample_factor=(2, 2, 2),
                                                      )
            self.attentionblock3 = GridAttentionBlock(in_channels=self.base_n_filter * 4,
                                                      gating_channels=self.base_n_filter * 8,
                                                      inter_channels=self.base_n_filter * 4,
                                                      sub_sample_factor=(2, 2, 2),
                                                      )
            self.attentionblock4 = GridAttentionBlock(in_channels=self.base_n_filter * 8,
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
        self.film_layer7 = FiLMlayerBlock(n_metadata, self.base_n_filter * 4) if film_layers and film_layers[6] else None
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
        self.film_layer8 = FiLMlayerBlock(n_metadata, self.base_n_filter * 2) if film_layers and film_layers[7] else None
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
        self.film_layer9 = FiLMlayerBlock(n_metadata, self.base_n_filter) if film_layers and film_layers[8] else None
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
        # self.film_layer10 = FiLMlayer(n_metadata, ) if film_layers and film_layers[9] else None

        self.ds2_1x1_conv3d = nn.Conv3d(
            self.base_n_filter * 8, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.ds3_1x1_conv3d = nn.Conv3d(
            self.base_n_filter * 4, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.film_layer10 = FiLMlayerBlock(n_metadata, self.n_classes) if film_layers and film_layers[9] else None

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

    def norm_lrelu(self, feat_in):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in, momentum=self.momentum),
            nn.LeakyReLU())

    def upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out, momentum=self.momentum),
            nn.LeakyReLU())

    def forward(self, x, context=None, w_film=None):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out

        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        out = self.lrelu(out)
        context_1 = out
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        if hasattr(self, 'film_layer1') and self.film_layer1:
            out, w_film = self.film_layer1(out, context, w_film)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        if hasattr(self, 'film_layer2') and self.film_layer2:
            out, w_film = self.film_layer2(out, context, w_film)
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
        if hasattr(self, 'film_layer3') and self.film_layer3:
            out, w_film = self.film_layer3(out, context, w_film)
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
        if hasattr(self, 'film_layer4') and self.film_layer4:
            out, w_film = self.film_layer4(out, context, w_film)
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

        if hasattr(self, 'film_layer5') and self.film_layer5:
            out = self.norm_lrelu_0(out)
            out, w_film = self.film_layer5(out, context, w_film)
            out = self.upscale_conv_norm_lrelu_0(out)
        else:
            out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)

        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        if hasattr(self, 'film_layer6') and self.film_layer6:
            out, w_film = self.film_layer6(out, context, w_film)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        if hasattr(self, 'film_layer7') and self.film_layer7:
            out, w_film = self.film_layer7(out, context, w_film)


        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        if hasattr(self, 'film_layer8') and self.film_layer8:
            out, w_film = self.film_layer8(out, context, w_film)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        if hasattr(self, 'film_layer9') and self.film_layer9:
            out, w_film = self.film_layer9(out, context, w_film)

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
        if hasattr(self, 'film_layer10') and self.film_layer10:
            out, w_film = self.film_layer10(out, context, w_film)
        seg_layer = out

        if hasattr(self, "final_activation") and self.final_activation not in ["softmax", "relu", "sigmoid"]:
            raise ValueError("final_activation value has to be either softmax, relu, or sigmoid")
        elif hasattr(self, "final_activation") and self.final_activation == "softmax":
            out = self.softmax(out)
        elif hasattr(self, "final_activation") and self.final_activation == "relu":
            out = nn.ReLU()(seg_layer) / nn.ReLU()(seg_layer).max() if bool(nn.ReLU()(seg_layer).max()) \
                else nn.ReLU()(seg_layer)
        else:
            out = torch.sigmoid(out)

        if self.n_classes > 1:
            # Remove background class
            out = out[:, 1:, ]

        return out


class UNet3D(Modified3DUNet):
    """To ensure retrocompatibility, when calling UNet3D (old model name), Modified3DUNet will be called.
    see Modified3DUNet to learn more about parameters.
    """
    def __init__(self, in_channel, out_channel, n_filters=16, attention=False, dropout_rate=0.3, bn_momentum=0.1,
                 final_activation="sigmoid", n_metadata=None, film_layers=None, **kwargs):
        super(UNet3D, self).__init__()
        Modified3DUNet(in_channel=in_channel, out_channel=out_channel, n_filters=n_filters, attention=attention,
                       dropout_rate=dropout_rate, bn_momentum=bn_momentum, final_activation=final_activation,
                       n_metadata=n_metadata, film_layers=film_layers, **kwargs)


class FiLMedUnet(Unet):
    """U-Net network containing FiLM layers to condition the model with another data type (i.e. not an image).

    Args:
        n_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        is_2d (bool): Indicates dimensionality of model.
        n_filters (int):  Number of base filters in the U-Net.
        **kwargs:

    Attributes:
        encoder (EncoderBlock): U-Net encoder.
        decoder (DecoderBlock): U-net decoder.
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, dropout_rate=0.3,
                 bn_momentum=0.1, n_metadata=None, film_layers=None, is_2d=True, n_filters=64, **kwargs):
        super().__init__(in_channel=in_channel, out_channel=out_channel, depth=depth,
                         dropout_rate=dropout_rate, bn_momentum=bn_momentum)

        # Verify if the length of boolean FiLM layers corresponds to the depth
        if film_layers:
            if len(film_layers) != 2 * depth + 2:
                raise ValueError("The number of FiLM layers {} entered does not correspond to the "
                                 "UNet depth. There should 2 * depth + 2 layers.".format(len(film_layers)))
        else:
            film_layers = [0] * (2 * depth + 2)
        # Encoder path
        self.encoder = EncoderBlock(in_channel=in_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                                    n_metadata=n_metadata, film_layers=film_layers, is_2d=is_2d, n_filters=n_filters)
        # Decoder path
        self.decoder = DecoderBlock(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                                    n_metadata=n_metadata, film_layers=film_layers, is_2d=is_2d, n_filters=n_filters)

    def forward(self, x, context=None):
        features, w_film = self.encoder(x, context)
        preds = self.decoder(features, context, w_film)

        return preds