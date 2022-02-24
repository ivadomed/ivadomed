import torch
from torch import nn
from torch.nn import Module

from ivadomed.architecture.blocks import EncoderBlock, DecoderBlock


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