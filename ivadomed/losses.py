import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(input, target):
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        input = input.clamp(self.eps, 1. - self.eps)

        cross_entropy = - (target * torch.log(input) + (1 - target) * torch.log(1-input))  # eq1
        logpt = - cross_entropy
        pt = torch.exp(logpt)  # eq2

        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        balanced_cross_entropy = - at * logpt  # eq3

        focal_loss = balanced_cross_entropy * ((1 - pt) ** self.gamma)  # eq5

        return focal_loss.sum()
        #return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    """
    Motivated by https://arxiv.org/pdf/1809.00076.pdf
    :param alpha: to bring the dice and focal losses at similar scale.
    :param gamma: gamma value used in the focal loss.
    """
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss: https://arxiv.org/pdf/1707.03237
    """
    def __init__(self, epsilon=1e-5):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        input = input.view(-1)
        target = target.view(-1)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = nn.Parameter(1. / (target_sum * target_sum).clamp(min=self.epsilon))

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)

