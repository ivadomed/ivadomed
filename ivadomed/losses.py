import torch
import torch.nn as nn


class MultiClassDiceLoss(nn.Module):
    # Inspired from https://arxiv.org/pdf/1802.10508.pdf
    def forward(self, prediction, target):
        smooth = 1.0
        dice_per_class = 0
        n_classes = prediction.shape[1]
        for i in range(n_classes):
            iflat = prediction[:, i, ].reshape(-1)
            tflat = target[:, i, ].reshape(-1)
            intersection = (iflat * tflat).sum()
            dice_per_class += intersection / (iflat.sum() + tflat.sum() + smooth)
        return -(2.0 * dice_per_class) / n_classes


class CombinedDiceLoss(nn.Module):
    """
    :param classes_of_interest: list containing the index of a class which dice will be added to the loss.
    :param multiclass_dice: boolean representing whether or not multiclass dice is included in the current loss.
    :param dice: boolean representing whether or not dice is included in the current loss.
    """

    def __init__(self, class_of_interest, multi_class_dice, dice):
        super().__init__()
        self.class_of_interest = class_of_interest
        self.multi_class_dice = multi_class_dice
        self.dice = dice

    def forward(self, prediction, target):
        loss = torch.zeros(1).cuda()
        if len(self.class_of_interest):
            for idx in self.class_of_interest:
                loss += - dice_loss(prediction[:, idx, ], target[:, idx, ])
        if self.multi_class_dice:
            loss += MultiClassDiceLoss()(prediction, target)
        if self.dice:
            loss += - dice_loss(prediction, target)
        return loss


def dice_loss(input, target):
    smooth = 1.0

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()

    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        input = input.clamp(self.eps, 1. - self.eps)

        cross_entropy = - (target * torch.log(input) + (1 - target) * torch.log(1 - input))  # eq1
        logpt = - cross_entropy
        pt = torch.exp(logpt)  # eq2

        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        balanced_cross_entropy = - at * logpt  # eq3

        focal_loss = balanced_cross_entropy * ((1 - pt) ** self.gamma)  # eq5

        return focal_loss.sum()
        # return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    """
    Motivated by https://arxiv.org/pdf/1809.00076.pdf
    :param beta: to bring the dice and focal losses at similar scale.
    :param gamma: gamma value used in the focal loss.
    :param alpha: alpha value used in the focal loss.
    """

    def __init__(self, beta=1, gamma=2, alpha=0.25):
        super().__init__()
        self.beta = beta
        self.focal = FocalLoss(gamma, alpha)

    def forward(self, input, target):
        dc_loss = dice_loss(input, target)
        fc_loss = self.focal(input, target)

        # used to fine tune beta
        # with torch.no_grad():
        #     print('DICE loss:', dc_loss.cpu().numpy(), 'Focal loss:', fc_loss.cpu().numpy())
        #     log_dc_loss = torch.log(torch.clamp(dc_loss, 1e-7))
        #     log_fc_loss = torch.log(torch.clamp(fc_loss, 1e-7))
        #     print('Log DICE loss:', log_dc_loss.cpu().numpy(), 'Log Focal loss:', log_fc_loss.cpu().numpy())
        #     print('*'*20)

        loss = torch.log(torch.clamp(fc_loss, 1e-7)) - self.beta * torch.log(torch.clamp(dc_loss, 1e-7))

        return loss


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
