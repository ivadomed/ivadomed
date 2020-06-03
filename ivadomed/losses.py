import torch
import torch.nn as nn


# Inspired from https://arxiv.org/pdf/1802.10508.pdf
class MultiClassDiceLoss(nn.Module):
    """
    :param classes_of_interest:  list containing the index of a class which dice will be added to the loss.
    """

    def __init__(self, classes_of_interest):
        super(MultiClassDiceLoss, self).__init__()
        self.classes_of_interest = classes_of_interest

    def forward(self, prediction, target):
        dice_per_class = 0
        n_classes = prediction.shape[1]

        if self.classes_of_interest is None:
            self.classes_of_interest = range(n_classes)

        for i in self.classes_of_interest:
            dice_per_class += dice_loss(prediction[:, i, ], target[:, i, ])

        return - dice_per_class / len(self.classes_of_interest)


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

    def __init__(self, epsilon=1e-5, include_background=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.include_background = include_background

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        if self.include_background:
            # init
            input_background = torch.zeros(input.size(), dtype=input.dtype)
            target_background = torch.zeros(target.size(), dtype=target.dtype)
            # fill with opposite
            input_background[input == 0] = 1
            target_background[target == 0] = 1
            # Concat
            input = torch.cat([input, input_background], dim=1)
            target = torch.cat([target, target_background], dim=1)

        input = input.view(-1)
        target = target.view(-1)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = nn.Parameter(1. / (target_sum * target_sum).clamp(min=self.epsilon))

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return - 2. * intersect / denominator.clamp(min=self.epsilon)


class TverskyLoss(nn.Module):
    """Tversky Loss.

    Compute the Tversky loss defined in:
        Sadegh et al. (2017) Tversky loss function for image segmentation using 3D fully convolutional deep networks

    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        """
        Args:
            alpha (float): weight of false positive voxels
            beta  (float): weight of false negative voxels
            smooth (float): epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros

        Notes:
            - setting alpha=beta=0.5: equivalent to DiceLoss
            - default parameters were suggested by https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def tversky_index(self, y_pred, y_true):
        """Compute Tversky index

        Args:
            y_pred (torch Tensor): prediction
            y_true (torch Tensor): target

        Returns:
            float: Tversky index
        """
        # Compute TP
        tp = torch.sum(y_true * y_pred)
        # Compute FN
        fn = torch.sum(y_true * (1 - y_pred))
        # Compute FP
        fp = torch.sum((1 - y_true) * y_pred)
        # Compute Tversky for the current class, see Equation 3 of the original paper
        numerator = tp + self.smooth
        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky_label = numerator / denominator
        return tversky_label

    def forward(self, input, target):
        n_classes = input.shape[1]
        tversky_sum = 0.

        # TODO: Add class_of_interest?
        for i_label in range(n_classes):
            # Get samples for a given class
            y_pred, y_true = input[:, i_label, ], target[:, i_label, ]
            # Compute Tversky index
            tversky_sum += self.tversky_index(y_pred, y_true)

        return - tversky_sum / n_classes


class FocalTverskyLoss(TverskyLoss):
    """Focal Tversky Loss.

    Compute the Focal Tversky loss defined in:
        Abraham et al. (2018) A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation

    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0):
        """
        Args:
            alpha (float): weight of false positive voxels
            beta  (float): weight of false negative voxels
            gamma (float): typically between 1 and 3. Control between easy background and hard ROI training examples.
            smooth (float): epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros

        Notes:
            - setting alpha=beta=0.5 and gamma=1: equivalent to DiceLoss
            - default parameters were suggested by https://arxiv.org/pdf/1810.07842.pdf
        """
        super(TverskyLoss).__init__(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, input, target):
        n_classes = input.shape[1]
        focal_tversky_sum = 0.

        # TODO: Add class_of_interest?
        for i_label in range(n_classes):
            # Get samples for a given class
            y_pred, y_true = input[:, i_label, ], target[:, i_label, ]
            # Compute Tversky index
            tversky_index = self.tversky.tversky_index(y_pred, y_true)
            # Compute Focal Tversky loss, Equation 4 in the original paper
            focal_tversky_sum += torch.pow(1 - tversky_index, exponent=1 / self.gamma)

        # TODO: Take the opposite?
        return focal_tversky_sum / n_classes
