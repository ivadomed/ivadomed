import torch
import torch.nn as nn
import scipy
import numpy as np

class MultiClassDiceLoss(nn.Module):
    """ Multi-class Dice Loss.

    Inspired from https://arxiv.org/pdf/1802.10508.

    :param classes_of_interest:  list containing the index of a class which dice will be added to the loss.
    """

    def __init__(self, classes_of_interest):
        super(MultiClassDiceLoss, self).__init__()
        self.classes_of_interest = classes_of_interest
        self.dice_loss = DiceLoss()

    def forward(self, prediction, target):
        dice_per_class = 0
        n_classes = prediction.shape[1]

        if self.classes_of_interest is None:
            self.classes_of_interest = range(n_classes)

        for i in self.classes_of_interest:
            dice_per_class += self.dice_loss(prediction[:, i, ], target[:, i, ])

        return dice_per_class / len(self.classes_of_interest)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

        return - (2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)


class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary Cross Entropy Loss, calls https://pytorch.org/docs/master/generated/torch.nn.BCELoss.html#bceloss
    """
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss_fct = nn.BCELoss()

    def forward(self, prediction, target):
        return self.loss_fct(prediction, target)


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
    Args:
            beta: to bring the dice and focal losses at similar scale.
            gamma: gamma value used in the focal loss.
            alpha: alpha value used in the focal loss.
    """

    def __init__(self, beta=1, gamma=2, alpha=0.25):
        super().__init__()
        self.beta = beta
        self.focal = FocalLoss(gamma, alpha)
        self.dice = DiceLoss()

    def forward(self, input, target):
        dc_loss = - self.dice(input, target)
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


class L2loss(nn.Module):
    """
    L2 loss between two images : inputs and target
    """

    def __init__(self):
        super(L2_loss, self).__init__()

    def forward(self, input, target):
        return torch.sum((input - target) ** 2) / 2


class AdapWingLoss(nn.Module):
    """
    adaptive Wing loss : https://arxiv.org/abs/1904.07399
    Used for heatmap ground truth.

    """

    def __init__(self):
        super(AdapWingLoss, self).__init__()

    def forward(self,input,target):
        theta = 0.5
        alpha = 2.1
        w = 14
        e = 1
        A = w * (1 / (1 + torch.pow(theta / e, alpha - target))) * (alpha - target) * torch.pow(theta / e,
                                                                                              alpha - target - 1) * (1 / e)
        C = (theta * A - w * torch.log(1 + torch.pow(theta / e, alpha - target)))

        batch_size = target.size()[0]
        hm_num = target.size()[1]

        mask = torch.zeros_like(target)
        # W=10
        kernel = scipy.ndimage.morphology.generate_binary_structure(2,2)
        for i in range(batch_size):
            img_list = []

            img_list.append(np.round(target[i].cpu().numpy() * 255))
            img_merge = np.concatenate(img_list)
            img_dilate = scipy.ndimage.morphology.binary_opening(img_merge, kernel)
            img_dilate[img_dilate < 51] = 1  # 0*W+1
            img_dilate[img_dilate >= 51] = 11  # 1*W+1
            img_dilate = np.array(img_dilate, dtype=np.int)

            mask[i] = torch.tensor(img_dilate)

        diff_hm = torch.abs(target - input)
        AWingLoss = A * diff_hm - C
        idx = diff_hm < theta
        AWingLoss[idx] = w * torch.log(1 + torch.pow(diff_hm / e, alpha - target))[idx]

        AWingLoss *= mask
        sum_loss = torch.sum(AWingLoss)
        all_pixel = torch.sum(mask)
        mean_loss = sum_loss  # / all_pixel

        return mean_loss
