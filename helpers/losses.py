# From https://github.com/usc-sail/media-eval-2020/blob/main/train/losses.py

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
# import librosa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
An implementation of focal loss for the multi-class, multilabel case.
It is based off of the Tensorflow implementation for SigmoidFocalCrossEntropy, and extended to the multilabel case.
Input is expected to be of shape (num_batches, num_classes). Output will also be of shape (num_batches, num_classes), 
allowing for further modification of the loss.
Note that alpha can be given as a float, which is applied uniformily to each class logit,
or it can be explicitly set for each class by providing a vector of length equal to the number of classes. 
'''


def focal_loss(input: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
               eps: float = 1e-8) -> torch.Tensor:
    # Numerical stability
    input = torch.clamp(input, min=eps, max=-1 * eps + 1.)

    # Get the cross_entropy for each entry
    bce = F.binary_cross_entropy(input, target, reduction='none')

    p_t = (target * input) + ((1 - target) * (1 - input))

    # If alpha is less than 0, set the alpha factor (a_t) to be uniformally 1 for all classes
    if alpha < 0:
        alpha_factor = target + (1 - target)
    else:
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)

    modulating_factor = torch.pow((1.0 - p_t), gamma)

    # compute the final element-wise loss and return
    return alpha_factor * modulating_factor * bce


'''
Args:
input: A torch tensor of class predictions (logits if from_logits is True, else sigmoid outputs). Shape: (batch_size, num_classes)
target: A torch tensor of binary ground truth labels. Shape: (batch_size, num_classes)
alpha: Focal loss weight, as defined in https://arxiv.org/abs/1708.02002. Float.
gamma: Focal loss focusing parameter. Float.
reduction: How to reduce from element-wise loss to total loss. String.
Returns:
Total loss as a single float value.
'''


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 from_logits: bool = True, reduction: str = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.from_logits: bool = from_logits
        self.eps: float = 1e-8

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:

        # If necessary, apply an activation function to raw input logits
        if self.from_logits:
            y = torch.sigmoid(input)
        else:
            y = input

        loss = focal_loss(y, target, self.alpha, self.gamma, eps=self.eps)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            # return loss
            # Default to batch average
            return torch.mean(torch.sum(loss, axis=-1))


'''
MIXUP
'''


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# Modify loss function for mixup
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
