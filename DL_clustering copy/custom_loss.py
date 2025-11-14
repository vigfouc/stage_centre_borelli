import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=3.5, high_weight=5.0):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.high_weight = high_weight

    def forward(self, preds, targets):
        # Compute base MSE loss
        loss = (preds - targets) ** 2

        weights = torch.ones_like(targets)
        weights[targets > self.threshold] = self.high_weight

        weighted_loss = weights * loss

        return weighted_loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):

        input = input.view(-1)
        target = target.view(-1)

        prob = torch.clamp(input, 1e-6, 1 - 1e-6)  # avoid log(0)

        pt = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        loss = -focal_weight * torch.log(pt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
        return loss
