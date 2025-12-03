import torch
import torch.nn as nn


def dice_loss(pred, target, epsilon=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice_score = (2.0 * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    return 1 - dice_score


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        return dice_loss(torch.sigmoid(pred), target, self.epsilon)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        """
        Combines BCE and Dice losses.

        Args:
            bce_weight (float): Weight of the BCE loss.
            dice_weight (float): Weight of the Dice loss.
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        """
        Binary Focal Loss
        Args:
            alpha: weight for positive class (float, e.g. 0.75)
            gamma: focusing parameter (float, e.g. 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # If inputs are logits, apply sigmoid
        inputs = torch.sigmoid(inputs)

        eps = 1e-8
        inputs = inputs.clamp(min=eps, max=1.0 - eps)

        # Calculate p_t
        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Focal loss
        loss = -alpha_t * ((1 - p_t) ** self.gamma) * torch.log(p_t)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalTverskyLoss(torch.nn.Module):
    r"""Focal Tversky loss for imbalanced binary segmentation.

    Computes (1 - Tversky)^γ with p = sigmoid(logits). Tversky balances
    FN and FP via α, β; γ>1 focuses learning on hard/small targets.

    Args:
        alpha (float): FN weight (↑alpha → higher recall). Default: 0.7
        beta  (float): FP weight (↑beta  → higher precision). Default: 0.3
        gamma (float): Focal exponent (>1 emphasizes hard examples). Default: 1.33
        eps   (float): Numerical stability. Default: 1e-7
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, eps=1e-7):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.eps = alpha, beta, gamma, eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = tuple(range(2, probs.ndim))
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1 - targets)).sum(dim=dims)
        fn = ((1 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.eps) / (tp + self.alpha*fn + self.beta*fp + self.eps)
        loss = (1 - tversky) ** self.gamma
        return loss.mean()
