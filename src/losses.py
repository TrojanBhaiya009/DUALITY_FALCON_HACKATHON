import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, num_classes=10):
        probs    = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes).permute(0,3,1,2).float()
        inter    = (probs * targets_oh).sum(dim=(2,3))
        union    = probs.sum(dim=(2,3)) + targets_oh.sum(dim=(2,3))
        dice     = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for better class imbalance handling"""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Focal + Dice — best for imbalanced segmentation
    Higher dice weight pushes mIoU directly."""
    def __init__(self, num_classes=10, focal_w=0.4, dice_w=0.6):
        super().__init__()
        self.focal       = FocalLoss(gamma=2.0)
        self.dice        = DiceLoss()
        self.focal_w     = focal_w
        self.dice_w      = dice_w
        self.num_classes = num_classes

    def forward(self, logits, targets):
        return (
            self.focal_w * self.focal(logits, targets) +
            self.dice_w  * self.dice(logits, targets, self.num_classes)
        )
