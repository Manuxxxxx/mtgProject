import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha            # Tensor [C] or scalar in [0,1]
        self.gamma = gamma
        self.pos_weight = pos_weight  # Tensor [C] or None
        self.reduction = reduction

    def forward(self, inputs, targets):
        # logits -> probs
        probs = torch.sigmoid(inputs)
        # BCE on logits, optionally with pos_weight for class imbalance
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", pos_weight=self.pos_weight
        )
        # focal modulating term
        p_t = probs * targets + (1 - probs) * (1 - targets)
        modulating_factor = (1 - p_t).pow(self.gamma)

        if self.alpha is not None:
            # per-class alpha for pos/neg
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * modulating_factor * ce_loss
        else:
            loss = modulating_factor * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
