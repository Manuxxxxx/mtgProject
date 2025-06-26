import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of shape [num_classes] or scalar
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)  # p_t = p if y=1 else 1-p

        modulating_factor = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * modulating_factor * ce_loss
        else:
            loss = modulating_factor * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
