"""
Focal Loss with automatic class balancing for occupancy prediction.
Solves extreme class imbalance (95% free space vs <1% rare classes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class FocalLossBalanced(nn.Module):
    """Focal Loss with automatic class weighting for imbalanced occupancy data.

    This loss addresses two problems:
    1. Extreme class imbalance (free space dominates)
    2. Hard examples being overwhelmed by easy examples

    Args:
        alpha (float or list): Weighting factor in [0, 1] to balance positive/negative examples,
            or a list of weights for each class. Default: None (auto-compute)
        gamma (float): Exponent of the modulating factor (1 - p_t)^gamma. Default: 2.0
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
        loss_weight (float): Weight of the loss. Default: 1.0
        auto_balance (bool): Automatically compute class weights from batch statistics. Default: True
        ignore_index (int): Specifies a target value that is ignored. Default: 255
    """

    def __init__(self,
                 alpha=None,
                 gamma=2.0,
                 reduction='mean',
                 loss_weight=1.0,
                 auto_balance=True,
                 ignore_index=255,
                 num_classes=16):
        super(FocalLossBalanced, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.auto_balance = auto_balance
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        # Initialize alpha if provided as list
        if isinstance(alpha, (list, tuple)):
            self.register_buffer('alpha_t', torch.tensor(alpha, dtype=torch.float32))
        elif alpha is not None:
            self.register_buffer('alpha_t', torch.full((num_classes,), alpha, dtype=torch.float32))
        else:
            self.alpha_t = None

        # Running statistics for auto-balancing
        if auto_balance:
            self.register_buffer('class_counts', torch.zeros(num_classes, dtype=torch.float32))
            self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))

    def update_class_weights(self, targets):
        """Update running class statistics for auto-balancing.

        Args:
            targets (Tensor): Ground truth labels [N]
        """
        if not self.auto_balance or not self.training:
            return

        # Count classes in current batch
        valid_mask = (targets != self.ignore_index)
        valid_targets = targets[valid_mask]

        for c in range(self.num_classes):
            count = (valid_targets == c).sum().float()
            self.class_counts[c] += count

        self.update_count += 1

        # Recompute alpha every 100 updates
        if self.update_count % 100 == 0:
            total = self.class_counts.sum()
            if total > 0:
                # Inverse frequency weighting
                freq = self.class_counts / total
                # Avoid division by zero
                weights = 1.0 / (freq + 1e-6)
                # Normalize so that mean weight = 1.0
                weights = weights / weights.mean()
                # Clip extreme weights
                weights = torch.clamp(weights, min=0.1, max=10.0)
                self.alpha_t = weights

    def forward(self, pred, target):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits [N, C] or [N, C, H, W, D]
            target (Tensor): Ground truth labels [N] or [N, H, W, D]

        Returns:
            Tensor: Calculated focal loss
        """
        # Reshape inputs if needed
        if pred.dim() == 5:  # [N, C, H, W, D]
            N, C, H, W, D = pred.shape
            pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [N*H*W*D, C]
            target = target.reshape(-1)  # [N*H*W*D]
        elif pred.dim() == 2:  # [N, C]
            pass
        else:
            raise ValueError(f"Unsupported pred shape: {pred.shape}")

        # Update class weights if auto-balancing
        self.update_class_weights(target)

        # Filter out ignore_index
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            return pred.sum() * 0.0  # Return zero loss if all ignored

        pred = pred[valid_mask]
        target = target[valid_mask]

        # Get class probabilities
        p = F.softmax(pred, dim=-1)  # [N, C]
        ce_loss = F.cross_entropy(pred, target, reduction='none')  # [N]

        # Get probability of true class
        p_t = p[torch.arange(len(target)), target]  # [N]

        # Focal modulation term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha balancing if available
        if self.alpha_t is not None:
            alpha_t = self.alpha_t[target]  # [N]
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # else: 'none', return as-is

        return self.loss_weight * loss

    def extra_repr(self):
        """Extra representation string."""
        s = f'gamma={self.gamma}, reduction={self.reduction}, loss_weight={self.loss_weight}'
        if self.alpha_t is not None:
            s += f', alpha={self.alpha_t[:5].tolist()}...'
        return s


@LOSSES.register_module()
class OccupancyFocalLoss(FocalLossBalanced):
    """Convenience wrapper with sensible defaults for occupancy prediction.

    Compared to standard FocalLoss:
    - Auto-balancing enabled by default
    - Higher gamma (2.5) for harder focusing
    - Ignores index 255 by default
    """

    def __init__(self,
                 gamma=2.5,
                 reduction='mean',
                 loss_weight=2.0,
                 ignore_index=255,
                 num_classes=16):
        super().__init__(
            alpha=None,  # Will be auto-computed
            gamma=gamma,
            reduction=reduction,
            loss_weight=loss_weight,
            auto_balance=True,
            ignore_index=ignore_index,
            num_classes=num_classes
        )
