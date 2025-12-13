"""Focal loss used for occupancy classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class FocalLossBalanced(nn.Module):
    """Focal loss with optional running class weights."""

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
        """Update running class statistics for auto-balancing."""
        if not self.auto_balance or not self.training:
            return

        # Count classes in current batch
        valid_mask = (targets != self.ignore_index)
        valid_targets = targets[valid_mask]

        for c in range(self.num_classes):
            count = (valid_targets == c).sum().float()
            self.class_counts[c] += count

        self.update_count += 1

        # Recompute alpha every N updates
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

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Compute focal loss."""

        if pred.dim() == 2:  # [N, C] (Mask2Former samples points before loss)
            pass
        elif pred.dim() == 5:  # [N, C, H, W, D] dense logits (not typical for this head)
            N, C, H, W, D = pred.shape
            pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [N*H*W*D, C]
            target = target.reshape(-1)  # [N*H*W*D]
        else:
            raise ValueError(f"Unsupported pred shape: {pred.shape}")

        # Update class weights if auto-balancing
        self.update_class_weights(target)

        # Filter out ignore_index
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            return pred.sum() * 0.0  # Return zero loss if all ignored

        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        # Apply sample-wise weight if provided
        if weight is not None:
            weight_valid = weight[valid_mask]
        else:
            weight_valid = None

        # CE term
        ce_loss = F.cross_entropy(pred_valid, target_valid, reduction='none')  # [N]

        # Compute p_t without storing full softmax.
        log_p = F.log_softmax(pred_valid, dim=-1)  # [N, C]
        log_p_t = log_p[torch.arange(len(target_valid)), target_valid]  # [N]
        p_t = torch.exp(log_p_t)  # [N]
        del log_p, log_p_t

        focal_weight = (1 - p_t).pow_(self.gamma)
        del p_t

        # Apply alpha balancing if available
        if self.alpha_t is not None:
            alpha_t = self.alpha_t[target_valid]  # [N]
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        # Apply sample-wise weight
        if weight_valid is not None:
            loss = loss * weight_valid

        # Apply reduction
        reduction = reduction_override if reduction_override else self.reduction
        if reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif reduction == 'sum':
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
    """Wrapper with defaults used by this repo."""

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
