from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _one_hot_targets(
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if targets.ndim == 4:
        return targets.float(), None

    if targets.ndim != 3:
        raise ValueError(
            "Targets must have shape (B, H, W) for class indices or "
            f"(B, C, H, W) for one-hot masks, got {tuple(targets.shape)}."
        )

    valid_mask = None
    safe_targets = targets.long()
    if ignore_index is not None:
        valid_mask = safe_targets != ignore_index
        safe_targets = safe_targets.masked_fill(~valid_mask, 0)

    one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.float()
    if valid_mask is not None:
        one_hot = one_hot * valid_mask.unsqueeze(1)

    return one_hot, valid_mask


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Multiclass soft Dice loss computed directly from logits."""

    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    target_one_hot, valid_mask = _one_hot_targets(targets, num_classes, ignore_index)
    target_one_hot = target_one_hot.to(device=logits.device, dtype=probs.dtype)

    if valid_mask is not None:
        valid_mask = valid_mask.to(device=logits.device)
        probs = probs * valid_mask.unsqueeze(1)

    if not include_background and num_classes > 1:
        probs = probs[:, 1:]
        target_one_hot = target_one_hot[:, 1:]
        if class_weights is not None:
            class_weights = class_weights[1:]

    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_one_hot, dim=dims)
    denominator = torch.sum(probs + target_one_hot, dim=dims)
    per_class_loss = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth))

    if class_weights is not None:
        weights = class_weights.to(device=logits.device, dtype=per_class_loss.dtype)
        weights = weights / weights.sum().clamp_min(smooth)
        per_class_loss = per_class_loss * weights
        return per_class_loss.sum()

    return per_class_loss.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Shared Dice + Cross-Entropy loss for ResNet-UNet and TransUNet."""

    if logits.ndim != 4:
        raise ValueError(f"Logits must have shape (B, C, H, W), got {tuple(logits.shape)}.")
    if targets.ndim == 4:
        ce_targets = torch.argmax(targets, dim=1)
    else:
        ce_targets = targets.long()

    weights = class_weights.to(logits.device) if class_weights is not None else None
    ce_ignore_index = -100 if ignore_index is None else ignore_index
    ce = F.cross_entropy(logits, ce_targets, weight=weights, ignore_index=ce_ignore_index)
    dice = dice_loss(
        logits=logits,
        targets=targets,
        class_weights=class_weights,
        ignore_index=ignore_index,
        include_background=include_background,
        smooth=smooth,
    )
    return dice_weight * dice + ce_weight * ce


class DiceCrossEntropyLoss(nn.Module):
    """Module wrapper around the shared segmentation loss formula."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.smooth = smooth
        self.register_buffer("class_weights", class_weights.float() if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return segmentation_loss(
            logits=logits,
            targets=targets,
            class_weights=self.class_weights,
            dice_weight=self.dice_weight,
            ce_weight=self.ce_weight,
            ignore_index=self.ignore_index,
            include_background=self.include_background,
            smooth=self.smooth,
        )