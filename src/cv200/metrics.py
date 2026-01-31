from __future__ import annotations

import torch


@torch.inference_mode()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """
    Returns scalar tensor in [0, 1].
    logits: [B, C]
    targets: [B]
    """
    if k <= 0:
        raise ValueError("k must be positive")
    _, pred = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1)
    correct = pred.eq(targets.view(-1, 1)).any(dim=-1)
    return correct.float().mean()
