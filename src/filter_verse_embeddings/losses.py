"""Loss helpers for verse-embedding models."""
from __future__ import annotations
from typing import Literal
import torch
import torch.nn.functional as F
IGNORE_INDEX = -100
INFO_NCE_MASK_FILL = -10000.0
INFO_NCE_NORM_EPS = 1e-06

def mlm_loss_and_count(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction='sum')
    count = (labels != IGNORE_INDEX).sum().to(dtype=loss_sum.dtype)
    return (loss_sum, count)

def info_nce_logits(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor, temperature: float, negative_mask: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    if student_embeddings.ndim != 2 or teacher_embeddings.ndim != 2:
        raise ValueError('student_embeddings and teacher_embeddings must be rank-2 tensors.')
    if student_embeddings.shape != teacher_embeddings.shape:
        raise ValueError('student_embeddings and teacher_embeddings must have the same shape.')
    if student_embeddings.shape[0] == 0:
        empty = student_embeddings.new_empty((0, 0))
        return (empty, empty.to(dtype=torch.bool))
    if temperature <= 0:
        raise ValueError('temperature must be positive.')
    student_den = student_embeddings.norm(dim=-1, keepdim=True).clamp_min(INFO_NCE_NORM_EPS)
    teacher_den = teacher_embeddings.norm(dim=-1, keepdim=True).clamp_min(INFO_NCE_NORM_EPS)
    student_norm = student_embeddings / student_den
    teacher_norm = teacher_embeddings / teacher_den
    logits = student_norm @ teacher_norm.transpose(0, 1) / temperature
    if negative_mask is None:
        allowed = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
    else:
        if negative_mask.shape != logits.shape:
            raise ValueError('negative_mask must match the logits shape.')
        allowed = negative_mask.to(dtype=torch.bool, device=logits.device)
    diag = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    allowed = allowed | diag
    valid_per_row = allowed.sum(dim=1)
    if torch.any(valid_per_row == 0):
        raise ValueError('InfoNCE row has no valid comparisons after masking.')
    logits = logits.masked_fill(~allowed, INFO_NCE_MASK_FILL)
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite logits before InfoNCE cross-entropy.')
    return (logits, allowed)

def symmetric_info_nce_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor, temperature: float, negative_mask: torch.Tensor | None=None, direction: Literal['forward_only', 'symmetric']='symmetric') -> torch.Tensor:
    if direction not in {'forward_only', 'symmetric'}:
        raise ValueError(f'Unsupported contrastive direction: {direction}')
    (logits, _) = info_nce_logits(student_embeddings=student_embeddings, teacher_embeddings=teacher_embeddings, temperature=temperature, negative_mask=negative_mask)
    if logits.numel() == 0:
        return student_embeddings.sum() * 0.0
    labels = torch.arange(logits.size(0), device=logits.device)
    forward_loss = F.cross_entropy(logits, labels)
    if direction == 'forward_only':
        return forward_loss
    else:
        backward_loss = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (forward_loss + backward_loss)

def weighted_cosine_alignment_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor, weights: torch.Tensor | None=None) -> torch.Tensor:
    if student_embeddings.ndim != 2 or teacher_embeddings.ndim != 2:
        raise ValueError('student_embeddings and teacher_embeddings must be rank-2 tensors.')
    if student_embeddings.shape != teacher_embeddings.shape:
        raise ValueError('student_embeddings and teacher_embeddings must have the same shape.')
    if student_embeddings.shape[0] == 0:
        return student_embeddings.sum() * 0.0
    student_norm = student_embeddings / student_embeddings.norm(dim=-1, keepdim=True).clamp_min(INFO_NCE_NORM_EPS)
    teacher_norm = teacher_embeddings / teacher_embeddings.norm(dim=-1, keepdim=True).clamp_min(INFO_NCE_NORM_EPS)
    distances = 1.0 - (student_norm * teacher_norm).sum(dim=-1)
    if weights is None:
        return distances.mean()
    if weights.shape != distances.shape:
        raise ValueError('weights must match the number of embedding pairs.')
    weights = weights.to(device=distances.device, dtype=distances.dtype)
    denominator = weights.sum().clamp_min(INFO_NCE_NORM_EPS)
    return (weights * distances).sum() / denominator
