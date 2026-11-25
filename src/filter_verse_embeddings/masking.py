"""MLM masking utilities."""
from __future__ import annotations
import torch

def apply_mlm_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, mask_token_id: int, vocab_size: int, mlm_probability: float, special_token_ids: set[int]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    probability_matrix = torch.full_like(input_ids, mlm_probability, dtype=torch.float32)
    special_tokens_mask = torch.isin(input_ids, torch.tensor(list(special_token_ids), device=input_ids.device, dtype=input_ids.dtype))
    probability_matrix = probability_matrix.masked_fill(special_tokens_mask, 0.0)
    probability_matrix = probability_matrix.masked_fill(attention_mask == 0, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    masked_inputs = input_ids.clone()
    replace_with_mask = torch.bernoulli(torch.full_like(input_ids, 0.8, dtype=torch.float32)).bool() & masked_indices
    masked_inputs[replace_with_mask] = mask_token_id
    replace_with_random = torch.bernoulli(torch.full_like(input_ids, 0.5, dtype=torch.float32)).bool() & masked_indices & ~replace_with_mask
    disallowed_token_ids = set(special_token_ids)
    disallowed_token_ids.add(mask_token_id)
    allowed_token_ids = torch.tensor([token_id for token_id in range(vocab_size) if token_id not in disallowed_token_ids], device=input_ids.device, dtype=input_ids.dtype)
    if allowed_token_ids.numel() > 0:
        random_token_positions = torch.randint(low=0, high=allowed_token_ids.numel(), size=input_ids.shape, device=input_ids.device)
        random_tokens = allowed_token_ids[random_token_positions]
        masked_inputs[replace_with_random] = random_tokens[replace_with_random]
    return (masked_inputs, labels)
