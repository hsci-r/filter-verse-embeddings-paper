"""Small configurable transformer encoder for FILTER verse modeling."""
from __future__ import annotations
from dataclasses import dataclass
import os
import torch
from torch import nn

def configure_attention_backend_from_env() -> None:
    if os.environ.get('FILTER_FORCE_MATH_SDP', '0') != '1':
        return
    cuda_backends = getattr(torch.backends, 'cuda', None)
    if cuda_backends is None:
        return
    for (name, enabled) in (('enable_flash_sdp', False), ('enable_mem_efficient_sdp', False), ('enable_math_sdp', True), ('enable_cudnn_sdp', False)):
        setter = getattr(cuda_backends, name, None)
        if setter is not None:
            setter(enabled)
configure_attention_backend_from_env()

def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = hidden_states * attention_mask.unsqueeze(-1)
    lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return masked.sum(dim=1) / lengths

@dataclass
class ScratchEncoderConfig:
    vocab_size: int
    max_position_embeddings: int
    num_layers: int
    hidden_size: int
    num_heads: int
    ffn_dim: int
    dropout: float
    pad_token_id: int

class ScratchTransformerEncoder(nn.Module):

    def __init__(self, config: ScratchEncoderConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(num_embeddings=config.max_position_embeddings, embedding_dim=config.hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_heads, dim_feedforward=config.ffn_dim, dropout=config.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=config.num_layers)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.mlm_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_activation = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(config.hidden_size)
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.mlm_decoder.weight = self.token_embeddings.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(self.token_embeddings, nn.Embedding) and self.config.pad_token_id is not None:
            with torch.no_grad():
                self.token_embeddings.weight[self.config.pad_token_id].fill_(0.0)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        (_, seq_len) = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.embedding_layer_norm(hidden_states)
        return self.embedding_dropout(hidden_states)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed(input_ids)
        key_padding_mask = attention_mask == 0
        hidden_states = self.encoder(hidden_states, src_key_padding_mask=key_padding_mask)
        return self.final_layer_norm(hidden_states)

    def pooled_embedding(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return masked_mean_pool(hidden_states, attention_mask)

    def mlm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.mlm_transform(hidden_states)
        hidden_states = self.mlm_activation(hidden_states)
        hidden_states = self.mlm_layer_norm(hidden_states)
        return self.mlm_decoder(hidden_states) + self.mlm_bias

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_pooled: bool=True) -> dict[str, torch.Tensor]:
        hidden_states = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        outputs = {'last_hidden_state': hidden_states, 'mlm_logits': self.mlm_logits(hidden_states)}
        if return_pooled:
            outputs['pooled_embedding'] = self.pooled_embedding(hidden_states, attention_mask)
        return outputs
