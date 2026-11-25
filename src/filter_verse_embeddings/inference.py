"""Load the published encoder and compute projected, normalized verse vectors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import sentencepiece as spm
import torch
from torch import nn
from torch.nn import functional as F

from .model import ScratchEncoderConfig, ScratchTransformerEncoder


class FilterVerseEncoder(nn.Module):
    """The paper encoder with its learned verse projection and exact tokenizer."""

    def __init__(self, config: dict, tokenizer_path: str | Path):
        super().__init__()
        if config.get('format_version') != 1:
            raise ValueError('Unsupported model package format')
        self.release_config = config
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.encoder = ScratchTransformerEncoder(ScratchEncoderConfig(**config['encoder']))
        self.projection = nn.Linear(self.encoder.config.hidden_size, config['projection_dim'], bias=False)
        if self.tokenizer.vocab_size() != self.encoder.config.vocab_size:
            raise ValueError('Tokenizer vocabulary does not match the encoder')
        for name, token_id in config['special_token_ids'].items():
            if self.tokenizer.piece_to_id(name) != token_id:
                raise ValueError(f'Tokenizer ID mismatch for {name}')
        self.eval()

    @classmethod
    def from_pretrained(cls, model_id_or_path: str | Path, *, revision: str | None = None,
                        device: str = 'cpu', local_files_only: bool = False):
        root = Path(model_id_or_path)
        if not root.is_dir():
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(str(model_id_or_path), revision=revision,
                                         local_files_only=local_files_only,
                                         allow_patterns=['config.json', 'model.safetensors', 'tokenizer.model']))
        from safetensors.torch import load_file
        config = json.loads((root / 'config.json').read_text())
        model = cls(config, root / 'tokenizer.model')
        state = load_file(str(root / 'model.safetensors'), device='cpu')
        # The MLM decoder is tied to the input embeddings, so it is stored once.
        encoder_state = {k.removeprefix('encoder.'): v for k, v in state.items() if k.startswith('encoder.')}
        result = model.encoder.load_state_dict(encoder_state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != {'mlm_decoder.weight'}:
            raise ValueError(f'Invalid encoder state: {result}')
        model.projection.load_state_dict({k.removeprefix('projection.'): v for k, v in state.items()
                                          if k.startswith('projection.')}, strict=True)
        return model.to(device).eval()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def max_length(self):
        return self.encoder.config.max_position_embeddings

    def _content(self, verse: str):
        reserved = int(self.release_config['add_bos']) + int(self.release_config['add_eos'])
        return self.tokenizer.encode(str(verse))[:self.max_length - reserved]

    def _sequence(self, content):
        return (([self.tokenizer.bos_id()] if self.release_config['add_bos'] else [])
                + content + ([self.tokenizer.eos_id()] if self.release_config['add_eos'] else []))

    def _batch(self, sequences):
        max_len = max(map(len, sequences))
        ids = torch.full((len(sequences), max_len), self.tokenizer.pad_id(), dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, tokens in enumerate(sequences):
            ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=self.device)
            mask[i, :len(tokens)] = 1
        return ids, mask

    @torch.inference_mode()
    def encode_verses(self, verses: Sequence[str], *, batch_size: int = 256) -> np.ndarray:
        """Encode each verse independently; pool BOS/EOS and content, ignoring padding."""
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        output = []
        self.eval()
        for start in range(0, len(verses), batch_size):
            sequences = [self._sequence(self._content(v)) for v in verses[start:start + batch_size]]
            ids, mask = self._batch(sequences)
            hidden = self.encoder.encode(ids, mask)
            pooled = self.encoder.pooled_embedding(hidden, mask)
            output.append(F.normalize(self.projection(pooled), dim=1).cpu().numpy())
        return np.concatenate(output).astype(np.float32, copy=False) if output else np.empty((0, self.projection.out_features), dtype=np.float32)

    def contextual_windows(self, verses: Sequence[str]):
        """Yield non-overlapping, verse-aligned windows without crossing poem boundaries."""
        current, spans, rows = [], [], []
        offset = int(self.release_config['add_bos'])
        reserved = offset + int(self.release_config['add_eos'])
        capacity = self.max_length - reserved
        sep = self.tokenizer.piece_to_id(self.release_config['verse_sep_token'])
        for row, verse in enumerate(verses):
            tokens = self._content(verse)
            if rows and len(current) + 1 + len(tokens) > capacity:
                yield self._sequence(current), spans, rows
                current, spans, rows = [], [], []
            if rows:
                current.append(sep)
            start = len(current) + offset
            current.extend(tokens)
            spans.append((start, len(current) + offset))
            rows.append(row)
        if rows:
            yield self._sequence(current), spans, rows

    @torch.inference_mode()
    def encode_poem(self, poem: str | Sequence[str]) -> np.ndarray:
        """Return one contextual verse vector per input line, in the original order."""
        verses = poem.splitlines() if isinstance(poem, str) else list(poem)
        result = np.empty((len(verses), self.projection.out_features), dtype=np.float32)
        self.eval()
        for sequence, spans, rows in self.contextual_windows(verses):
            ids, mask = self._batch([sequence])
            hidden = self.encoder.encode(ids, mask)[0]
            pooled = torch.stack([hidden[start:end].sum(dim=0) / max(1, end - start) for start, end in spans])
            result[rows] = F.normalize(self.projection(pooled), dim=1).cpu().numpy()
        return result
