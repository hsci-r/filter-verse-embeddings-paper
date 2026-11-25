"""Token-level English teacher embeddings for word alignment."""
from __future__ import annotations
import os
from pathlib import Path
import time
import logging
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from filter_verse_embeddings.teachers.sentence import pick_teacher_device
log = logging.getLogger(__name__)

class TokenTeacher:
    """Frozen transformer teacher that returns one contextual embedding per word-table row."""

    def __init__(self, model_name: str, device: str='auto'):
        self.device = pick_teacher_device(device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def encode_word_rows(self, texts: list[str] | pd.Series, *, max_length: int=512, chunk_rows: int=128, dtype: np.dtype=np.float16) -> np.ndarray:
        """Encode English word translations row-by-row, then contextualize them in stitched chunks.

        Tokenizing each row independently gives us exact row token spans, so hidden states never
        straddle word-table rows. We then concatenate several rows into one bounded MPNet sequence
        to let the teacher produce contextual token representations. The returned array preserves
        the input row order; rows with empty text remain all-zero and are skipped by the dataset.
        """
        rows = ['' if text is None else str(text).strip() for text in texts]
        embeddings = np.zeros((len(rows), self.hidden_size), dtype=dtype)
        current_indices: list[int] = []
        current_ids: list[list[int]] = []
        current_length = 2

        def flush() -> None:
            nonlocal current_indices, current_ids, current_length
            if not current_indices:
                return
            if self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None:
                raise ValueError(f'Tokenizer for {self.model_name} must define CLS and SEP token ids.')
            input_ids = [self.tokenizer.cls_token_id]
            spans: list[tuple[int, int]] = []
            for ids in current_ids:
                start = len(input_ids)
                input_ids.extend(ids)
                spans.append((start, len(input_ids)))
            input_ids.append(self.tokenizer.sep_token_id)
            attention_mask = [1] * len(input_ids)
            with torch.no_grad():
                outputs = self.model(input_ids=torch.tensor([input_ids], dtype=torch.long, device=self.device), attention_mask=torch.tensor([attention_mask], dtype=torch.long, device=self.device))
            hidden = outputs.last_hidden_state[0].detach().cpu()
            for (row_index, (start, end)) in zip(current_indices, spans):
                embeddings[row_index] = hidden[start:end].mean(dim=0).numpy().astype(dtype, copy=False)
            current_indices = []
            current_ids = []
            current_length = 2
        for (row_index, text) in enumerate(rows):
            if not text:
                continue
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            token_ids = token_ids[:max(1, max_length - 2)]
            would_overflow = current_length + len(token_ids) > max_length
            if would_overflow or len(current_indices) >= chunk_rows:
                flush()
            current_indices.append(row_index)
            current_ids.append(token_ids)
            current_length += len(token_ids)
        flush()
        return embeddings

def load_or_compute_token_teacher_embeddings(cache_path: str | Path | None, teacher: TokenTeacher, texts: list[str] | pd.Series, *, max_length: int=512, chunk_rows: int=128, dtype: np.dtype=np.float16, lock_timeout_seconds: int=21600, expected_rows: int | None=None) -> np.ndarray:
    """Load token-teacher embeddings from an npy cache, or compute them under a directory lock."""

    def load_valid_cache() -> np.ndarray | None:
        try:
            cached = np.load(cache_path, mmap_mode='r')
        except FileNotFoundError:
            return None
        if expected_rows is not None and int(cached.shape[0]) != int(expected_rows):
            log.warning('Ignoring token-teacher cache %s with %d rows; expected %d rows.', cache_path, int(cached.shape[0]), int(expected_rows))
            return None
        return cached
    if cache_path is not None:
        cache_path = Path(cache_path)
        cached = load_valid_cache()
        if cached is not None:
            return cached
        lock_path = cache_path.with_suffix(cache_path.suffix + '.lock')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        while True:
            try:
                lock_path.mkdir(parents=False)
                break
            except FileExistsError:
                cached = load_valid_cache()
                if cached is not None:
                    return cached
                if time.monotonic() - start > lock_timeout_seconds:
                    raise TimeoutError(f'Timed out waiting for token teacher cache lock: {lock_path}')
                time.sleep(30)
    else:
        lock_path = None
    try:
        if cache_path is not None:
            cached = load_valid_cache()
            if cached is not None:
                return cached
        embeddings = teacher.encode_word_rows(texts, max_length=max_length, chunk_rows=chunk_rows, dtype=dtype)
        if cache_path is not None:
            tmp_path = cache_path.with_name(f'.{cache_path.name}.{os.getpid()}.tmp.npy')
            np.save(tmp_path, embeddings)
            os.replace(tmp_path, cache_path)
            return np.load(cache_path, mmap_mode='r')
        return embeddings
    finally:
        if lock_path is not None:
            try:
                lock_path.rmdir()
            except FileNotFoundError:
                pass
