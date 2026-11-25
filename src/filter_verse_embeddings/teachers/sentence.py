"""Helpers for fixed English sentence-teacher embeddings."""
from __future__ import annotations
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def pick_teacher_device(device_name: str) -> str:
    if device_name != 'auto':
        return device_name
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'

class SentenceTeacher:

    def __init__(self, model_name: str, device: str='auto'):
        self.device = pick_teacher_device(device)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_texts(self, texts: list[str] | pd.Series, *, batch_size: int=128, normalize_embeddings: bool=True, show_progress_bar: bool=False) -> np.ndarray:
        encoded = self.model.encode(list(('' if text is None else str(text) for text in texts)), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=show_progress_bar, normalize_embeddings=normalize_embeddings)
        return np.asarray(encoded, dtype=np.float32)

def load_or_compute_teacher_embeddings(cache_path: str | Path | None, teacher: SentenceTeacher, texts: list[str] | pd.Series, *, batch_size: int=128, normalize_embeddings: bool=True, show_progress_bar: bool=False, lock_timeout_seconds: int=21600, expected_rows: int | None=None) -> np.ndarray:

    def load_valid_cache(path: Path) -> np.ndarray | None:
        cached = np.load(path).astype(np.float32, copy=False)
        if expected_rows is not None and int(cached.shape[0]) != int(expected_rows):
            return None
        return cached
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = load_valid_cache(cache_path)
            if cached is not None:
                return cached
        lock_path = cache_path.with_suffix(cache_path.suffix + '.lock')
        start = time.monotonic()
        while True:
            try:
                lock_path.mkdir(parents=True)
                break
            except FileExistsError:
                if cache_path.exists():
                    cached = load_valid_cache(cache_path)
                    if cached is not None:
                        return cached
                if time.monotonic() - start > lock_timeout_seconds:
                    raise TimeoutError(f'Timed out waiting for teacher cache lock: {lock_path}')
                time.sleep(30)
    else:
        lock_path = None
    try:
        if cache_path is not None and cache_path.exists():
            cached = load_valid_cache(cache_path)
            if cached is not None:
                return cached
        embeddings = teacher.encode_texts(texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings, show_progress_bar=show_progress_bar)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(f'.{cache_path.name}.{os.getpid()}.tmp.npy')
            np.save(tmp_path, embeddings)
            os.replace(tmp_path, cache_path)
        return embeddings
    finally:
        if lock_path is not None:
            try:
                lock_path.rmdir()
            except FileNotFoundError:
                pass
