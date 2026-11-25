"""
Compute contextualised English verse embeddings using ModernBERT.

Each poem is processed as one packed sequence so every verse's representation
is informed by its neighbours.  When a poem exceeds `--max-length` tokens the
poem is split into chunks; the last `--prefix-verses` verses of the preceding
chunk are prepended to the next chunk as a warm-up prefix (their embeddings are
NOT re-extracted — only new verses are extracted per chunk).

Output mirrors the existing teacher-cache format:
    one .npy file per split, shape (N_verses, hidden_dim), float32
    rows are aligned with split_frame() order (sorted by poem_id, verse_position, verse_id).

Usage example (CPU, for testing):
    python eval/export_modernbert_contextual_embeddings.py         --verses-path data/splits/verses_with_splits.parquet         --out-dir artifacts/outputs/teacher_cache/modernbert_base_contextual         --splits dev test         --device cpu

Usage example (GPU via sbatch):
    python eval/export_modernbert_contextual_embeddings.py         --verses-path data/splits/verses_with_splits.parquet         --out-dir artifacts/outputs/teacher_cache/modernbert_base_contextual         --device cuda
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
DEFAULT_MODEL = 'answerdotai/ModernBERT-base'
DEFAULT_VERSES_PATH = 'data/splits/verses_with_splits.parquet'
DEFAULT_OUT_DIR = 'artifacts/outputs/teacher_cache/modernbert_base_contextual'
DEFAULT_SPLITS = ['train', 'dev', 'test']
DEFAULT_MAX_LENGTH = 8192
DEFAULT_PREFIX_VERSES = 5
DEFAULT_TEXT_COLUMN = 'english_verse'

def masked_mean_pool(hidden_states: torch.Tensor, start: int, end: int) -> np.ndarray:
    """Mean-pool token hidden states in [start, end) into a single vector."""
    span = hidden_states[start:end]
    if span.shape[0] == 0:
        return np.zeros(hidden_states.shape[-1], dtype=np.float32)
    return span.mean(dim=0).float().cpu().numpy()

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (vectors / norms).astype(np.float32)

def embed_poem(verse_texts: list[str], tokenizer, model: torch.nn.Module, device: torch.device, max_length: int, prefix_verses: int) -> list[np.ndarray]:
    """
    Return one embedding vector per verse, in input order.

    Strategy
    --------
    1.  Tokenize every verse individually (no special tokens) to know token lengths.
    2.  Greedily pack verses into chunks that fit within max_length, reserving
        space for [CLS] and a trailing [SEP].
    3.  At each chunk boundary, prepend the last `prefix_verses` verses of the
        previous chunk as a warm-up prefix so boundary verses receive context.
    4.  Run a single forward pass per chunk; mean-pool each non-prefix verse span.
    """
    cls_id: int = tokenizer.cls_token_id
    sep_id: int = tokenizer.sep_token_id
    verse_token_ids: list[list[int]] = [tokenizer.encode(text or '', add_special_tokens=False) for text in verse_texts]
    n_verses = len(verse_token_ids)
    embeddings: list[np.ndarray | None] = [None] * n_verses
    usable = max_length - 2
    chunks: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for (vi, tids) in enumerate(verse_token_ids):
        cost = len(tids) + 1
        if current and current_tokens + cost > usable:
            chunks.append(current)
            current = current[-prefix_verses:] if prefix_verses > 0 else []
            current_tokens = sum((len(verse_token_ids[j]) + 1 for j in current))
        current.append(vi)
        current_tokens += cost
    if current:
        chunks.append(current)
    for (chunk_idx, verse_indices) in enumerate(chunks):
        if chunk_idx == 0:
            prefix_count = 0
        else:
            prefix_count = min(prefix_verses, len(verse_indices) - 1)
        token_ids: list[int] = [cls_id]
        spans: list[tuple[int, int]] = []
        for vi in verse_indices:
            tids = verse_token_ids[vi]
            start = len(token_ids)
            token_ids.extend(tids)
            end = len(token_ids)
            spans.append((start, end))
            token_ids.append(sep_id)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state[0]
        for (local_idx, (vi, (start, end))) in enumerate(zip(verse_indices, spans)):
            if local_idx < prefix_count:
                continue
            end = min(end, hidden.shape[0])
            start = min(start, end)
            embeddings[vi] = masked_mean_pool(hidden, start, end)
    hidden_dim = model.config.hidden_size
    for i in range(n_verses):
        if embeddings[i] is None:
            embeddings[i] = np.zeros(hidden_dim, dtype=np.float32)
    return embeddings

def export_split(split_df: pd.DataFrame, tokenizer, model: torch.nn.Module, device: torch.device, out_path: Path, max_length: int, prefix_verses: int, text_column: str) -> None:
    n_verses = len(split_df)
    hidden_dim = model.config.hidden_size
    result = np.zeros((n_verses, hidden_dim), dtype=np.float32)
    poems = list(split_df.groupby('poem_id', sort=False))
    t0 = time.time()
    for (poem_idx, (poem_id, poem_df)) in enumerate(tqdm(poems, desc=f'poems', unit='poem')):
        verse_texts = poem_df[text_column].fillna('').tolist()
        row_indices = poem_df.index.tolist()
        embeddings = embed_poem(verse_texts, tokenizer=tokenizer, model=model, device=device, max_length=max_length, prefix_verses=prefix_verses)
        for (local_i, (row_i, emb)) in enumerate(zip(row_indices, embeddings)):
            result[row_i] = emb
    result = l2_normalize(result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f'.{out_path.name}.{os.getpid()}.tmp.npy')
    np.save(tmp, result)
    os.replace(tmp, out_path)
    elapsed = time.time() - t0
    print(f'  saved {out_path}  shape={result.shape}  ({elapsed:.0f}s)')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=DEFAULT_MODEL, help='HuggingFace model id or local path')
    p.add_argument('--verses-path', default=DEFAULT_VERSES_PATH)
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--splits', nargs='+', default=DEFAULT_SPLITS)
    p.add_argument('--device', default='auto')
    p.add_argument('--max-length', type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument('--prefix-verses', type=int, default=DEFAULT_PREFIX_VERSES, help='Number of verses from previous chunk to use as warm-up context (default 5)')
    p.add_argument('--text-column', default=DEFAULT_TEXT_COLUMN)
    p.add_argument('--skip-existing', action='store_true', help='Skip splits whose output .npy already exists')
    return p.parse_args()

def resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

def load_split_table(verses_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(verses_path)

def split_frame(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return df[df['split'] == split_name].sort_values(['poem_id', 'verse_position', 'verse_id'], kind='stable').reset_index(drop=True)

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f'device: {device}')
    print(f'model:  {args.model}')
    print(f'max_length: {args.max_length}  prefix_verses: {args.prefix_verses}')
    print('loading tokenizer …')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print('loading model …')
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    model.to(device)
    print('loading verse table …')
    split_table = load_split_table(args.verses_path)
    for split_name in args.splits:
        out_path = args.out_dir / f'{split_name}.npy'
        if args.skip_existing and out_path.exists():
            print(f'  {split_name}: skipping (exists)')
            continue
        split_df = split_frame(split_table, split_name)
        print(f"\n=== {split_name}: {len(split_df)} verses in {split_df['poem_id'].nunique()} poems ===")
        export_split(split_df, tokenizer=tokenizer, model=model, device=device, out_path=out_path, max_length=args.max_length, prefix_verses=args.prefix_verses, text_column=args.text_column)
    print('\nDone.')
if __name__ == '__main__':
    main()
