"""Export dense verse embeddings in the evaluator run format."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader
from filter_verse_embeddings.data import SentencePieceTokenizer, collate_token_sequences
DEFAULT_METADATA_COLUMNS = ['verse_id', 'poem_id', 'verse_position', 'split', 'source_collection', 'filter_text', 'english_verse']

def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return array / norms

def encode_text_column(model, tokenizer: SentencePieceTokenizer, texts: pd.Series, max_length: int, batch_size: int, device: torch.device, add_bos: bool, add_eos: bool) -> np.ndarray:
    sequences: list[list[int]] = []
    reserved = int(add_bos) + int(add_eos)
    max_content_length = max_length - reserved
    for text in texts.fillna(''):
        piece_ids = tokenizer.encode_text(text)[:max_content_length]
        sequence: list[int] = []
        if add_bos:
            sequence.append(tokenizer.bos_id)
        sequence.extend(piece_ids)
        if add_eos:
            sequence.append(tokenizer.eos_id)
        sequences.append(sequence)
    loader = DataLoader(sequences, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_token_sequences(batch, tokenizer.pad_id))
    vectors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_pooled=True)
            pooled = outputs['pooled_embedding'].detach().cpu().numpy().astype(np.float32)
            vectors.append(pooled)
    stacked = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return l2_normalize(stacked).astype(np.float32)

def save_dense_embeddings(output_path: Path, embeddings: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

def save_metadata(metadata_dir: Path, split_df: pd.DataFrame) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    split_df[DEFAULT_METADATA_COLUMNS].reset_index(drop=True).to_parquet(metadata_dir / f"{split_df['split'].iloc[0]}_metadata.parquet", index=False)

def export_split_artifacts(model, tokenizer: SentencePieceTokenizer, split_df: pd.DataFrame, split_name: str, run_dir: Path, max_length: int, batch_size: int, device: torch.device, add_bos: bool, add_eos: bool) -> dict[str, int]:
    embeddings_dir = run_dir / 'embeddings'
    metadata_dir = run_dir / 'metadata'
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    filter_embeddings = encode_text_column(model=model, tokenizer=tokenizer, texts=split_df['filter_text'], max_length=max_length, batch_size=batch_size, device=device, add_bos=add_bos, add_eos=add_eos)
    english_embeddings = encode_text_column(model=model, tokenizer=tokenizer, texts=split_df['english_verse'], max_length=max_length, batch_size=batch_size, device=device, add_bos=add_bos, add_eos=add_eos)
    save_dense_embeddings(embeddings_dir / f'{split_name}_filter.npy', filter_embeddings)
    save_dense_embeddings(embeddings_dir / f'{split_name}_english.npy', english_embeddings)
    save_metadata(metadata_dir, split_df.assign(split=split_name))
    return {'num_rows': int(len(split_df))}
