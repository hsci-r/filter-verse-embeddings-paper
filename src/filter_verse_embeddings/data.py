"""Data preparation helpers for scratch MLM training and export."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

@dataclass
class SentencePieceTokenizer:
    model_path: str

    def __post_init__(self) -> None:
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    @property
    def pad_id(self) -> int:
        return int(self.processor.pad_id())

    @property
    def bos_id(self) -> int:
        return int(self.processor.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self.processor.eos_id())

    @property
    def unk_id(self) -> int:
        return int(self.processor.unk_id())

    def piece_to_id(self, piece: str) -> int:
        return int(self.processor.piece_to_id(piece))

    def id_to_piece(self, token_id: int) -> str:
        return self.processor.id_to_piece(int(token_id))

    def encode_text(self, text: str) -> list[int]:
        return list(self.processor.encode('' if text is None else str(text), out_type=int))

@dataclass
class HFTokenizer:
    """HuggingFace tokenizer wrapper with the same interface as SentencePieceTokenizer.

    Maps BERT special tokens: [CLS]→bos, [SEP]→eos, [MASK]→mask, [PAD]→pad.
    Use verse_sep_token='[SEP]' in the training config when using this tokenizer.
    """
    hf_model_name: str

    def __post_init__(self) -> None:
        from transformers import AutoTokenizer
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)

    @property
    def vocab_size(self) -> int:
        return self.hf_tokenizer.vocab_size

    @property
    def pad_id(self) -> int:
        return self.hf_tokenizer.pad_token_id or 0

    @property
    def bos_id(self) -> int:
        return self.hf_tokenizer.cls_token_id

    @property
    def eos_id(self) -> int:
        return self.hf_tokenizer.sep_token_id

    @property
    def unk_id(self) -> int:
        return self.hf_tokenizer.unk_token_id or 0

    def piece_to_id(self, piece: str) -> int:
        return int(self.hf_tokenizer.convert_tokens_to_ids(piece))

    def id_to_piece(self, token_id: int) -> str:
        piece = self.hf_tokenizer.convert_ids_to_tokens(int(token_id)) or ''
        return piece[2:] if piece.startswith('##') else piece

    def encode_text(self, text: str) -> list[int]:
        return self.hf_tokenizer.encode('' if text is None else str(text), add_special_tokens=False)

def load_config(config_path: str | Path) -> dict:
    import yaml
    with open(config_path, 'r', encoding='utf-8') as fin:
        return yaml.safe_load(fin)

def save_yaml(payload: dict, path: str | Path) -> None:
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fout:
        yaml.safe_dump(payload, fout, sort_keys=False)

def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=True)

def load_split_table(verses_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(verses_path)

def split_frame(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return df[df['split'] == split_name].sort_values(['poem_id', 'verse_position', 'verse_id'], kind='stable').reset_index(drop=True)

def truncate_tokens(tokens: list[int], max_content_length: int) -> tuple[list[int], bool]:
    if len(tokens) <= max_content_length:
        return (tokens, False)
    return (tokens[:max_content_length], True)

def build_single_verse_sequences(split_df: pd.DataFrame, tokenizer: SentencePieceTokenizer, text_column: str, max_length: int, add_bos: bool, add_eos: bool) -> tuple[list[list[int]], dict]:
    sequences: list[list[int]] = []
    truncated_count = 0
    reserved = int(add_bos) + int(add_eos)
    max_content_length = max_length - reserved
    for text in split_df[text_column].fillna(''):
        tokens = tokenizer.encode_text(text)
        (tokens, was_truncated) = truncate_tokens(tokens, max_content_length)
        truncated_count += int(was_truncated)
        sequence: list[int] = []
        if add_bos:
            sequence.append(tokenizer.bos_id)
        sequence.extend(tokens)
        if add_eos:
            sequence.append(tokenizer.eos_id)
        sequences.append(sequence)
    return (sequences, {'truncated_verses': int(truncated_count)})

def build_packed_poem_sequences(split_df: pd.DataFrame, tokenizer: SentencePieceTokenizer, text_column: str, max_length: int, add_bos: bool, add_eos: bool, verse_sep_id: int) -> tuple[list[list[int]], dict]:
    sequences: list[list[int]] = []
    truncated_count = 0
    poems_packed = 0
    reserved = int(add_bos) + int(add_eos)
    max_content_length = max_length - reserved
    for (_, poem_df) in split_df.groupby('poem_id', sort=False):
        current_tokens: list[int] = []
        for text in poem_df[text_column].fillna(''):
            verse_tokens = tokenizer.encode_text(text)
            (verse_tokens, was_truncated) = truncate_tokens(verse_tokens, max_content_length)
            truncated_count += int(was_truncated)
            addition = verse_tokens if not current_tokens else [verse_sep_id] + verse_tokens
            if len(current_tokens) + len(addition) <= max_content_length:
                current_tokens.extend(addition)
                continue
            sequence: list[int] = []
            if add_bos:
                sequence.append(tokenizer.bos_id)
            sequence.extend(current_tokens)
            if add_eos:
                sequence.append(tokenizer.eos_id)
            sequences.append(sequence)
            poems_packed += 1
            current_tokens = verse_tokens.copy()
        if current_tokens:
            sequence = []
            if add_bos:
                sequence.append(tokenizer.bos_id)
            sequence.extend(current_tokens)
            if add_eos:
                sequence.append(tokenizer.eos_id)
            sequences.append(sequence)
            poems_packed += 1
    return (sequences, {'truncated_verses': int(truncated_count), 'packed_sequences': int(len(sequences)), 'poem_windows_emitted': int(poems_packed)})

def build_training_sequences(split_df: pd.DataFrame, tokenizer: SentencePieceTokenizer, text_column: str, sequence_mode: str, max_length: int, add_bos: bool, add_eos: bool, verse_sep_token: str) -> tuple[list[list[int]], dict]:
    if sequence_mode == 'single_verse':
        return build_single_verse_sequences(split_df=split_df, tokenizer=tokenizer, text_column=text_column, max_length=max_length, add_bos=add_bos, add_eos=add_eos)
    if sequence_mode == 'packed_poem_window':
        return build_packed_poem_sequences(split_df=split_df, tokenizer=tokenizer, text_column=text_column, max_length=max_length, add_bos=add_bos, add_eos=add_eos, verse_sep_id=tokenizer.piece_to_id(verse_sep_token))
    raise ValueError(f'Unsupported sequence_mode: {sequence_mode}')

class TokenSequenceDataset(Dataset):

    def __init__(self, sequences: list[list[int]], *, return_indices: bool=False):
        self.sequences = sequences
        self.return_indices = bool(return_indices)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[int] | dict[str, int | list[int]]:
        sequence = self.sequences[index]
        if not self.return_indices:
            return sequence
        return {'sequence': sequence, 'sequence_index': int(index)}

def collate_token_sequences(batch: list[list[int] | dict[str, int | list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    if not batch:
        return {'input_ids': torch.empty((0, 0), dtype=torch.long), 'attention_mask': torch.empty((0, 0), dtype=torch.long)}
    sequence_indices: list[int] | None = None
    if isinstance(batch[0], dict):
        sequence_indices = [int(item['sequence_index']) for item in batch]
        sequences = [list(item['sequence']) for item in batch]
    else:
        sequences = [list(sequence) for sequence in batch]
    max_len = max((len(sequence) for sequence in sequences))
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for (row_idx, sequence) in enumerate(sequences):
        seq_len = len(sequence)
        input_ids[row_idx, :seq_len] = torch.as_tensor(sequence, dtype=torch.long)
        attention_mask[row_idx, :seq_len] = 1
    payload = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if sequence_indices is not None:
        payload['sequence_indices'] = torch.tensor(sequence_indices, dtype=torch.long)
    return payload
