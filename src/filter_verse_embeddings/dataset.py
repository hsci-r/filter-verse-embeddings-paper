"""Packed poem-window dataset and word-alignment helpers for FILTER."""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from filter_verse_embeddings.data import SentencePieceTokenizer, collate_token_sequences

def normalize_alignment_text(value: object) -> str:
    text = '' if value is None else str(value).casefold()
    return re.sub('[^\\w]+', '', text, flags=re.UNICODE)

def normalized_token_pieces(tokenizer: SentencePieceTokenizer, token_ids: list[int]) -> list[str]:
    return [normalize_alignment_text(tokenizer.id_to_piece(int(token_id)).replace('▁', '')) for token_id in token_ids]

def unique_tokenized_candidates(tokenizer: SentencePieceTokenizer, values: list[object]) -> list[list[str]]:
    candidates: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for value in values:
        text = normalize_alignment_text(value)
        if not text:
            continue
        token_ids = tokenizer.encode_text(text)
        normalized_tokens = [piece for piece in normalized_token_pieces(tokenizer, token_ids) if piece]
        key = tuple(normalized_tokens)
        if normalized_tokens and key not in seen:
            candidates.append(normalized_tokens)
            seen.add(key)
    return candidates

def find_subsequence(sequence: list[str], pattern: list[str], start: int=0) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    last_start = len(sequence) - len(pattern)
    for index in range(max(0, start), last_start + 1):
        if sequence[index:index + len(pattern)] == pattern:
            return index
    return -1

def build_word_rows_by_poem(word_df: pd.DataFrame, tokenizer: SentencePieceTokenizer) -> dict[str, list[dict[str, object]]]:
    rows_by_poem: dict[str, list[dict[str, object]]] = {}
    for (teacher_index, row) in word_df.reset_index(drop=True).iterrows():
        candidates = unique_tokenized_candidates(tokenizer, [row.get('original_form', '')])
        if not candidates:
            continue
        has_teacher_target = bool(normalize_alignment_text(row.get('english_translation', '')))
        poem_id = str(row['poem_id'])
        rows_by_poem.setdefault(poem_id, []).append({'teacher_index': int(teacher_index), 'has_teacher_target': has_teacher_target, 'candidate_token_ids': candidates, 'word_id': row.get('word_id', None), 'word_position': row.get('word_position', None)})
    return rows_by_poem

def token_teacher_cache_path(token_cache_dir: Path, split_name: str, word_df: pd.DataFrame) -> Path:
    """Build a cache filename that changes when the word-row subset/order changes."""
    identity_columns = [column for column in ('poem_id', 'word_id', 'word_position', 'original_form', 'english_translation') if column in word_df.columns]
    hasher = hashlib.sha1()
    hasher.update(str(len(word_df)).encode('utf-8'))
    if identity_columns and len(word_df) > 0:
        row_hashes = pd.util.hash_pandas_object(word_df[identity_columns].astype(str), index=False).to_numpy(dtype=np.uint64, copy=False)
        hasher.update(row_hashes.tobytes())
    digest = hasher.hexdigest()[:16]
    return token_cache_dir / f'{split_name}_{len(word_df)}_{digest}.npy'

def align_word_rows_to_verse_tokens(*, tokenizer: SentencePieceTokenizer, verse_tokens: list[int], poem_word_rows: list[dict[str, object]], word_cursor: int, token_offset: int, word_teacher_embeddings: np.ndarray | None, max_skip_rows: int=8) -> tuple[list[tuple[int, int]], list[np.ndarray], int, int]:
    if word_teacher_embeddings is None or not poem_word_rows or (not verse_tokens):
        return ([], [], word_cursor, 0)
    spans: list[tuple[int, int]] = []
    teacher_rows: list[np.ndarray] = []
    skipped_rows = 0
    token_cursor = 0
    normalized_verse_tokens = normalized_token_pieces(tokenizer, verse_tokens)
    while word_cursor < len(poem_word_rows) and token_cursor < len(verse_tokens):
        best_match: tuple[int, int, int, int] | None = None
        search_stop = min(len(poem_word_rows), word_cursor + max_skip_rows + 1)
        for row_index in range(word_cursor, search_stop):
            row = poem_word_rows[row_index]
            for candidate in row['candidate_token_ids']:
                start = find_subsequence(normalized_verse_tokens, candidate, start=token_cursor)
                if start < 0:
                    continue
                end = start + len(candidate)
                candidate_match = (start, end, row_index, len(candidate))
                if best_match is None or candidate_match < best_match:
                    best_match = candidate_match
        if best_match is None:
            break
        (start, end, matched_row_index, _) = best_match
        skipped_rows += max(0, matched_row_index - word_cursor)
        matched_row = poem_word_rows[matched_row_index]
        if bool(matched_row.get('has_teacher_target', True)):
            teacher_index = int(matched_row['teacher_index'])
            spans.append((token_offset + start, token_offset + end))
            teacher_rows.append(np.asarray(word_teacher_embeddings[teacher_index], dtype=np.float32))
        word_cursor = matched_row_index + 1
        token_cursor = end
    return (spans, teacher_rows, word_cursor, skipped_rows)

class PackedContrastiveWindowDataset(Dataset):

    def __init__(self, split_df, tokenizer: SentencePieceTokenizer, teacher_embeddings: np.ndarray, word_rows_by_poem: dict[str, list[dict[str, object]]] | None=None, word_teacher_embeddings: np.ndarray | None=None, *, max_length: int, add_bos: bool, add_eos: bool, verse_sep_token: str, filter_text_column: str):
        self.examples: list[dict[str, object]] = []
        self.alignment_stats = {'word_rows_available': 0, 'word_rows_matched': 0, 'word_rows_skipped': 0, 'windows_with_words': 0}
        reserved = int(add_bos) + int(add_eos)
        max_content_length = max_length - reserved
        bos_offset = int(add_bos)
        verse_sep_id = tokenizer.piece_to_id(verse_sep_token)
        teacher_embeddings = teacher_embeddings.astype(np.float32, copy=False)
        word_rows_by_poem = word_rows_by_poem or {}
        word_teacher_dim = int(word_teacher_embeddings.shape[1]) if word_teacher_embeddings is not None else 0
        for (_, poem_df) in split_df.groupby('poem_id', sort=False):
            poem_id = str(poem_df['poem_id'].iloc[0])
            current_tokens: list[int] = []
            current_spans: list[tuple[int, int]] = []
            current_teacher_rows: list[np.ndarray] = []
            current_word_spans: list[tuple[int, int]] = []
            current_word_teacher_rows: list[np.ndarray] = []
            window_index = 0
            poem_word_rows = word_rows_by_poem.get(poem_id, [])
            self.alignment_stats['word_rows_available'] += len(poem_word_rows)
            word_cursor = 0
            for row in poem_df.itertuples(index=True):
                verse_tokens = tokenizer.encode_text(getattr(row, filter_text_column) or '')
                verse_tokens = verse_tokens[:max_content_length]
                addition_length = len(verse_tokens) if not current_tokens else len(verse_tokens) + 1
                if current_tokens and len(current_tokens) + addition_length > max_content_length:
                    self.examples.append(finalize_window(poem_id=poem_id, tokens=current_tokens, spans=current_spans, teacher_rows=current_teacher_rows, word_spans=current_word_spans, word_teacher_rows=current_word_teacher_rows, word_teacher_dim=word_teacher_dim, add_bos=add_bos, add_eos=add_eos, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, window_index=window_index))
                    if current_word_spans:
                        self.alignment_stats['windows_with_words'] += 1
                    window_index += 1
                    current_tokens = []
                    current_spans = []
                    current_teacher_rows = []
                    current_word_spans = []
                    current_word_teacher_rows = []
                if current_tokens:
                    current_tokens.append(verse_sep_id)
                span_start = bos_offset + len(current_tokens)
                (verse_word_spans, verse_word_teacher_rows, word_cursor, skipped_count) = align_word_rows_to_verse_tokens(tokenizer=tokenizer, verse_tokens=verse_tokens, poem_word_rows=poem_word_rows, word_cursor=word_cursor, token_offset=span_start, word_teacher_embeddings=word_teacher_embeddings)
                self.alignment_stats['word_rows_skipped'] += skipped_count
                current_tokens.extend(verse_tokens)
                span_end = bos_offset + len(current_tokens)
                current_spans.append((span_start, span_end))
                current_teacher_rows.append(teacher_embeddings[int(row.Index)])
                current_word_spans.extend(verse_word_spans)
                current_word_teacher_rows.extend(verse_word_teacher_rows)
                self.alignment_stats['word_rows_matched'] += len(verse_word_spans)
            if current_tokens:
                self.examples.append(finalize_window(poem_id=poem_id, tokens=current_tokens, spans=current_spans, teacher_rows=current_teacher_rows, word_spans=current_word_spans, word_teacher_rows=current_word_teacher_rows, word_teacher_dim=word_teacher_dim, add_bos=add_bos, add_eos=add_eos, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, window_index=window_index))
            if current_word_spans:
                self.alignment_stats['windows_with_words'] += 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.examples[index]

def finalize_window(*, poem_id: str, tokens: list[int], spans: list[tuple[int, int]], teacher_rows: list[np.ndarray], word_spans: list[tuple[int, int]], word_teacher_rows: list[np.ndarray], word_teacher_dim: int, add_bos: bool, add_eos: bool, bos_id: int, eos_id: int, window_index: int) -> dict[str, object]:
    sequence: list[int] = []
    if add_bos:
        sequence.append(bos_id)
    sequence.extend(tokens)
    if add_eos:
        sequence.append(eos_id)
    if word_teacher_rows:
        word_teacher_array = np.stack(word_teacher_rows, axis=0).astype(np.float32, copy=False)
    else:
        word_teacher_array = np.zeros((0, word_teacher_dim), dtype=np.float32)
    return {'sequence': sequence, 'poem_id': poem_id, 'window_index': int(window_index), 'window_id': f'{poem_id}::window_{int(window_index):04d}', 'verse_spans': spans, 'teacher_embeddings': np.stack(teacher_rows, axis=0).astype(np.float32, copy=False), 'word_spans': list(word_spans), 'word_teacher_embeddings': word_teacher_array}

def collate_packed_contrastive_batch(batch: list[dict[str, object]], pad_id: int) -> dict[str, object]:
    padded = collate_token_sequences([item['sequence'] for item in batch], pad_id=pad_id)
    padded['poem_ids'] = [str(item['poem_id']) for item in batch]
    padded['window_indices'] = torch.as_tensor([int(item['window_index']) for item in batch], dtype=torch.long)
    padded['window_ids'] = [str(item['window_id']) for item in batch]
    max_verses = max((len(item['verse_spans']) for item in batch))
    teacher_dim = int(batch[0]['teacher_embeddings'].shape[1])
    teacher_embeddings = np.zeros((len(batch), max_verses, teacher_dim), dtype=np.float32)
    verse_spans = torch.zeros((len(batch), max_verses, 2), dtype=torch.long)
    verse_mask = torch.zeros((len(batch), max_verses), dtype=torch.bool)
    for (row_idx, item) in enumerate(batch):
        num_verses = len(item['verse_spans'])
        verse_mask[row_idx, :num_verses] = True
        verse_spans[row_idx, :num_verses] = torch.as_tensor(item['verse_spans'], dtype=torch.long)
        teacher_embeddings[row_idx, :num_verses] = item['teacher_embeddings']
    padded['teacher_embeddings'] = torch.as_tensor(teacher_embeddings, dtype=torch.float32)
    padded['verse_spans'] = verse_spans
    padded['verse_mask'] = verse_mask
    max_words = max((len(item.get('word_spans', [])) for item in batch))
    word_dim = max((int(item.get('word_teacher_embeddings', np.zeros((0, 0))).shape[1]) for item in batch))
    word_teacher_embeddings = np.zeros((len(batch), max_words, word_dim), dtype=np.float32)
    word_spans = torch.zeros((len(batch), max_words, 2), dtype=torch.long)
    word_mask = torch.zeros((len(batch), max_words), dtype=torch.bool)
    for (row_idx, item) in enumerate(batch):
        item_word_spans = item.get('word_spans', [])
        num_words = len(item_word_spans)
        if num_words == 0:
            continue
        word_mask[row_idx, :num_words] = True
        word_spans[row_idx, :num_words] = torch.as_tensor(item_word_spans, dtype=torch.long)
        word_teacher_embeddings[row_idx, :num_words] = item['word_teacher_embeddings']
    padded['word_teacher_embeddings'] = torch.as_tensor(word_teacher_embeddings, dtype=torch.float32)
    padded['word_spans'] = word_spans
    padded['word_mask'] = word_mask
    return padded
