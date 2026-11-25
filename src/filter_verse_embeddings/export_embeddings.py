"""Export whole-corpus verse embeddings from the selected Paper best model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Iterable
import numpy as np
import pandas as pd
import torch
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.model import ScratchEncoderConfig, ScratchTransformerEncoder
from filter_verse_embeddings.data import HFTokenizer, SentencePieceTokenizer, collate_token_sequences, load_config
from filter_verse_embeddings.export_utils import l2_normalize
DEFAULT_CONFIG = Path('configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml')
DEFAULT_CHECKPOINT = Path('artifacts/checkpoints/verse_balanced_v0p45_w0p45_mlm0p10/epoch_200.pt')
DEFAULT_VERSES = Path('artifacts/data/paper_verses.parquet')
DEFAULT_OUTPUT = Path('artifacts/embeddings')
DEFAULT_METADATA_COLUMNS = ['poem_id', 'verse_id', 'verse_position', 'source_collection', 'split', 'filter_text', 'english_verse', 'source_verse_type', 'paper_has_source_pos', 'paper_position_source', 'paper_translation_row_number']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--verses-path', type=Path, default=DEFAULT_VERSES)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--modes', nargs='+', choices=('isolated', 'contextual'), default=['isolated', 'contextual'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size-isolated', type=int, default=512)
    parser.add_argument('--batch-size-contextual', type=int, default=256)
    parser.add_argument('--dtype', choices=('float32', 'float16'), default='float32')
    parser.add_argument('--limit-rows', type=int, default=None, help='Debug only: export the first N sorted rows.')
    parser.add_argument('--no-normalize', action='store_true', help='Write raw projected vectors instead of unit vectors.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path

def resolve_device(device_name: str) -> torch.device:
    if device_name != 'auto':
        return torch.device(device_name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_scratch_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = load_config(config_path)
    if config['tokenizer'].get('hf_model_name'):
        tokenizer = HFTokenizer(config['tokenizer']['hf_model_name'])
    else:
        tokenizer = SentencePieceTokenizer(config['tokenizer']['model_path'])
    model_config = ScratchEncoderConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=int(config['tokenizer']['max_length']), num_layers=int(config['model']['num_layers']), hidden_size=int(config['model']['hidden_size']), num_heads=int(config['model']['num_heads']), ffn_dim=int(config['model']['ffn_dim']), dropout=float(config['model']['dropout']), pad_token_id=tokenizer.pad_id)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = ScratchTransformerEncoder(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    projection_dim = int(checkpoint.get('teacher_embedding_dim', model_config.hidden_size))
    projection = torch.nn.Linear(model_config.hidden_size, projection_dim, bias=False)
    projection.load_state_dict(checkpoint['projection_state_dict'])
    projection.to(device)
    projection.eval()
    return (config, tokenizer, model, projection, projection_dim)

def load_verses(path: Path, text_column: str, limit_rows: int | None) -> pd.DataFrame:
    import pyarrow.parquet as pq
    available = pq.ParquetFile(path).schema.names
    needed = [column for column in DEFAULT_METADATA_COLUMNS if column in available]
    if text_column not in needed:
        needed.append(text_column)
    frame = pd.read_parquet(path, columns=needed)
    frame['poem_id'] = frame['poem_id'].astype(str)
    if 'verse_id' in frame.columns:
        frame['verse_id'] = frame['verse_id'].astype(str)
    frame[text_column] = frame[text_column].fillna('').astype(str)
    sort_columns = [column for column in ('poem_id', 'verse_position', 'verse_id') if column in frame.columns]
    frame = frame.sort_values(sort_columns, kind='stable').reset_index(drop=True)
    if limit_rows is not None:
        frame = frame.head(int(limit_rows)).reset_index(drop=True)
    frame.insert(0, 'embedding_row', np.arange(len(frame), dtype=np.int64))
    return frame

def save_metadata(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / 'metadata.parquet', index=False)

def output_path(output_dir: Path, mode: str) -> Path:
    return output_dir / f'{mode}_embeddings.npy'

def temporary_output_path(output_dir: Path, mode: str) -> Path:
    return output_dir / f'{mode}_embeddings.tmp.npy'

def prepare_output(path: Path, tmp_path: Path, overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        raise FileExistsError(f'Refusing to overwrite existing file: {path}')
    if tmp_path.exists():
        if not overwrite:
            raise FileExistsError(f'Temporary output already exists from a previous run: {tmp_path}')
        tmp_path.unlink()

def finalize_output(path: Path, tmp_path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f'Refusing to overwrite existing file: {path}')
        path.unlink()
    tmp_path.replace(path)

def maybe_normalize(vectors: np.ndarray, normalize: bool) -> np.ndarray:
    if not normalize:
        return vectors.astype(np.float32, copy=False)
    return l2_normalize(vectors.astype(np.float32, copy=False)).astype(np.float32, copy=False)

def encode_batch_sequences(*, model: ScratchTransformerEncoder, projection: torch.nn.Linear, tokenizer, sequences: list[list[int]], device: torch.device, normalize: bool) -> np.ndarray:
    batch = collate_token_sequences(sequences, tokenizer.pad_id)
    with torch.no_grad():
        outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), return_pooled=True)
        vectors = projection(outputs['pooled_embedding']).detach().cpu().numpy().astype(np.float32)
    return maybe_normalize(vectors, normalize=normalize)

def export_isolated(*, frame: pd.DataFrame, config: dict, tokenizer, model: ScratchTransformerEncoder, projection: torch.nn.Linear, projection_dim: int, output_dir: Path, batch_size: int, device: torch.device, dtype: np.dtype, normalize: bool, overwrite: bool) -> dict[str, int | str]:
    path = output_path(output_dir, 'isolated')
    tmp_path = temporary_output_path(output_dir, 'isolated')
    prepare_output(path, tmp_path, overwrite=overwrite)
    matrix = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=dtype, shape=(len(frame), projection_dim))
    add_bos = bool(config['tokenizer']['add_bos'])
    add_eos = bool(config['tokenizer']['add_eos'])
    reserved = int(add_bos) + int(add_eos)
    max_content = int(config['tokenizer']['max_length']) - reserved
    text_column = str(config['data']['filter_text_column'])
    truncated = 0
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        sequences: list[list[int]] = []
        for text in frame[text_column].iloc[start:stop]:
            token_ids = tokenizer.encode_text(text)
            if len(token_ids) > max_content:
                truncated += 1
                token_ids = token_ids[:max_content]
            sequence: list[int] = []
            if add_bos:
                sequence.append(tokenizer.bos_id)
            sequence.extend(token_ids)
            if add_eos:
                sequence.append(tokenizer.eos_id)
            sequences.append(sequence)
        matrix[start:stop] = encode_batch_sequences(model=model, projection=projection, tokenizer=tokenizer, sequences=sequences, device=device, normalize=normalize).astype(dtype, copy=False)
        if start and start // batch_size % 200 == 0:
            print(f'isolated: wrote {stop:,}/{len(frame):,}', flush=True)
    matrix.flush()
    del matrix
    finalize_output(path, tmp_path, overwrite=overwrite)
    return {'path': str(path), 'rows': int(len(frame)), 'dim': int(projection_dim), 'truncated_verses': int(truncated)}

def make_sequence(tokens: list[int], tokenizer, add_bos: bool, add_eos: bool) -> list[int]:
    sequence: list[int] = []
    if add_bos:
        sequence.append(tokenizer.bos_id)
    sequence.extend(tokens)
    if add_eos:
        sequence.append(tokenizer.eos_id)
    return sequence

def iter_contextual_windows(*, frame: pd.DataFrame, config: dict, tokenizer) -> Iterable[dict[str, object]]:
    text_column = str(config['data']['filter_text_column'])
    add_bos = bool(config['tokenizer']['add_bos'])
    add_eos = bool(config['tokenizer']['add_eos'])
    bos_offset = int(add_bos)
    reserved = int(add_bos) + int(add_eos)
    max_content = int(config['tokenizer']['max_length']) - reserved
    verse_sep_id = tokenizer.piece_to_id(str(config['tokenizer']['verse_sep_token']))
    for (poem_id, poem_df) in frame.groupby('poem_id', sort=False):
        current_tokens: list[int] = []
        current_spans: list[tuple[int, int]] = []
        current_rows: list[int] = []
        window_index = 0

        def emit_window() -> dict[str, object]:
            return {'poem_id': str(poem_id), 'window_index': int(window_index), 'sequence': make_sequence(current_tokens, tokenizer, add_bos=add_bos, add_eos=add_eos), 'verse_spans': list(current_spans), 'row_indices': list(current_rows)}
        for row in poem_df.itertuples(index=False):
            text = getattr(row, text_column)
            row_index = int(getattr(row, 'embedding_row'))
            verse_tokens = tokenizer.encode_text(text)
            if len(verse_tokens) > max_content:
                verse_tokens = verse_tokens[:max_content]
            addition_length = len(verse_tokens) + (1 if current_rows else 0)
            if current_rows and len(current_tokens) + addition_length > max_content:
                yield emit_window()
                window_index += 1
                current_tokens = []
                current_spans = []
                current_rows = []
            if current_rows:
                current_tokens.append(verse_sep_id)
            span_start = bos_offset + len(current_tokens)
            current_tokens.extend(verse_tokens)
            span_end = bos_offset + len(current_tokens)
            current_spans.append((span_start, span_end))
            current_rows.append(row_index)
        if current_rows:
            yield emit_window()

def collate_contextual_windows(batch: list[dict[str, object]], pad_id: int) -> dict[str, object]:
    sequences = [list(item['sequence']) for item in batch]
    padded = collate_token_sequences(sequences, pad_id)
    max_verses = max((len(item['verse_spans']) for item in batch))
    verse_spans = torch.zeros((len(batch), max_verses, 2), dtype=torch.long)
    verse_mask = torch.zeros((len(batch), max_verses), dtype=torch.bool)
    row_indices: list[int] = []
    for (batch_index, item) in enumerate(batch):
        spans = list(item['verse_spans'])
        rows = [int(row) for row in item['row_indices']]
        verse_spans[batch_index, :len(spans)] = torch.as_tensor(spans, dtype=torch.long)
        verse_mask[batch_index, :len(spans)] = True
        row_indices.extend(rows)
    padded['verse_spans'] = verse_spans
    padded['verse_mask'] = verse_mask
    padded['row_indices'] = np.asarray(row_indices, dtype=np.int64)
    return padded

def pool_spans(hidden_states: torch.Tensor, verse_spans: torch.Tensor, verse_mask: torch.Tensor) -> torch.Tensor:
    prefix = torch.cat([hidden_states.new_zeros(hidden_states.size(0), 1, hidden_states.size(-1)), hidden_states.cumsum(dim=1)], dim=1)
    starts = verse_spans[..., 0]
    ends = verse_spans[..., 1]
    batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device).unsqueeze(1)
    span_sums = prefix[batch_index, ends] - prefix[batch_index, starts]
    lengths = (ends - starts).clamp_min(1).unsqueeze(-1)
    return (span_sums / lengths)[verse_mask]

def export_contextual(*, frame: pd.DataFrame, config: dict, tokenizer, model: ScratchTransformerEncoder, projection: torch.nn.Linear, projection_dim: int, output_dir: Path, batch_size: int, device: torch.device, dtype: np.dtype, normalize: bool, overwrite: bool) -> dict[str, int | str]:
    path = output_path(output_dir, 'contextual')
    tmp_path = temporary_output_path(output_dir, 'contextual')
    prepare_output(path, tmp_path, overwrite=overwrite)
    matrix = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=dtype, shape=(len(frame), projection_dim))
    written = np.zeros(len(frame), dtype=np.bool_)
    window_batch: list[dict[str, object]] = []
    num_windows = 0
    max_verses_per_window = 0
    max_sequence_length = 0

    def flush_batch() -> None:
        nonlocal window_batch, max_verses_per_window, max_sequence_length
        if not window_batch:
            return
        batch = collate_contextual_windows(window_batch, tokenizer.pad_id)
        max_verses_per_window = max(max_verses_per_window, int(batch['verse_mask'].sum(dim=1).max().item()))
        max_sequence_length = max(max_sequence_length, int(batch['input_ids'].shape[1]))
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), return_pooled=False)
            vectors = pool_spans(outputs['last_hidden_state'], batch['verse_spans'].to(device), batch['verse_mask'].to(device))
            vectors = projection(vectors).detach().cpu().numpy().astype(np.float32)
        vectors = maybe_normalize(vectors, normalize=normalize).astype(dtype, copy=False)
        rows = batch['row_indices']
        matrix[rows] = vectors
        written[rows] = True
        window_batch = []
    for window in iter_contextual_windows(frame=frame, config=config, tokenizer=tokenizer):
        window_batch.append(window)
        num_windows += 1
        if len(window_batch) >= batch_size:
            flush_batch()
        if num_windows and num_windows % 20000 == 0:
            print(f'contextual: processed {num_windows:,} windows; rows written={int(written.sum()):,}', flush=True)
    flush_batch()
    missing = np.flatnonzero(~written)
    if len(missing):
        examples = missing[:10].astype(int).tolist()
        raise RuntimeError(f'Contextual export missed {len(missing)} rows; examples={examples}')
    matrix.flush()
    del matrix
    finalize_output(path, tmp_path, overwrite=overwrite)
    return {'path': str(path), 'rows': int(len(frame)), 'dim': int(projection_dim), 'windows': int(num_windows), 'max_verses_per_window': int(max_verses_per_window), 'max_sequence_length': int(max_sequence_length)}

def write_manifest(payload: dict, output_dir: Path) -> None:
    with (output_dir / 'manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    verses_path = resolve_path(args.verses_path)
    output_dir = resolve_path(args.output_dir)
    dtype = np.dtype(args.dtype)
    normalize = not bool(args.no_normalize)
    device = resolve_device(str(args.device))
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not verses_path.exists():
        raise FileNotFoundError(verses_path)
    (config, tokenizer, model, projection, projection_dim) = load_scratch_model(config_path, checkpoint_path, device)
    text_column = str(config['data']['filter_text_column'])
    frame = load_verses(verses_path, text_column=text_column, limit_rows=args.limit_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'metadata.parquet').exists() and (not args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing metadata: {output_dir / 'metadata.parquet'}")
    save_metadata(frame, output_dir)
    summaries: dict[str, dict[str, int | str]] = {}
    if 'isolated' in args.modes:
        summaries['isolated'] = export_isolated(frame=frame, config=config, tokenizer=tokenizer, model=model, projection=projection, projection_dim=projection_dim, output_dir=output_dir, batch_size=int(args.batch_size_isolated), device=device, dtype=dtype, normalize=normalize, overwrite=bool(args.overwrite))
    if 'contextual' in args.modes:
        summaries['contextual'] = export_contextual(frame=frame, config=config, tokenizer=tokenizer, model=model, projection=projection, projection_dim=projection_dim, output_dir=output_dir, batch_size=int(args.batch_size_contextual), device=device, dtype=dtype, normalize=normalize, overwrite=bool(args.overwrite))
    write_manifest({'model_role': 'best_semantic_embedding_model', 'model_label': 'paper_versebal_v0p45_w0p45_mlm0p10_epoch200', 'config': str(config_path), 'checkpoint': str(checkpoint_path), 'verses_path': str(verses_path), 'output_dir': str(output_dir), 'metadata_path': str(output_dir / 'metadata.parquet'), 'rows': int(len(frame)), 'embedding_dim': int(projection_dim), 'dtype': str(dtype), 'normalized': bool(normalize), 'row_alignment': 'Row i in each .npy matrix corresponds to metadata.parquet row with embedding_row == i.', 'modes': summaries}, output_dir)
    print(f'Done. Wrote outputs under {output_dir}', flush=True)
if __name__ == '__main__':
    main()
