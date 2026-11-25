"""Build an English-mined original-verse similarity benchmark.

The benchmark mines positives from frozen English translation embeddings, then
evaluators use those positives against original-side FILTER embeddings.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.data import load_split_table, split_frame
from filter_verse_embeddings.evaluation.verse_ids import DEFAULT_POEMS_PATH, DEFAULT_VERSES_CL_PATH, DEFAULT_VERSES_TRANSLATED_PATH, attach_canonical_verse_ids, normalized_text_codes
FLASH_ROOT = Path('artifacts')
SCRATCH_ROOT = Path('artifacts')
DEFAULT_VERSES_PATH = Path('artifacts/data/paper_verses.parquet')
DEFAULT_ENGLISH_EMBEDDINGS = Path('artifacts/cache/english_all_mpnet_base_v2/test.npy')
DEFAULT_OUTPUT_DIR = Path('artifacts/data/benchmarks/retrieval')
DEFAULT_THRESHOLDS = (0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)
BUCKET_BY_COLLECTION = {'erab': 'est', 'skvr': 'fin', 'kr': 'fin', 'jr': 'fin'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verses-path', type=Path, default=DEFAULT_VERSES_PATH)
    parser.add_argument('--english-embeddings', type=Path, default=DEFAULT_ENGLISH_EMBEDDINGS)
    parser.add_argument('--verses-translated-path', type=Path, default=DEFAULT_VERSES_TRANSLATED_PATH)
    parser.add_argument('--poems-path', type=Path, default=DEFAULT_POEMS_PATH)
    parser.add_argument('--verses-cl-path', type=Path, default=DEFAULT_VERSES_CL_PATH)
    parser.add_argument('--split', default='test')
    parser.add_argument('--thresholds', nargs='+', type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument('--min-positive-queries', type=int, default=10000)
    parser.add_argument('--ignore-threshold', type=float, default=0.95, help='English-similarity threshold for ambiguous neighbours ignored during filtered evaluation.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float16')
    parser.add_argument('--chunk-size', type=int, default=8192)
    parser.add_argument('--limit-rows', type=int, default=None, help='Smoke-test cap after split/eligibility filtering.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def resolve_device(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == 'float16' else torch.float32

def bucket_for_collection(value: object) -> str:
    return BUCKET_BY_COLLECTION.get(str(value), 'other')

def pair_schema() -> pa.Schema:
    return pa.schema([pa.field('query_row', pa.int64()), pa.field('target_row', pa.int64()), pa.field('query_v_id', pa.int64()), pa.field('target_v_id', pa.int64()), pa.field('query_p_id', pa.int64()), pa.field('target_p_id', pa.int64()), pa.field('query_pos', pa.int32()), pa.field('target_pos', pa.int32()), pa.field('query_verse_id', pa.string()), pa.field('target_verse_id', pa.string()), pa.field('query_paper_verse_id', pa.string()), pa.field('target_paper_verse_id', pa.string()), pa.field('query_poem_id', pa.string()), pa.field('target_poem_id', pa.string()), pa.field('query_source_collection', pa.string()), pa.field('target_source_collection', pa.string()), pa.field('query_bucket', pa.string()), pa.field('target_bucket', pa.string()), pa.field('english_similarity', pa.float32()), pa.field('same_poem', pa.bool_()), pa.field('same_original_raw', pa.bool_()), pa.field('same_original_normalized', pa.bool_()), pa.field('same_english_raw', pa.bool_()), pa.field('same_english_normalized', pa.bool_()), pa.field('exact_original_match', pa.bool_())])

def load_metadata_and_embeddings(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    frame = split_frame(load_split_table(args.verses_path), str(args.split))
    (frame, canonical_summary) = attach_canonical_verse_ids(frame, verses_translated_path=args.verses_translated_path, poems_path=args.poems_path, verses_cl_path=args.verses_cl_path)
    english_text = frame['english_verse'].fillna('').astype(str).str.strip()
    has_canonical_identity = frame['canonical_v_id'].notna().to_numpy()
    eligible = english_text.astype(bool).to_numpy() & has_canonical_identity
    eligible_indices = np.flatnonzero(eligible).astype(np.int64)
    if args.limit_rows is not None:
        eligible_indices = eligible_indices[:int(args.limit_rows)]
    embeddings = np.load(args.english_embeddings, mmap_mode='r')
    if int(embeddings.shape[0]) != int(len(frame)):
        raise ValueError(f'English embedding row count does not match split frame: embeddings={embeddings.shape[0]} split_rows={len(frame)}')
    selected = np.asarray(embeddings[eligible_indices], dtype=np.float32)
    canonical_summary['eligible_rows_with_english_and_canonical_v_id'] = int(len(eligible_indices))
    return (frame, eligible_indices, selected, canonical_summary)

def normalize_embeddings(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    tensor = tensor / tensor.norm(dim=1, keepdim=True).clamp(min=1e-08)
    if device.type == 'cuda':
        try:
            return tensor.pin_memory().to(device=device, dtype=dtype, non_blocking=True)
        except RuntimeError:
            return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device, dtype=dtype)

def text_codes(values: pd.Series) -> np.ndarray:
    (codes, _) = pd.factorize(values.fillna('').astype(str).to_numpy(dtype=object), sort=False)
    return codes.astype(np.int64, copy=False)

def mask_identical_identity_pairs(sim: torch.Tensor, *, left_ids: np.ndarray, right_ids: np.ndarray, device: torch.device) -> None:
    forbidden = left_ids[:, None] == right_ids[None, :]
    if np.any(forbidden):
        sim.masked_fill_(torch.from_numpy(forbidden).to(device=device, non_blocking=True), -2.0)

def calibrate_thresholds(embeddings: torch.Tensor, thresholds: list[float], chunk_size: int, identity_ids: np.ndarray) -> list[dict[str, int | float]]:
    n_rows = int(embeddings.shape[0])
    max_sim = torch.full((n_rows,), -2.0, dtype=torch.float32, device=embeddings.device)
    pair_counts = [0 for _ in thresholds]
    for i_start in range(0, n_rows, int(chunk_size)):
        i_end = min(i_start + int(chunk_size), n_rows)
        left = embeddings[i_start:i_end]
        for j_start in range(i_start, n_rows, int(chunk_size)):
            j_end = min(j_start + int(chunk_size), n_rows)
            right = embeddings[j_start:j_end]
            sim = torch.mm(left, right.T).float()
            same_block = i_start == j_start
            mask_identical_identity_pairs(sim, left_ids=identity_ids[i_start:i_end], right_ids=identity_ids[j_start:j_end], device=embeddings.device)
            max_sim[i_start:i_end] = torch.maximum(max_sim[i_start:i_end], sim.max(dim=1).values)
            if not same_block:
                max_sim[j_start:j_end] = torch.maximum(max_sim[j_start:j_end], sim.max(dim=0).values)
            for (threshold_idx, threshold) in enumerate(thresholds):
                if same_block:
                    count = int(torch.triu(sim >= float(threshold), diagonal=1).sum().item())
                else:
                    count = int((sim >= float(threshold)).sum().item())
                pair_counts[threshold_idx] += count
    max_sim_cpu = max_sim.detach().cpu().numpy()
    return [{'threshold': float(threshold), 'undirected_pairs': int(pair_counts[idx]), 'directed_positive_rows': int(pair_counts[idx] * 2), 'positive_query_verses': int(np.sum(max_sim_cpu >= float(threshold)))} for (idx, threshold) in enumerate(thresholds)]

def choose_threshold(calibration: list[dict[str, int | float]], min_positive_queries: int) -> float:
    candidates = [float(row['threshold']) for row in calibration if int(row['positive_query_verses']) >= int(min_positive_queries)]
    if not candidates:
        raise ValueError(f"No threshold produced at least {min_positive_queries} positive query verses. Lowest threshold query count was {max((int(row['positive_query_verses']) for row in calibration))}.")
    return max(candidates)

def choose_ignore_threshold(selected_threshold: float, requested_threshold: float | None, thresholds: list[float]) -> float | None:
    if requested_threshold is None:
        return None
    if float(requested_threshold) < float(selected_threshold):
        return float(requested_threshold)
    lower = [float(value) for value in thresholds if float(value) < float(selected_threshold)]
    if not lower:
        return None
    return max(lower)

def write_calibration(path: Path, rows: list[dict[str, int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['threshold', 'undirected_pairs', 'directed_positive_rows', 'positive_query_verses'])
        writer.writeheader()
        writer.writerows(rows)

def write_directed_batch(*, writer: pq.ParquetWriter, left_indices: np.ndarray, right_indices: np.ndarray, similarities: np.ndarray, full_rows: np.ndarray, canonical_v_ids: np.ndarray, canonical_p_ids: np.ndarray, canonical_positions: np.ndarray, paper_verse_ids: np.ndarray, poem_ids: np.ndarray, source_collections: np.ndarray, buckets: np.ndarray, original_raw_codes: np.ndarray, original_normalized_codes: np.ndarray, english_raw_codes: np.ndarray, english_normalized_codes: np.ndarray, stats: Counter) -> None:
    query_indices = np.concatenate([left_indices, right_indices])
    target_indices = np.concatenate([right_indices, left_indices])
    sim_values = np.concatenate([similarities, similarities]).astype(np.float32, copy=False)
    same_poem = poem_ids[query_indices] == poem_ids[target_indices]
    same_original_raw = original_raw_codes[query_indices] == original_raw_codes[target_indices]
    same_original_normalized = original_normalized_codes[query_indices] == original_normalized_codes[target_indices]
    same_english_raw = english_raw_codes[query_indices] == english_raw_codes[target_indices]
    same_english_normalized = english_normalized_codes[query_indices] == english_normalized_codes[target_indices]
    directed_buckets = [f'{q}-{t}' for (q, t) in zip(buckets[query_indices], buckets[target_indices])]
    stats['directed_positive_rows'] += int(len(query_indices))
    stats['same_poem_directed_rows'] += int(np.sum(same_poem))
    stats['same_original_raw_directed_rows'] += int(np.sum(same_original_raw))
    stats['same_original_normalized_directed_rows'] += int(np.sum(same_original_normalized))
    stats['same_english_raw_directed_rows'] += int(np.sum(same_english_raw))
    stats['same_english_normalized_directed_rows'] += int(np.sum(same_english_normalized))
    stats['exact_original_match_directed_rows'] += int(np.sum(same_original_normalized))
    for (key, count) in Counter(directed_buckets).items():
        stats[f'bucket_pair_count::{key}'] += int(count)
    flag_payload = {'same_poem': same_poem, 'same_original_raw': same_original_raw, 'same_original_normalized': same_original_normalized, 'same_english_raw': same_english_raw, 'same_english_normalized': same_english_normalized}
    for (flag_name, mask) in flag_payload.items():
        for (key, count) in Counter(np.asarray(directed_buckets, dtype=object)[mask].tolist()).items():
            stats[f'{flag_name}_bucket_pair_count::{key}'] += int(count)
    table = pa.table({'query_row': pa.array(full_rows[query_indices], type=pa.int64()), 'target_row': pa.array(full_rows[target_indices], type=pa.int64()), 'query_v_id': pa.array(canonical_v_ids[query_indices], type=pa.int64()), 'target_v_id': pa.array(canonical_v_ids[target_indices], type=pa.int64()), 'query_p_id': pa.array(canonical_p_ids[query_indices], type=pa.int64()), 'target_p_id': pa.array(canonical_p_ids[target_indices], type=pa.int64()), 'query_pos': pa.array(canonical_positions[query_indices], type=pa.int32()), 'target_pos': pa.array(canonical_positions[target_indices], type=pa.int32()), 'query_verse_id': pa.array([str(value) for value in canonical_v_ids[query_indices]], type=pa.string()), 'target_verse_id': pa.array([str(value) for value in canonical_v_ids[target_indices]], type=pa.string()), 'query_paper_verse_id': pa.array(paper_verse_ids[query_indices].tolist(), type=pa.string()), 'target_paper_verse_id': pa.array(paper_verse_ids[target_indices].tolist(), type=pa.string()), 'query_poem_id': pa.array(poem_ids[query_indices].tolist(), type=pa.string()), 'target_poem_id': pa.array(poem_ids[target_indices].tolist(), type=pa.string()), 'query_source_collection': pa.array(source_collections[query_indices].tolist(), type=pa.string()), 'target_source_collection': pa.array(source_collections[target_indices].tolist(), type=pa.string()), 'query_bucket': pa.array(buckets[query_indices].tolist(), type=pa.string()), 'target_bucket': pa.array(buckets[target_indices].tolist(), type=pa.string()), 'english_similarity': pa.array(sim_values, type=pa.float32()), 'same_poem': pa.array(same_poem, type=pa.bool_()), 'same_original_raw': pa.array(same_original_raw, type=pa.bool_()), 'same_original_normalized': pa.array(same_original_normalized, type=pa.bool_()), 'same_english_raw': pa.array(same_english_raw, type=pa.bool_()), 'same_english_normalized': pa.array(same_english_normalized, type=pa.bool_()), 'exact_original_match': pa.array(same_original_normalized, type=pa.bool_())}, schema=pair_schema())
    writer.write_table(table)

def mine_pairs(*, frame, eligible_indices: np.ndarray, embeddings: torch.Tensor, threshold: float, output_path: Path, chunk_size: int) -> dict[str, object]:
    n_rows = int(embeddings.shape[0])
    selected = frame.iloc[eligible_indices].reset_index(drop=True)
    full_rows = eligible_indices.astype(np.int64, copy=False)
    canonical_v_ids = pd.to_numeric(selected['canonical_v_id'], errors='raise').astype('int64').to_numpy()
    canonical_p_ids = pd.to_numeric(selected['canonical_p_id'], errors='raise').astype('int64').to_numpy()
    canonical_positions = pd.to_numeric(selected['canonical_pos'], errors='raise').astype('int32').to_numpy()
    paper_verse_ids = selected['verse_id'].fillna('').astype(str).to_numpy(dtype=object)
    poem_ids = selected['poem_id'].fillna('').astype(str).to_numpy(dtype=object)
    source_collections = selected['source_collection'].fillna('').astype(str).to_numpy(dtype=object)
    buckets = np.asarray([bucket_for_collection(value) for value in source_collections], dtype=object)
    original_raw_codes = text_codes(selected['canonical_original_text'])
    original_normalized_codes = normalized_text_codes(selected['canonical_original_text'])
    english_raw_codes = text_codes(selected['english_verse'])
    english_normalized_codes = normalized_text_codes(selected['english_verse'])
    has_positive = np.zeros(n_rows, dtype=bool)
    stats: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f'.{output_path.name}.tmp')
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with pq.ParquetWriter(tmp_path, pair_schema(), compression='zstd') as writer:
            for i_start in range(0, n_rows, int(chunk_size)):
                i_end = min(i_start + int(chunk_size), n_rows)
                left = embeddings[i_start:i_end]
                for j_start in range(i_start, n_rows, int(chunk_size)):
                    j_end = min(j_start + int(chunk_size), n_rows)
                    right = embeddings[j_start:j_end]
                    sim = torch.mm(left, right.T).float()
                    mask_identical_identity_pairs(sim, left_ids=canonical_v_ids[i_start:i_end], right_ids=canonical_v_ids[j_start:j_end], device=embeddings.device)
                    if i_start == j_start:
                        mask = torch.triu(sim >= float(threshold), diagonal=1)
                    else:
                        mask = sim >= float(threshold)
                    (li, lj) = mask.nonzero(as_tuple=True)
                    if not li.numel():
                        continue
                    left_abs = li.cpu().numpy().astype(np.int64, copy=False) + i_start
                    right_abs = lj.cpu().numpy().astype(np.int64, copy=False) + j_start
                    sims = sim[li, lj].detach().cpu().numpy().astype(np.float32, copy=False)
                    has_positive[left_abs] = True
                    has_positive[right_abs] = True
                    stats['undirected_pairs'] += int(len(left_abs))
                    write_directed_batch(writer=writer, left_indices=left_abs, right_indices=right_abs, similarities=sims, full_rows=full_rows, canonical_v_ids=canonical_v_ids, canonical_p_ids=canonical_p_ids, canonical_positions=canonical_positions, paper_verse_ids=paper_verse_ids, poem_ids=poem_ids, source_collections=source_collections, buckets=buckets, original_raw_codes=original_raw_codes, original_normalized_codes=original_normalized_codes, english_raw_codes=english_raw_codes, english_normalized_codes=english_normalized_codes, stats=stats)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    positive_query_counts = Counter(buckets[has_positive].tolist())
    eligible_counts = Counter(buckets.tolist())
    bucket_pair_counts = {key.split('::', maxsplit=1)[1]: int(value) for (key, value) in stats.items() if key.startswith('bucket_pair_count::')}
    directed_rows = int(stats['directed_positive_rows'])
    same_poem_rows = int(stats['same_poem_directed_rows'])
    exact_original_rows = int(stats['exact_original_match_directed_rows'])
    same_original_raw_rows = int(stats['same_original_raw_directed_rows'])
    same_original_normalized_rows = int(stats['same_original_normalized_directed_rows'])
    same_english_raw_rows = int(stats['same_english_raw_directed_rows'])
    same_english_normalized_rows = int(stats['same_english_normalized_directed_rows'])
    flag_bucket_counts = {flag: {key.split('::', maxsplit=1)[1]: int(value) for (key, value) in stats.items() if key.startswith(f'{flag}_bucket_pair_count::')} for flag in ('same_poem', 'same_original_raw', 'same_original_normalized', 'same_english_raw', 'same_english_normalized')}
    return {'output_path': str(output_path), 'threshold': float(threshold), 'eligible_rows': int(n_rows), 'undirected_pairs': int(stats['undirected_pairs']), 'directed_positive_rows': directed_rows, 'positive_query_verses': int(np.sum(has_positive)), 'eligible_rows_by_bucket': dict(sorted(((str(k), int(v)) for (k, v) in eligible_counts.items()))), 'positive_query_verses_by_bucket': dict(sorted(((str(k), int(v)) for (k, v) in positive_query_counts.items()))), 'directed_pair_counts_by_bucket': dict(sorted(bucket_pair_counts.items())), 'same_poem_directed_rows': same_poem_rows, 'same_poem_directed_rate': float(same_poem_rows / directed_rows) if directed_rows else 0.0, 'same_original_raw_directed_rows': same_original_raw_rows, 'same_original_raw_directed_rate': float(same_original_raw_rows / directed_rows) if directed_rows else 0.0, 'same_original_normalized_directed_rows': same_original_normalized_rows, 'same_original_normalized_directed_rate': float(same_original_normalized_rows / directed_rows) if directed_rows else 0.0, 'same_english_raw_directed_rows': same_english_raw_rows, 'same_english_raw_directed_rate': float(same_english_raw_rows / directed_rows) if directed_rows else 0.0, 'same_english_normalized_directed_rows': same_english_normalized_rows, 'same_english_normalized_directed_rate': float(same_english_normalized_rows / directed_rows) if directed_rows else 0.0, 'exact_original_match_directed_rows': exact_original_rows, 'exact_original_match_directed_rate': float(exact_original_rows / directed_rows) if directed_rows else 0.0, 'directed_pair_counts_by_flag_and_bucket': {flag: dict(sorted(counts.items())) for (flag, counts) in flag_bucket_counts.items()}}

def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pairs_path = output_dir / f'original_similarity_pairs_{args.split}.parquet'
    ignore_pairs_path = output_dir / f'original_similarity_ignore_pairs_{args.split}.parquet'
    calibration_path = output_dir / f'threshold_calibration_{args.split}.csv'
    manifest_path = output_dir / f'original_similarity_benchmark_{args.split}.json'
    for path in (pairs_path, ignore_pairs_path, calibration_path, manifest_path):
        if path.exists() and (not args.overwrite):
            raise FileExistsError(f'Refusing to overwrite existing output without --overwrite: {path}')
    thresholds = sorted({float(value) for value in args.thresholds})
    device = resolve_device(str(args.device))
    dtype = dtype_from_name(str(args.dtype))
    print(f'Loading metadata and English embeddings for split={args.split}', flush=True)
    (frame, eligible_indices, english_embeddings, canonical_summary) = load_metadata_and_embeddings(args)
    print(f'Split rows={len(frame):,}; eligible English rows={len(eligible_indices):,}; embedding_dim={english_embeddings.shape[1]}', flush=True)
    eligible_canonical_v_ids = pd.to_numeric(frame.iloc[eligible_indices]['canonical_v_id'], errors='raise').astype('int64').to_numpy()
    embeddings = normalize_embeddings(english_embeddings, device=device, dtype=dtype)
    print(f'Calibrating thresholds on {device} with dtype={dtype}', flush=True)
    calibration = calibrate_thresholds(embeddings, thresholds, int(args.chunk_size), eligible_canonical_v_ids)
    selected_threshold = choose_threshold(calibration, int(args.min_positive_queries))
    ignore_threshold = choose_ignore_threshold(selected_threshold, args.ignore_threshold, thresholds)
    write_calibration(calibration_path, calibration)
    print(f'Selected threshold={selected_threshold}', flush=True)
    print(f'Ignore threshold={ignore_threshold}', flush=True)
    mining_summary = mine_pairs(frame=frame, eligible_indices=eligible_indices, embeddings=embeddings, threshold=selected_threshold, output_path=pairs_path, chunk_size=int(args.chunk_size))
    ignore_summary = None
    if ignore_threshold is not None:
        ignore_summary = mine_pairs(frame=frame, eligible_indices=eligible_indices, embeddings=embeddings, threshold=ignore_threshold, output_path=ignore_pairs_path, chunk_size=int(args.chunk_size))
    manifest = {'benchmark': 'english_mined_original_similarity', 'split': str(args.split), 'verses_path': str(args.verses_path), 'english_embeddings': str(args.english_embeddings), 'verses_translated_path': str(args.verses_translated_path), 'poems_path': str(args.poems_path), 'verses_cl_path': str(args.verses_cl_path), 'canonical_identity_summary': canonical_summary, 'thresholds': thresholds, 'min_positive_queries': int(args.min_positive_queries), 'selected_threshold': float(selected_threshold), 'ignore_threshold': None if ignore_threshold is None else float(ignore_threshold), 'total_split_rows': int(len(frame)), 'limit_rows': args.limit_rows, 'calibration_csv': str(calibration_path), 'pairs_parquet': str(pairs_path), 'ignore_pairs_parquet': None if ignore_threshold is None else str(ignore_pairs_path), 'calibration': calibration, 'mining_summary': mining_summary, 'ignore_summary': ignore_summary, 'bucket_policy': {'erab': 'est', 'skvr': 'fin', 'kr': 'fin', 'jr': 'fin', 'default': 'other'}, 'identity_policy': 'canonical FILTER v_id from verses_translated joined through poems.nro and verse_position', 'self_mask_policy': 'exclude identical canonical v_id', 'exact_original_policy': 'flag exact original matches; do not exclude them from mined positives', 'filtered_ranking_policy': 'evaluation may ignore non-positive neighbours at or above ignore_threshold so near-positive unlabeled lines do not count against a model', 'paper_verse_id_policy': 'generated Paper poem-local ids are retained only as *_paper_verse_id labels'}
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=True), flush=True)
if __name__ == '__main__':
    main()
