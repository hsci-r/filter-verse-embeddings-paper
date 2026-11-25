"""Evaluate original-side retrieval against English-near positive sets."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BUCKET_BY_COLLECTION = {'erab': 'est', 'skvr': 'fin', 'kr': 'fin', 'jr': 'fin'}
QUERY_BUCKETS = ('fin', 'est')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run', required=True)
    parser.add_argument('--benchmark-pairs', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--english-similarity-threshold', type=float, default=0.99)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--cross-top-k', type=int, default=10)
    parser.add_argument('--max-queries', type=int, default=None, help='Debug cap after positive-set construction.')
    parser.add_argument('--evaluation-name', default='english_mined_formulaic_set_p99')
    return parser.parse_args()

def bucket_for_collection(value: object) -> str:
    return BUCKET_BY_COLLECTION.get(str(value).lower(), 'other')

def load_split_matrix(run_dir: str | Path, split: str, side: str):
    run_dir = Path(run_dir)
    sparse_path = run_dir / 'embeddings' / f'{split}_{side}.npz'
    dense_path = run_dir / 'embeddings' / f'{split}_{side}.npy'
    if sparse_path.exists():
        from scipy import sparse
        return sparse.load_npz(sparse_path).tocsr()
    if dense_path.exists():
        return np.load(dense_path).astype(np.float32, copy=False)
    raise FileNotFoundError(f'No embedding matrix found for split={split!r} side={side!r}')

def load_split_metadata(run_dir: str | Path, split: str) -> pd.DataFrame:
    run_dir = Path(run_dir)
    return pd.read_parquet(run_dir / 'metadata' / f'{split}_metadata.parquet')

def load_split_artifacts(run_dir: str | Path, split: str, side: str):
    return (load_split_matrix(run_dir, split, side), load_split_metadata(run_dir, split))

def load_run_metadata(run_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(run_dir) / 'metadata.json'
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding='utf-8'))

def matrix_take_rows(matrix, row_indices: np.ndarray):
    return matrix[row_indices]

def similarity_scores(query_matrix, gallery_matrix) -> np.ndarray:
    try:
        from scipy import sparse
    except ModuleNotFoundError:
        sparse = None
    if sparse is not None and (sparse.issparse(query_matrix) or sparse.issparse(gallery_matrix)):
        query_sparse = query_matrix if sparse.issparse(query_matrix) else sparse.csr_matrix(query_matrix)
        gallery_sparse = gallery_matrix if sparse.issparse(gallery_matrix) else sparse.csr_matrix(gallery_matrix)
        return np.asarray(query_sparse.dot(gallery_sparse.T).toarray(), dtype=np.float32)
    return np.asarray(query_matrix @ gallery_matrix.T, dtype=np.float32)

def topk_indices(score_row: np.ndarray, k: int) -> np.ndarray:
    if len(score_row) <= k:
        return np.argsort(-score_row, kind='stable')
    partition = np.argpartition(-score_row, k - 1)[:k]
    return partition[np.argsort(-score_row[partition], kind='stable')]

def write_json(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')

def align_pair_rows_to_metadata(pairs: pd.DataFrame, metadata: pd.DataFrame, num_rows: int) -> tuple[pd.DataFrame, dict[str, int]]:
    checks: dict[str, int] = {}
    metadata_verse_ids = metadata['verse_id'].astype(str).to_numpy()
    metadata_row_by_verse_id: dict[str, int] = {}
    duplicate_metadata_ids = 0
    for (idx, verse_id) in enumerate(metadata_verse_ids):
        if verse_id in metadata_row_by_verse_id:
            duplicate_metadata_ids += 1
            continue
        metadata_row_by_verse_id[str(verse_id)] = int(idx)
    if duplicate_metadata_ids:
        raise ValueError(f'Metadata has {duplicate_metadata_ids} duplicate verse_id values.')
    aligned = pairs.copy()
    for side in ('query', 'target'):
        row_column = f'{side}_row'
        id_column = f'{side}_paper_verse_id'
        if row_column not in aligned.columns:
            continue
        rows = aligned[row_column].astype('int64').to_numpy()
        valid = (rows >= 0) & (rows < int(num_rows))
        checks[f'{side}_alignment_checked_rows'] = int(len(rows))
        checks[f'{side}_alignment_invalid_rows'] = int((~valid).sum())
        if id_column not in aligned.columns:
            if not np.all(valid):
                raise ValueError(f'Benchmark has {int((~valid).sum())} invalid {side} row indices.')
            continue
        expected = aligned[id_column].astype(str).to_numpy()
        mapped_rows = np.fromiter((metadata_row_by_verse_id.get(str(verse_id), -1) for verse_id in expected), dtype=np.int64, count=len(expected))
        missing = mapped_rows < 0
        checks[f'{side}_alignment_missing_ids'] = int(missing.sum())
        if np.any(missing):
            examples = aligned.loc[missing, [row_column, id_column]].head(5).to_dict('records')
            raise ValueError(f'Benchmark {side} IDs are missing from metadata; examples={examples}')
        actual = np.full(len(rows), '', dtype=object)
        actual[valid] = metadata_verse_ids[rows[valid]]
        mismatches = ~valid | (actual != expected)
        checks[f'{side}_alignment_mismatches'] = int(mismatches.sum())
        checks[f'{side}_alignment_remapped_rows'] = int(mismatches.sum())
        if np.any(mismatches):
            aligned[row_column] = mapped_rows
    return (aligned, checks)

def build_positive_sets(pairs: pd.DataFrame, *, num_rows: int, threshold: float) -> tuple[dict[int, set[int]], dict[str, int]]:
    required = {'query_row', 'target_row', 'english_similarity'}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f'Benchmark pair file is missing required columns: {missing}')
    filtered = pairs[pd.to_numeric(pairs['english_similarity'], errors='coerce') > float(threshold)]
    positive_sets: dict[int, set[int]] = defaultdict(set)
    dropped = {'below_threshold': int(len(pairs) - len(filtered)), 'missing_query': 0, 'missing_target': 0, 'self_edges': 0}
    for row in filtered.itertuples(index=False):
        query_idx = int(getattr(row, 'query_row'))
        target_idx = int(getattr(row, 'target_row'))
        if query_idx < 0 or query_idx >= int(num_rows):
            dropped['missing_query'] += 1
            continue
        if target_idx < 0 or target_idx >= int(num_rows):
            dropped['missing_target'] += 1
            continue
        if query_idx == target_idx:
            dropped['self_edges'] += 1
            continue
        positive_sets[query_idx].add(target_idx)
    return (dict(positive_sets), dropped)

def stable_positive_ranks(score_row: np.ndarray, positives: np.ndarray) -> list[int]:
    ranks = []
    for positive_idx in sorted((int(idx) for idx in positives)):
        positive_score = float(score_row[positive_idx])
        better = int(np.sum(score_row > positive_score))
        tied_before = int(np.sum(score_row[:positive_idx] == positive_score))
        ranks.append(better + tied_before + 1)
    return ranks

def average_precision(ranks: list[int]) -> float:
    if not ranks:
        return math.nan
    ranked = np.asarray(sorted(ranks), dtype=np.float64)
    precision_at_positive = np.arange(1, len(ranked) + 1, dtype=np.float64) / ranked
    return float(np.mean(precision_at_positive))

class MetricAccumulator:

    def __init__(self, top_k: int, cross_top_k: int) -> None:
        self.top_k = int(top_k)
        self.cross_top_k = int(cross_top_k)
        self.num_queries = 0
        self.positive_edges = 0
        self.p_at_1_values: list[float] = []
        self.p_at_5_values: list[float] = []
        self.p_at_100_values: list[float] = []
        self.ap_values: list[float] = []
        self.positive_counts: list[int] = []
        self.same_language_positive_counts: list[int] = []
        self.cross_language_positive_counts: list[int] = []
        self.positive_scores: list[float] = []
        self.same_language_positive_scores: list[float] = []
        self.cross_language_positive_scores: list[float] = []
        self.first_correct_cross_language_positive_ranks: list[int] = []
        self.queries_with_cross_language_positive = 0
        self.queries_with_any_cross_language_topk = 0
        self.queries_with_any_correct_cross_language_topk = 0
        self.cross_language_topk_count = 0
        self.correct_cross_language_topk_count = 0

    def add(self, *, score_row: np.ndarray, positives: np.ndarray, top_indices: np.ndarray, query_bucket: str, gallery_buckets: np.ndarray) -> None:
        positive_set = set((int(idx) for idx in positives))
        positive_count = len(positive_set)
        if positive_count == 0:
            return
        top1 = top_indices[:1]
        top5 = top_indices[:5]
        top100 = top_indices[:min(100, len(top_indices))]
        cross_top = top_indices[:min(self.cross_top_k, len(top_indices))]
        positive_scores = [float(score_row[int(idx)]) for idx in positives]
        positive_buckets = gallery_buckets[positives]
        cross_positive_mask = positive_buckets != query_bucket
        cross_positive_indices = positives[cross_positive_mask]
        same_language_positive_count = int(np.sum(~cross_positive_mask))
        cross_language_positive_count = int(np.sum(cross_positive_mask))
        cross_top_mask = gallery_buckets[cross_top] != query_bucket
        correct_cross_top_mask = np.asarray([int(idx) in positive_set for idx in cross_top], dtype=bool) & cross_top_mask
        self.num_queries += 1
        self.positive_edges += positive_count
        self.positive_counts.append(positive_count)
        self.same_language_positive_counts.append(same_language_positive_count)
        self.cross_language_positive_counts.append(cross_language_positive_count)
        self.p_at_1_values.append(float(sum((int(idx) in positive_set for idx in top1))) / 1.0)
        self.p_at_5_values.append(float(sum((int(idx) in positive_set for idx in top5))) / 5.0)
        self.p_at_100_values.append(float(sum((int(idx) in positive_set for idx in top100))) / 100.0)
        self.ap_values.append(average_precision(stable_positive_ranks(score_row, positives)))
        self.positive_scores.extend(positive_scores)
        self.same_language_positive_scores.extend((float(score) for (score, is_cross) in zip(positive_scores, cross_positive_mask) if not bool(is_cross)))
        self.cross_language_positive_scores.extend((float(score) for (score, is_cross) in zip(positive_scores, cross_positive_mask) if bool(is_cross)))
        if bool(np.any(cross_positive_mask)):
            self.queries_with_cross_language_positive += 1
            self.first_correct_cross_language_positive_ranks.append(min(stable_positive_ranks(score_row, cross_positive_indices)))
        if bool(np.any(cross_top_mask)):
            self.queries_with_any_cross_language_topk += 1
        if bool(np.any(correct_cross_top_mask)):
            self.queries_with_any_correct_cross_language_topk += 1
        self.cross_language_topk_count += int(np.sum(cross_top_mask))
        self.correct_cross_language_topk_count += int(np.sum(correct_cross_top_mask))

    def summary(self) -> dict[str, float | int]:
        if self.num_queries == 0:
            return {'p_at_1': 0.0, 'uap': 0.0, 'num_queries': 0, 'positive_edges': 0}
        result: dict[str, float | int] = {'p_at_1': float(np.mean(self.p_at_1_values)), 'uap': float(np.mean(self.ap_values)), 'num_queries': int(self.num_queries), 'positive_edges': int(self.positive_edges), 'mean_positives_per_query': float(np.mean(self.positive_counts)), 'median_positives_per_query': float(np.median(self.positive_counts)), 'mean_same_language_positives_per_query': float(np.mean(self.same_language_positive_counts)), 'median_same_language_positives_per_query': float(np.median(self.same_language_positive_counts)), 'mean_cross_language_positives_per_query': float(np.mean(self.cross_language_positive_counts)), 'median_cross_language_positives_per_query': float(np.median(self.cross_language_positive_counts)), 'p_at_5_diagnostic': float(np.mean(self.p_at_5_values)), 'p_at_100_diagnostic': float(np.mean(self.p_at_100_values)), 'has_cross_language_positive_rate': float(self.queries_with_cross_language_positive / self.num_queries), f'any_cross_language_top{self.cross_top_k}_rate': float(self.queries_with_any_cross_language_topk / self.num_queries), f'any_correct_cross_language_top{self.cross_top_k}_rate': float(self.queries_with_any_correct_cross_language_topk / self.num_queries), f'mean_cross_language_top{self.cross_top_k}_count': float(self.cross_language_topk_count / self.num_queries), f'mean_correct_cross_language_top{self.cross_top_k}_count': float(self.correct_cross_language_topk_count / self.num_queries), f'cross_language_top{self.cross_top_k}_precision_micro': float(self.correct_cross_language_topk_count / self.cross_language_topk_count) if self.cross_language_topk_count else 0.0}
        if self.positive_scores:
            result['mean_positive_model_similarity'] = float(np.mean(self.positive_scores))
            result['median_positive_model_similarity'] = float(np.median(self.positive_scores))
        if self.same_language_positive_scores:
            result['mean_same_language_positive_model_similarity'] = float(np.mean(self.same_language_positive_scores))
        if self.cross_language_positive_scores:
            result['mean_cross_language_positive_model_similarity'] = float(np.mean(self.cross_language_positive_scores))
        if self.first_correct_cross_language_positive_ranks:
            result['mean_first_correct_cross_language_positive_rank'] = float(np.mean(self.first_correct_cross_language_positive_ranks))
            result['median_first_correct_cross_language_positive_rank'] = float(np.median(self.first_correct_cross_language_positive_ranks))
        return result

def evaluate_positive_sets(*, query_matrix, gallery_matrix, positive_sets: dict[int, set[int]], query_buckets: np.ndarray, gallery_buckets: np.ndarray, chunk_size: int, top_k: int, cross_top_k: int, max_queries: int | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    query_indices = np.asarray(sorted(positive_sets), dtype=np.int64)
    if max_queries is not None:
        query_indices = query_indices[:int(max_queries)]
    overall = MetricAccumulator(top_k=top_k, cross_top_k=cross_top_k)
    cross_language_positive_subset = MetricAccumulator(top_k=top_k, cross_top_k=cross_top_k)
    by_query_bucket = {bucket: MetricAccumulator(top_k=top_k, cross_top_k=cross_top_k) for bucket in QUERY_BUCKETS}
    top_count = max(int(top_k), int(cross_top_k), 100)
    for start in range(0, len(query_indices), int(chunk_size)):
        query_chunk_indices = query_indices[start:start + int(chunk_size)]
        query_chunk = matrix_take_rows(query_matrix, query_chunk_indices)
        scores = similarity_scores(query_chunk, gallery_matrix)
        for (local_idx, query_idx) in enumerate(query_chunk_indices):
            score_row = np.asarray(scores[local_idx], dtype=np.float32).copy()
            score_row[int(query_idx)] = -np.inf
            positives = np.asarray(sorted(positive_sets[int(query_idx)] - {int(query_idx)}), dtype=np.int64)
            if positives.size == 0:
                continue
            top_indices = topk_indices(score_row, min(top_count, len(score_row)))
            query_bucket = str(query_buckets[int(query_idx)])
            has_cross_language_positive = bool(np.any(gallery_buckets[positives] != query_bucket))
            overall.add(score_row=score_row, positives=positives, top_indices=top_indices, query_bucket=query_bucket, gallery_buckets=gallery_buckets)
            if has_cross_language_positive:
                cross_language_positive_subset.add(score_row=score_row, positives=positives, top_indices=top_indices, query_bucket=query_bucket, gallery_buckets=gallery_buckets)
            if query_bucket in by_query_bucket:
                by_query_bucket[query_bucket].add(score_row=score_row, positives=positives, top_indices=top_indices, query_bucket=query_bucket, gallery_buckets=gallery_buckets)
    return (overall.summary(), {bucket: accumulator.summary() for (bucket, accumulator) in by_query_bucket.items()}, cross_language_positive_subset.summary())

def main() -> None:
    args = parse_args()
    (filter_matrix, filter_meta) = load_split_artifacts(args.model_run, args.split, 'filter')
    if len(filter_meta) != filter_matrix.shape[0]:
        raise ValueError('Metadata row count does not match FILTER embedding matrix.')
    pairs = pd.read_parquet(args.benchmark_pairs)
    (pairs, alignment_checks) = align_pair_rows_to_metadata(pairs, filter_meta, int(filter_matrix.shape[0]))
    (positive_sets, dropped) = build_positive_sets(pairs, num_rows=int(filter_matrix.shape[0]), threshold=float(args.english_similarity_threshold))
    if not positive_sets:
        raise ValueError('No positive sets remain after thresholding benchmark pairs.')
    buckets = filter_meta['source_collection'].map(bucket_for_collection).to_numpy(dtype=object)
    (overall_metrics, query_bucket_metrics, cross_language_subset_metrics) = evaluate_positive_sets(query_matrix=filter_matrix, gallery_matrix=filter_matrix, positive_sets=positive_sets, query_buckets=buckets, gallery_buckets=buckets, chunk_size=int(args.chunk_size), top_k=int(args.top_k), cross_top_k=int(args.cross_top_k), max_queries=args.max_queries)
    run_metadata = load_run_metadata(args.model_run)
    run_name = Path(args.model_run).name
    report = {'run_name': run_name, 'model_run': str(args.model_run), 'model_type': run_metadata.get('model_type', 'unknown'), 'split': str(args.split), 'benchmark_pairs': str(args.benchmark_pairs), 'evaluation': str(args.evaluation_name), 'positive_policy': f'all other verses with English MPNet similarity > {float(args.english_similarity_threshold):.4f}', 'ranking_policy': 'rank all other original-side FILTER verses; mask only the query row itself', 'main_metrics': ['p_at_1', 'uap'], 'diagnostic_top_k': int(args.top_k), 'cross_language_diagnostic_top_k': int(args.cross_top_k), 'row_alignment_checks': alignment_checks, 'alignment_dropped_edges': dropped, 'overall': overall_metrics, 'by_query_bucket': query_bucket_metrics, 'cross_language_positive_subset': cross_language_subset_metrics}
    if args.max_queries is not None:
        report['debug_max_queries'] = int(args.max_queries)
    if 'baseline_interpretation' in run_metadata:
        report['baseline_interpretation'] = run_metadata['baseline_interpretation']
    write_json(report, args.output_path)
if __name__ == '__main__':
    main()
