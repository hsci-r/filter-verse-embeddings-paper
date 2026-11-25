"""Shared helpers for supervised poem-type probe evaluations."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
DEFAULT_TYPES_PATH = Path('artifacts/data/metadata/types.csv')
DEFAULT_POEM_TYPES_PATH = Path('artifacts/data/metadata/poem_types.csv')
DEFAULT_VERSES_PATH = Path('artifacts/data/paper_verses.parquet')
DEFAULT_EXCLUDED_TYPE_PREFIXES = ('erab_orig',)

@dataclass
class LabelBundle:
    """Poem-level multi-label target matrices split by train/dev/test."""
    labels: list[str]
    label_names: dict[str, str]
    poem_frames: dict[str, pd.DataFrame]
    y: dict[str, np.ndarray]
    coverage: dict[str, object]

def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')

def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    out = np.empty_like(logits, dtype=np.float32)
    positive = logits >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    out[~positive] = exp_logits / (1.0 + exp_logits)
    return out

def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return (array / norms).astype(np.float32, copy=False)

def root_type_id(type_id: str, parent_map: dict[str, str]) -> str:
    current = str(type_id)
    seen: set[str] = set()
    root = current
    while current and current not in seen:
        seen.add(current)
        root = current
        parent = str(parent_map.get(current, '') or '')
        if not parent or parent == 'nan':
            break
        current = parent
    return root

def normalize_excluded_type_prefixes(prefixes: Iterable[str] | None) -> tuple[str, ...]:
    if prefixes is None:
        return tuple()
    return tuple((str(prefix).strip() for prefix in prefixes if str(prefix).strip()))

def has_excluded_prefix(type_id: str, prefixes: tuple[str, ...]) -> bool:
    value = str(type_id)
    return any((value.startswith(prefix) for prefix in prefixes))

def load_poem_table(verses_path: Path, text_column: str='filter_text') -> pd.DataFrame:
    verses = pd.read_parquet(verses_path, columns=['poem_id', 'verse_id', 'verse_position', 'split', 'source_collection', text_column])
    verses['poem_id'] = verses['poem_id'].astype(str)
    verses[text_column] = verses[text_column].fillna('').astype(str)
    verses = verses.sort_values(['poem_id', 'verse_position', 'verse_id'], kind='stable')
    poem_rows = []
    for (poem_id, group) in verses.groupby('poem_id', sort=False):
        poem_rows.append({'poem_id': str(poem_id), 'split': str(group['split'].iloc[0]), 'source_collection': str(group['source_collection'].iloc[0]), 'poem_text': '\n'.join(group[text_column].fillna('').astype(str).tolist()), 'num_verses': int(len(group))})
    return pd.DataFrame(poem_rows)

def load_primary_top_labels(*, verses_path: Path, types_path: Path, poem_types_path: Path, min_train_support: int, exclude_type_prefixes: Iterable[str] | None=DEFAULT_EXCLUDED_TYPE_PREFIXES) -> LabelBundle:
    excluded_prefixes = normalize_excluded_type_prefixes(exclude_type_prefixes)
    poems = load_poem_table(verses_path)
    types = pd.read_csv(types_path, dtype=str).fillna('')
    poem_types = pd.read_csv(poem_types_path, dtype=str).fillna('')
    poem_types = poem_types[poem_types['type_is_minor'].astype(str) == '0'].copy()
    parent_map = dict(zip(types['type_id'].astype(str), types['type_parent_id'].astype(str)))
    name_map = dict(zip(types['type_id'].astype(str), types['type_name'].astype(str)))
    poem_types['top_type_id'] = poem_types['type_id'].map(lambda value: root_type_id(str(value), parent_map))
    poem_types = poem_types[poem_types['top_type_id'].astype(str) != '']
    if excluded_prefixes:
        poem_types = poem_types[~poem_types['top_type_id'].map(lambda value: has_excluded_prefix(str(value), excluded_prefixes))].copy()
    labels_long = poem_types[['poem_id', 'top_type_id']].drop_duplicates()
    labels_long['poem_id'] = labels_long['poem_id'].astype(str)
    labels_with_split = labels_long.merge(poems[['poem_id', 'split']], on='poem_id', how='inner')
    train_counts = labels_with_split[labels_with_split['split'] == 'train']['top_type_id'].value_counts()
    keep_labels = sorted(train_counts[train_counts >= int(min_train_support)].index.astype(str).tolist())
    label_to_idx = {label: idx for (idx, label) in enumerate(keep_labels)}
    labels_grouped_all = labels_long.groupby('poem_id')['top_type_id'].agg(lambda vals: sorted(set(map(str, vals))))
    labels_long_kept = labels_long[labels_long['top_type_id'].isin(keep_labels)].copy()
    labels_grouped_kept = labels_long_kept.groupby('poem_id')['top_type_id'].agg(lambda vals: sorted(set(map(str, vals))))
    poem_frames: dict[str, pd.DataFrame] = {}
    matrices: dict[str, np.ndarray] = {}
    coverage: dict[str, object] = {'min_train_support': int(min_train_support), 'num_labels': int(len(keep_labels)), 'total_primary_top_labels_train': int(len(train_counts)), 'excluded_type_prefixes': list(excluded_prefixes), 'splits': {}}
    for split_name in ('train', 'dev', 'test'):
        split_poems = poems[poems['split'] == split_name].copy()
        split_poems['all_primary_top_labels'] = split_poems['poem_id'].map(labels_grouped_all)
        split_poems['kept_primary_top_labels'] = split_poems['poem_id'].map(labels_grouped_kept)
        labeled_all = split_poems['all_primary_top_labels'].map(lambda value: isinstance(value, list) and bool(value))
        labeled_kept = split_poems['kept_primary_top_labels'].map(lambda value: isinstance(value, list) and bool(value))
        eval_poems = split_poems[labeled_kept].reset_index(drop=True)
        y = np.zeros((len(eval_poems), len(keep_labels)), dtype=np.float32)
        for (row_index, labels) in enumerate(eval_poems['kept_primary_top_labels']):
            for label in labels:
                y[row_index, label_to_idx[str(label)]] = 1.0
        poem_frames[split_name] = eval_poems
        matrices[split_name] = y
        coverage['splits'][split_name] = {'total_poems': int(len(split_poems)), 'poems_with_any_primary_top_label': int(labeled_all.sum()), 'poems_with_kept_primary_top_label': int(labeled_kept.sum()), 'primary_top_label_assignments': int(sum((len(v) for v in split_poems.loc[labeled_all, 'all_primary_top_labels']))), 'kept_primary_top_label_assignments': int(y.sum())}
    return LabelBundle(labels=keep_labels, label_names={label: name_map.get(label, '') for label in keep_labels}, poem_frames=poem_frames, y=matrices, coverage=coverage)

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    per_label_f1 = []
    per_label_precision = []
    per_label_recall = []
    label_support = y_true.sum(axis=0)
    for label_idx in np.flatnonzero(label_support > 0):
        true_col = y_true[:, label_idx]
        pred_col = y_pred[:, label_idx]
        col_tp = float(np.logical_and(true_col, pred_col).sum())
        col_fp = float(np.logical_and(~true_col, pred_col).sum())
        col_fn = float(np.logical_and(true_col, ~pred_col).sum())
        precision = col_tp / (col_tp + col_fp) if col_tp + col_fp else 0.0
        recall = col_tp / (col_tp + col_fn) if col_tp + col_fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label_precision.append(precision)
        per_label_recall.append(recall)
        per_label_f1.append(f1)
    return {'micro_precision': float(micro_precision), 'micro_recall': float(micro_recall), 'micro_f1': float(micro_f1), 'macro_precision': float(np.mean(per_label_precision) if per_label_precision else 0.0), 'macro_recall': float(np.mean(per_label_recall) if per_label_recall else 0.0), 'macro_f1': float(np.mean(per_label_f1) if per_label_f1 else 0.0), 'subset_accuracy': float(np.all(y_true == y_pred, axis=1).mean() if len(y_true) else 0.0), 'mean_predicted_labels': float(y_pred.sum(axis=1).mean() if len(y_pred) else 0.0), 'mean_true_labels': float(y_true.sum(axis=1).mean() if len(y_true) else 0.0)}

def hit_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    k = min(int(k), scores.shape[1])
    top_indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = []
    for (row_index, labels) in enumerate(top_indices):
        hits.append(bool(y_true[row_index, labels].sum() > 0))
    return float(np.mean(hits))

def evaluate_scores(*, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    probs = sigmoid(scores)
    predictions = probs >= float(threshold)
    metrics = binary_metrics(y_true, predictions)
    for k in (1, 3, 5):
        metrics[f'hit_at_{k}'] = hit_at_k(y_true, scores, k)
    metrics['threshold'] = float(threshold)
    metrics['num_poems'] = int(len(y_true))
    metrics['num_labels'] = int(y_true.shape[1])
    return metrics

def choose_threshold(y_true: np.ndarray, scores: np.ndarray, thresholds: Iterable[float] | None=None) -> tuple[float, dict[str, float]]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_f1 = -1.0
    for threshold in thresholds:
        metrics = evaluate_scores(y_true=y_true, scores=scores, threshold=float(threshold))
        score = float(metrics['micro_f1'])
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return (best_threshold, best_metrics)
