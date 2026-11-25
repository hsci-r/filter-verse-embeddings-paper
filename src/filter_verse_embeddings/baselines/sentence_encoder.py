"""Export an off-the-shelf SentenceTransformer baseline for retrieval evals."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.teachers.sentence import SentenceTeacher
from filter_verse_embeddings.data import load_config, load_split_table, split_frame
from filter_verse_embeddings.export_utils import l2_normalize, save_dense_embeddings, save_metadata

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Encode FILTER text and English translations with the same frozen SentenceTransformer and write the standard evaluator run layout.')
    parser.add_argument('--config', default='configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml')
    parser.add_argument('--verses-path', type=Path, default=None, help='Override config data.verses_path, useful for Phase-specific cleaned splits.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['test'])
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--no-cache', action='store_true', help='Do not read or write baseline embedding caches.')
    return parser.parse_args()

def safe_model_tag(model_name: str) -> str:
    readable = model_name.replace('/', '__').replace('-', '_').replace('.', '_')
    digest = hashlib.sha1(model_name.encode('utf-8')).hexdigest()[:8]
    return f'{readable}_{digest}'

def load_or_encode_side(*, teacher: SentenceTeacher, texts, cache_path: Path | None, batch_size: int, expected_rows: int | None=None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path).astype(np.float32, copy=False)
        if expected_rows is None or int(cached.shape[0]) == int(expected_rows):
            return cached
    embeddings = teacher.encode_texts(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f'.{cache_path.name}.tmp.npy')
        np.save(tmp_path, embeddings)
        tmp_path.replace(cache_path)
    return embeddings

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.verses_path is not None:
        config['data']['verses_path'] = str(args.verses_path)
    split_table = load_split_table(config['data']['verses_path'])
    model_tag = safe_model_tag(str(args.model_name))
    cache_dir = args.cache_dir
    if cache_dir is None and (not args.no_cache):
        cache_dir = Path('artifacts/outputs/teacher_cache') / 'sentence_transformer_baselines' / model_tag
    teacher = SentenceTeacher(str(args.model_name), device=str(args.device))
    export_summary: dict[str, dict[str, object]] = {}
    for split_name in args.splits:
        split_df = split_frame(split_table, split_name)
        side_specs = {'filter': config['data']['filter_text_column'], 'english': config['data']['english_text_column']}
        side_summaries: dict[str, object] = {}
        for (side, column) in side_specs.items():
            cache_path = None if args.no_cache or cache_dir is None else cache_dir / f'{split_name}_{side}.npy'
            embeddings = load_or_encode_side(teacher=teacher, texts=split_df[column], cache_path=cache_path, batch_size=int(args.batch_size), expected_rows=len(split_df))
            if embeddings.shape[0] != len(split_df):
                raise ValueError(f'Row mismatch for split={split_name} side={side}: embeddings={embeddings.shape[0]} rows={len(split_df)}')
            save_dense_embeddings(args.run_dir / 'embeddings' / f'{split_name}_{side}.npy', embeddings)
            side_summaries[side] = {'num_rows': int(embeddings.shape[0]), 'embedding_dim': int(embeddings.shape[1]), 'cache_path': str(cache_path) if cache_path is not None else None}
        save_metadata(args.run_dir / 'metadata', split_df.assign(split=split_name))
        export_summary[split_name] = side_summaries
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'metadata.json').write_text(json.dumps({'run_name': args.run_dir.name, 'model_type': 'sentence_transformer_baseline', 'sentence_transformer_model_name': str(args.model_name), 'baseline_interpretation': 'Off-the-shelf SentenceTransformer baseline: FILTER verses and English translations are embedded directly with the same frozen multilingual model. Local retrieval evaluates FILTER-side embeddings; translation retrieval evaluates cross-lingual FILTER-to-English alignment from the baseline model.', 'export_summary': export_summary}, indent=2, ensure_ascii=True), encoding='utf-8')
    print(f'Wrote SentenceTransformer baseline run to {args.run_dir}')
if __name__ == '__main__':
    main()
