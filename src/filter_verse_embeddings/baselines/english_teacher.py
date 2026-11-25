"""Export an English-teacher-only baseline in the standard evaluator format."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.teachers.sentence import SentenceTeacher, load_or_compute_teacher_embeddings
from filter_verse_embeddings.data import load_config, load_split_table, split_frame
from filter_verse_embeddings.export_utils import l2_normalize, save_dense_embeddings, save_metadata

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export English translation embeddings from the fixed sentence teacher as both evaluator sides. Local retrieval is then English-only; translation retrieval is an identity sanity check.')
    parser.add_argument('--config', default='configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml')
    parser.add_argument('--verses-path', type=Path, default=None, help='Override config data.verses_path with a frozen input snapshot.')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['test'])
    parser.add_argument('--teacher-device', default='auto')
    parser.add_argument('--teacher-cache-dir', type=Path, default=None)
    parser.add_argument('--no-compute-english', action='store_true')
    parser.add_argument('--copy-cache', action='store_true', help='Copy arrays instead of hard-linking cache files into the run directory.')
    return parser.parse_args()

def teacher_sentence_model_name(config: dict) -> str:
    teacher_config = config.get('teacher', {})
    model_name = teacher_config.get('sentence_model_name') or teacher_config.get('model_name')
    if not model_name:
        raise KeyError('teacher.sentence_model_name is required.')
    return str(model_name)

def link_or_save_embeddings(output_path: Path, embeddings: np.ndarray, source_path: Path | None, copy_cache: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path is not None and source_path.exists() and (not copy_cache):
        try:
            if output_path.exists():
                output_path.unlink()
            os.link(source_path, output_path)
            return
        except OSError:
            pass
    save_dense_embeddings(output_path, embeddings)

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.verses_path is not None:
        config['data']['verses_path'] = str(args.verses_path)
    split_table = load_split_table(config['data']['verses_path'])
    teacher_name = teacher_sentence_model_name(config)
    teacher_cache_dir = Path(args.teacher_cache_dir or str(config['teacher'].get('cache_dir') or args.run_dir / 'teacher_cache'))
    teacher: SentenceTeacher | None = None
    export_summary: dict[str, dict[str, object]] = {}
    for split_name in args.splits:
        split_df = split_frame(split_table, split_name)
        cache_path = teacher_cache_dir / f'{split_name}.npy'
        if cache_path.exists() and args.no_compute_english:
            english_embeddings = np.load(cache_path).astype(np.float32, copy=False)
            source_path: Path | None = cache_path
        elif args.no_compute_english:
            raise FileNotFoundError(f'English teacher cache missing: {cache_path}')
        else:
            if teacher is None:
                teacher = SentenceTeacher(teacher_name, device=args.teacher_device if args.teacher_device != 'auto' else str(config['teacher'].get('device', 'auto')))
            english_embeddings = load_or_compute_teacher_embeddings(cache_path, teacher, split_df[config['data']['english_text_column']], batch_size=int(config['teacher'].get('batch_size', 128)), normalize_embeddings=True, show_progress_bar=True, expected_rows=len(split_df))
            source_path = cache_path if cache_path.exists() else None
        if english_embeddings.shape[0] != len(split_df):
            raise ValueError(f'Cache row mismatch for split={split_name}: embeddings={english_embeddings.shape[0]} rows={len(split_df)}')
        english_embeddings = l2_normalize(np.asarray(english_embeddings, dtype=np.float32))
        embeddings_dir = args.run_dir / 'embeddings'
        link_or_save_embeddings(embeddings_dir / f'{split_name}_filter.npy', english_embeddings, source_path, bool(args.copy_cache))
        link_or_save_embeddings(embeddings_dir / f'{split_name}_english.npy', english_embeddings, source_path, bool(args.copy_cache))
        split_summary = {'num_rows': int(len(split_df)), 'embedding_dim': int(english_embeddings.shape[1]), 'source_cache': str(source_path) if source_path is not None else None}
        save_metadata(args.run_dir / 'metadata', split_df.assign(split=split_name))
        export_summary[split_name] = split_summary
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'metadata.json').write_text(json.dumps({'run_name': args.run_dir.name, 'model_type': 'english_teacher_baseline', 'teacher_model_name': teacher_name, 'baseline_interpretation': 'English-only baseline: fixed teacher embeddings of English translations are used as FILTER-side embeddings for local retrieval and as both sides for translation retrieval. Translation retrieval is therefore an identity sanity check, not a cross-lingual score.', 'export_summary': export_summary}, indent=2, ensure_ascii=True), encoding='utf-8')
    print(f'Wrote English-teacher baseline run to {args.run_dir}')
if __name__ == '__main__':
    main()
