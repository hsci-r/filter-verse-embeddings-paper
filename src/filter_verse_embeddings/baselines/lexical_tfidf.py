"""Build the lexical TF-IDF run artifacts used by the evaluators."""
from __future__ import annotations
import argparse
import json
import logging
import pickle
import sys
from itertools import chain
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.baselines.text_normalization import AggressiveNormalizer, build_allowed_chars, iter_texts
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)
DEFAULT_METADATA_COLUMNS = ['verse_id', 'poem_id', 'verse_position', 'split', 'source_collection']

def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {'root': output_dir, 'embeddings': output_dir / 'embeddings', 'metadata': output_dir / 'metadata', 'debug': output_dir / 'debug'}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

def matrix_path(embeddings_dir: Path, split: str, side: str) -> Path:
    return embeddings_dir / f'{split}_{side}.npz'

def batched_texts(series: pd.Series, batch_size: int):
    batch: list[str] = []
    for text in iter_texts(series):
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def transform_series(vectorizer: TfidfVectorizer, series: pd.Series, batch_size: int) -> sparse.csr_matrix:
    matrices = [vectorizer.transform(batch) for batch in batched_texts(series, batch_size)]
    if not matrices:
        return sparse.csr_matrix((0, len(vectorizer.vocabulary_)), dtype='float32')
    if len(matrices) == 1:
        return matrices[0].tocsr()
    return sparse.vstack(matrices, format='csr')

def save_split_metadata(df: pd.DataFrame, split: str, metadata_dir: Path) -> None:
    extra_cols = [column for column in ['filter_text', 'english_verse'] if column in df.columns]
    columns = DEFAULT_METADATA_COLUMNS + extra_cols
    df[columns].reset_index(drop=True).to_parquet(metadata_dir / f'{split}_metadata.parquet', index=False)

def save_preprocessing_examples(df: pd.DataFrame, split: str, normalizer: AggressiveNormalizer, debug_dir: Path, example_count: int) -> None:
    sample_df = df.head(example_count).copy()
    sample_df['filter_text_normalized'] = sample_df['filter_text'].map(normalizer)
    sample_df['english_verse_normalized'] = sample_df['english_verse'].map(normalizer)
    columns = ['verse_id', 'poem_id', 'verse_position', 'filter_text', 'filter_text_normalized', 'english_verse', 'english_verse_normalized']
    sample_df[columns].to_csv(debug_dir / f'{split}_preprocessing_examples.csv', index=False)

def build_config(args: argparse.Namespace, allowed_chars: set[str]) -> dict:
    return {'run_name': args.run_name, 'model_type': 'lexical_tfidf', 'baseline_interpretation': 'FILTER-to-English retrieval is surface-overlap only for this baseline; intended as a lower bound / debug baseline.', 'data': {'verses_path': args.verses_path}, 'normalization': {'unicode_form': 'NFKD', 'lowercase': True, 'strip_diacritics': True, 'replace_non_alnum_with_space': True, 'collapse_whitespace': True, 'min_char_count': args.min_char_count, 'num_allowed_chars': len(allowed_chars)}, 'vectorizer': {'vectorizer_type': 'sklearn_tfidf', 'analyzer': 'char', 'ngram_range': [args.ngram_min, args.ngram_max], 'dtype': 'float32', 'min_ngram_doc_freq': args.min_ngram_doc_freq, 'transform_batch_size': args.transform_batch_size}, 'debug': {'preprocessing_example_count': args.debug_example_count}}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a lexical char n-gram TF-IDF baseline.')
    parser.add_argument('--verses_path', default='data/splits/verses_with_splits.parquet', help='Split-annotated verse parquet.')
    parser.add_argument('--output_dir', default='outputs/runs/lex', help='Directory for saved run artifacts.')
    parser.add_argument('--run_name', default='lex')
    parser.add_argument('--ngram_min', type=int, default=3)
    parser.add_argument('--ngram_max', type=int, default=5)
    parser.add_argument('--min_char_count', type=int, default=25, help='Drop characters seen fewer than this many times in train text.')
    parser.add_argument('--min_ngram_doc_freq', type=int, default=2, help='Drop n-grams seen in fewer than this many training documents.')
    parser.add_argument('--transform_batch_size', type=int, default=50000, help='Number of documents to vectorize per sparse batch.')
    parser.add_argument('--debug_example_count', type=int, default=20, help='Number of preprocessing examples to save per split.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    paths = ensure_dirs(output_dir)
    log.info('Loading split table from %s', args.verses_path)
    df = pd.read_parquet(args.verses_path)
    required = {'verse_id', 'poem_id', 'verse_position', 'split', 'filter_text', 'english_verse', 'source_collection'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df['filter_text'] = df['filter_text'].fillna('')
    df['english_verse'] = df['english_verse'].fillna('')
    train_df = df[df['split'] == 'train']
    if train_df.empty:
        raise ValueError('No train rows found in split parquet.')
    (allowed_chars, train_char_counts) = build_allowed_chars(train_df=train_df, min_char_count=args.min_char_count)
    log.info('Allowed %d characters after rare-char filtering (min count=%d)', len(allowed_chars), args.min_char_count)
    normalizer = AggressiveNormalizer(allowed_chars)
    train_corpus = chain(iter_texts(train_df['filter_text']), iter_texts(train_df['english_verse']))
    log.info('Fitting TF-IDF vectorizer on train FILTER + English text')
    vectorizer = TfidfVectorizer(analyzer='char', preprocessor=normalizer, lowercase=False, ngram_range=(args.ngram_min, args.ngram_max), min_df=args.min_ngram_doc_freq, dtype=np.float32)
    vectorizer.fit(train_corpus)
    log.info('Vectorizer fitted with %d features', len(vectorizer.vocabulary_))
    for split in ['train', 'dev', 'test']:
        split_df = df[df['split'] == split].reset_index(drop=True)
        if split_df.empty:
            log.warning('Split %s is empty; skipping', split)
            continue
        log.info('Transforming %s split (%d verses)', split, len(split_df))
        filter_matrix = transform_series(vectorizer, split_df['filter_text'], args.transform_batch_size)
        english_matrix = transform_series(vectorizer, split_df['english_verse'], args.transform_batch_size)
        sparse.save_npz(matrix_path(paths['embeddings'], split, 'filter'), filter_matrix)
        sparse.save_npz(matrix_path(paths['embeddings'], split, 'english'), english_matrix)
        save_split_metadata(split_df, split, paths['metadata'])
        save_preprocessing_examples(split_df, split=split, normalizer=normalizer, debug_dir=paths['debug'], example_count=args.debug_example_count)
    with open(paths['root'] / 'config.yaml', 'w', encoding='utf-8') as fout:
        yaml.safe_dump(build_config(args, allowed_chars), fout, sort_keys=False)
    metadata = {'run_name': args.run_name, 'model_type': 'lexical_tfidf', 'baseline_interpretation': 'FILTER-to-English retrieval is surface-overlap only for this baseline; intended as a lower bound / debug baseline.', 'verses_path': args.verses_path, 'output_dir': str(output_dir), 'split_counts': {split: int((df['split'] == split).sum()) for split in ['train', 'dev', 'test']}, 'num_features': len(vectorizer.vocabulary_), 'ngram_range': [args.ngram_min, args.ngram_max], 'min_char_count': args.min_char_count, 'min_ngram_doc_freq': args.min_ngram_doc_freq, 'transform_batch_size': args.transform_batch_size, 'num_allowed_chars': len(allowed_chars), 'allowed_chars_preview': ''.join(sorted(allowed_chars))[:200], 'char_count_preview': sorted(({'char': char, 'count': count} for (char, count) in train_char_counts.items() if char.strip()), key=lambda item: (-item['count'], item['char']))[:50], 'artifacts': {'vectorizer': 'vectorizer.pkl', 'metadata_dir': 'metadata', 'embeddings_dir': 'embeddings', 'debug_dir': 'debug'}}
    with open(paths['root'] / 'metadata.json', 'w', encoding='utf-8') as fout:
        json.dump(metadata, fout, indent=2, ensure_ascii=True)
    with open(paths['root'] / 'vectorizer.pkl', 'wb') as fout:
        pickle.dump({'vectorizer': vectorizer, 'ngram_range': (args.ngram_min, args.ngram_max), 'min_char_count': args.min_char_count, 'min_ngram_doc_freq': args.min_ngram_doc_freq, 'transform_batch_size': args.transform_batch_size, 'allowed_chars': sorted(allowed_chars)}, fout)
    log.info('Wrote lexical baseline run to %s', output_dir)
if __name__ == '__main__':
    main()
