import argparse
import logging
from pathlib import Path
import pandas as pd
log = logging.getLogger(__name__)
CORPUS_DIR_MAP = {'skvr': 'skvr', 'erab': 'erab', 'jr': 'jr', 'kr': 'kr'}
WORD_EXPORT_COLUMNS = ['poem_id', 'original form', 'comment', 'English translation', 'normalized orthography', 'lemma (original)', 'lemma (modern)', 'etymological root']
WORD_RENAME_MAP = {'poem_id': 'poem_id', 'original form': 'original_form', 'comment': 'comment', 'English translation': 'english_translation', 'normalized orthography': 'normalized_orthography', 'lemma (original)': 'lemma_original', 'lemma (modern)': 'lemma_modern', 'etymological root': 'etymological_root'}

def find_corpus_csvs(input_dir: str) -> dict[str, Path]:
    found = {}
    for csv_path in sorted(Path(input_dir).rglob('final_0001.csv')):
        parts = csv_path.parts
        for part in parts:
            if part in CORPUS_DIR_MAP:
                corpus = CORPUS_DIR_MAP[part]
                if corpus in found:
                    log.warning('Duplicate CSV for corpus %s: %s (keeping first: %s)', corpus, csv_path, found[corpus])
                else:
                    found[corpus] = csv_path
                break
    return found

def load_word_corpus_csv(path: Path, corpus: str) -> pd.DataFrame:
    log.info('Loading word-analysis %s from %s', corpus, path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines='warn')
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in WORD_EXPORT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing expected columns in {path}: {missing}')
    df = df[WORD_EXPORT_COLUMNS].rename(columns=WORD_RENAME_MAP)
    df['source_collection'] = corpus
    return df

def assign_word_positions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['word_position'] = df.groupby('poem_id').cumcount() + 1
    df['word_id'] = df['poem_id'] + '_w' + df['word_position'].astype(str).str.zfill(5)
    return df

def build_word_analysis_table(input_dir: str, output_path: str) -> None:
    csv_map = find_corpus_csvs(input_dir)
    if not csv_map:
        raise FileNotFoundError(f'No word-level CSVs found under {input_dir}')
    log.info('Found %d word-analysis CSVs: %s', len(csv_map), list(csv_map.keys()))
    frames = []
    for (corpus, path) in sorted(csv_map.items()):
        frames.append(load_word_corpus_csv(path, corpus))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined['poem_id'].str.strip().astype(bool)].copy()
    combined = combined[combined['original_form'].str.strip().astype(bool)].copy()
    combined = combined.sort_values(['source_collection', 'poem_id']).reset_index(drop=True)
    combined = assign_word_positions(combined)
    canonical_cols = ['poem_id', 'word_id', 'word_position', 'original_form', 'comment', 'english_translation', 'normalized_orthography', 'lemma_original', 'lemma_modern', 'etymological_root', 'source_collection']
    combined = combined[canonical_cols]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    log.info('Wrote %s (%d rows)', output_path, len(combined))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the original ordered word-analysis training table.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    build_word_analysis_table(args.input_dir, args.output_path)
