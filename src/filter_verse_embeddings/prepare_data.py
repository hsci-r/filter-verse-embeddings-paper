"""Build the Paper clean split from strict translation/source-position audit rows."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pandas as pd
REPO_ROOT = Path.cwd()
DEFAULT_AUDIT_PATH = Path('artifacts/data/cleaning/verse_position_audit.csv.gz')
DEFAULT_FROZEN_SPLIT_PATH = Path('artifacts/data/splits/poem_splits.parquet')
DEFAULT_OUTPUT_DIR = Path('artifacts/data')
AUDIT_COLUMNS = ['corpus', 'poem_id', 'pos', 'original_verse', 'english_translation', 'comments', 'translation_row_number', 'position_certainty', 'match_status', 'match_level', 'selection_rule', 'source_verse_type', 'matched_source_table', 'source_sent_to_llm', 'match_candidate_count']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-path', type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument('--frozen-splits-path', type=Path, default=DEFAULT_FROZEN_SPLIT_PATH)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--missing-pos-threshold', type=float, default=0.5, help='Drop poems whose missing-position row fraction is strictly greater than this value.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def file_info(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {'path': str(path), 'size_bytes': int(stat.st_size), 'mtime': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()}

def split_assignment_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for (poem_id, split) in frame[['poem_id', 'split']].drop_duplicates().sort_values('poem_id').itertuples(index=False):
        digest.update(str(poem_id).encode('utf-8'))
        digest.update(b'\t')
        digest.update(str(split).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()

def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    split_path = output_dir / 'paper_verses.parquet'
    if split_path.exists() and (not overwrite):
        raise FileExistsError(f'{split_path} exists; pass --overwrite to replace it')

def read_audit(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=lambda column: column in AUDIT_COLUMNS)
    missing_columns = sorted(set(AUDIT_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f'Audit file is missing required columns: {missing_columns}')
    frame['poem_id'] = frame['poem_id'].astype(str)
    frame['corpus'] = frame['corpus'].astype(str)
    frame['translation_row_number_num'] = pd.to_numeric(frame['translation_row_number'], errors='coerce').astype('Int64')
    frame['verse_position'] = pd.to_numeric(frame['pos'].replace('', pd.NA), errors='coerce').astype('Int64')
    frame['has_source_pos'] = frame['verse_position'].notna()
    return frame

def suspicious_poem_table(audit: pd.DataFrame, threshold: float) -> pd.DataFrame:
    grouped = audit.groupby('poem_id', sort=True).agg(source_collection=('corpus', 'first'), rows_total=('poem_id', 'size'), rows_missing_pos=('has_source_pos', lambda values: int((~values).sum())))
    grouped['missing_pos_fraction'] = grouped['rows_missing_pos'] / grouped['rows_total']
    grouped['is_suspicious'] = grouped['missing_pos_fraction'] > float(threshold)
    return grouped.reset_index()

def make_verse_ids(frame: pd.DataFrame) -> pd.Series:
    ids: list[str] = []
    for (row_index, row) in frame.reset_index(drop=True).iterrows():
        poem_id = str(row['poem_id'])
        if pd.notna(row['verse_position']):
            ids.append(f"{poem_id}_v{int(row['verse_position']):04d}")
            continue
        row_number = row.get('translation_row_number_num', pd.NA)
        suffix = int(row_number) if pd.notna(row_number) else int(row_index)
        ids.append(f'{poem_id}__unassigned__row{suffix}')
    return pd.Series(ids, index=frame.index, dtype='string')

def attach_frozen_splits(clean: pd.DataFrame, frozen_splits_path: Path) -> pd.DataFrame:
    frozen = pd.read_parquet(frozen_splits_path, columns=['poem_id', 'split'])
    frozen['poem_id'] = frozen['poem_id'].astype(str)
    poem_split = frozen[['poem_id', 'split']].drop_duplicates().copy()
    split_counts = poem_split.groupby('poem_id')['split'].nunique()
    leaking = sorted(split_counts[split_counts > 1].index.astype(str))
    if leaking:
        raise ValueError(f'Frozen split table has poems assigned to multiple splits: {leaking[:20]}')
    split_map = poem_split.drop_duplicates('poem_id').set_index('poem_id')['split']
    clean = clean.copy()
    clean['split'] = clean['poem_id'].map(split_map)
    missing_split = sorted(clean.loc[clean['split'].isna(), 'poem_id'].astype(str).unique())
    if missing_split:
        raise ValueError(f'Audit poems missing from Frozen split table: {missing_split[:20]}')
    return clean

def build_clean_split(audit: pd.DataFrame, frozen_splits_path: Path, *, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    suspicious = suspicious_poem_table(audit, threshold)
    suspicious_poems = set(suspicious.loc[suspicious['is_suspicious'], 'poem_id'].astype(str))
    clean = audit[~audit['poem_id'].isin(suspicious_poems)].copy()
    clean = attach_frozen_splits(clean, frozen_splits_path)
    clean['verse_id'] = make_verse_ids(clean)
    clean['source_collection'] = clean['corpus']
    clean['filter_text'] = clean['original_verse']
    clean['english_verse'] = clean['english_translation']
    clean['translation_comments'] = clean['comments']
    clean['paper_has_source_pos'] = clean['has_source_pos']
    clean['paper_position_source'] = clean['position_certainty'].where(clean['has_source_pos'], 'unassigned')
    assigned = clean[clean['verse_position'].notna()].copy()
    duplicated_pos = assigned.duplicated(['poem_id', 'verse_position'], keep=False)
    if bool(duplicated_pos.any()):
        examples = assigned.loc[duplicated_pos, ['poem_id', 'verse_position', 'filter_text']].head(20)
        raise ValueError(f'Duplicate assigned verse positions in cleaned data:\n{examples.to_string(index=False)}')
    duplicated_ids = clean['verse_id'].duplicated(keep=False)
    if bool(duplicated_ids.any()):
        examples = clean.loc[duplicated_ids, ['poem_id', 'verse_id', 'filter_text']].head(20)
        raise ValueError(f'Duplicate verse_id values in cleaned data:\n{examples.to_string(index=False)}')
    clean['paper_translation_row_number'] = clean['translation_row_number_num']
    clean = clean.sort_values(['source_collection', 'poem_id', 'verse_position', 'paper_translation_row_number', 'verse_id'], kind='stable', na_position='last').reset_index(drop=True)
    out_cols = ['poem_id', 'verse_id', 'verse_position', 'filter_text', 'english_verse', 'translation_comments', 'source_collection', 'split', 'paper_has_source_pos', 'paper_position_source', 'paper_translation_row_number', 'position_certainty', 'match_status', 'match_level', 'selection_rule', 'source_verse_type', 'matched_source_table', 'source_sent_to_llm', 'match_candidate_count']
    return (clean[out_cols], suspicious)

def write_poem_list(path: Path, poem_ids: list[str]) -> None:
    path.write_text(''.join((f'{poem_id}\n' for poem_id in poem_ids)), encoding='utf-8')

def write_outputs(clean: pd.DataFrame, suspicious: pd.DataFrame, *, output_dir: Path, audit_path: Path, frozen_splits_path: Path, threshold: float) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / 'paper_verses.parquet'
    clean.to_parquet(split_path, index=False)
    suspicious_path = output_dir / 'suspicious_poems.tsv'
    suspicious.to_csv(suspicious_path, sep='\t', index=False)
    retained_unassigned = clean[~clean['paper_has_source_pos']].copy()
    retained_unassigned_path = output_dir / 'retained_unassigned_rows.tsv'
    retained_unassigned.to_csv(retained_unassigned_path, sep='\t', index=False)
    for split_name in ('train', 'dev', 'test'):
        poem_ids = sorted(clean.loc[clean['split'] == split_name, 'poem_id'].astype(str).unique())
        write_poem_list(output_dir / f'{split_name}_poems.txt', poem_ids)
    row_counts = clean['split'].value_counts().sort_index().to_dict()
    poem_counts = clean.drop_duplicates('poem_id')['split'].value_counts().sort_index().to_dict()
    by_split_and_collection: dict[str, dict[str, dict[str, int]]] = {}
    for ((split, collection), rows) in clean.groupby(['split', 'source_collection'], sort=True):
        by_split_and_collection.setdefault(str(split), {})[str(collection)] = {'rows': int(len(rows)), 'poems': int(rows['poem_id'].nunique()), 'retained_unassigned_rows': int((~rows['paper_has_source_pos']).sum())}
    suspicious_only = suspicious[suspicious['is_suspicious']].copy()
    manifest = {'protocol': 'paper_clean_v1', 'created_at': datetime.now(timezone.utc).isoformat(), 'audit_input': file_info(audit_path), 'frozen_splits_input': file_info(frozen_splits_path), 'output_dir': str(output_dir), 'split_path': str(split_path), 'suspicious_poems_path': str(suspicious_path), 'retained_unassigned_rows_path': str(retained_unassigned_path), 'missing_pos_threshold': float(threshold), 'drop_rule': 'drop poem when missing_pos_fraction > threshold', 'rows_written': int(len(clean)), 'poems_written': int(clean['poem_id'].nunique()), 'retained_unassigned_rows': int((~clean['paper_has_source_pos']).sum()), 'suspicious_poems': int(len(suspicious_only)), 'suspicious_rows_total': int(suspicious_only['rows_total'].sum()), 'suspicious_missing_pos_rows': int(suspicious_only['rows_missing_pos'].sum()), 'rows_by_split': {str(k): int(v) for (k, v) in row_counts.items()}, 'poems_by_split': {str(k): int(v) for (k, v) in poem_counts.items()}, 'by_split_and_collection': by_split_and_collection, 'split_assignment_sha256': split_assignment_hash(clean)}
    manifest_path = output_dir / 'split_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    return manifest

def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.missing_pos_threshold) <= 1.0:
        raise ValueError('--missing-pos-threshold must be between 0 and 1')
    ensure_output_dir(args.output_dir, overwrite=args.overwrite)
    audit = read_audit(args.audit_path)
    (clean, suspicious) = build_clean_split(audit, args.frozen_splits_path, threshold=float(args.missing_pos_threshold))
    manifest = write_outputs(clean, suspicious, output_dir=args.output_dir, audit_path=args.audit_path, frozen_splits_path=args.frozen_splits_path, threshold=float(args.missing_pos_threshold))
    print(json.dumps(manifest, indent=2, sort_keys=True))
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        raise SystemExit(f'ERROR: {exc}') from None
