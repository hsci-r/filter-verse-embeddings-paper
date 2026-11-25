"""Attach canonical FILTER/Allas verse ids to Paper verse metadata."""
from __future__ import annotations
import html
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
FLASH_ROOT = Path('artifacts')
DEFAULT_SOURCE_TABLE_DIR = Path('artifacts/data/metadata')
DEFAULT_VERSES_TRANSLATED_PATH = Path('artifacts/data/verses_translated.parquet')
DEFAULT_POEMS_PATH = DEFAULT_SOURCE_TABLE_DIR / 'poems.parquet'
DEFAULT_VERSES_CL_PATH = DEFAULT_SOURCE_TABLE_DIR / 'verses_cl.parquet'

def normalize_original_text(value: object) -> str:
    try:
        is_missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        is_missing = False
    if value is None or is_missing:
        text = ''
    else:
        text = html.unescape(str(value).lower())
    text = re.sub('amp(?:quot|apos|lt|gt|amp)', ' ', text)
    text = re.sub('&(?:quot|apos|lt|gt|amp);', ' ', text)
    return re.sub('\\W+', '', text, flags=re.UNICODE)

def normalized_text_codes(values: pd.Series) -> np.ndarray:
    normalized = values.map(normalize_original_text).to_numpy(dtype=object)
    (codes, _) = pd.factorize(normalized, sort=False)
    return codes.astype(np.int64, copy=False)

def read_canonical_mapping(verses_translated_path: str | Path=DEFAULT_VERSES_TRANSLATED_PATH, poems_path: str | Path=DEFAULT_POEMS_PATH, verses_cl_path: str | Path=DEFAULT_VERSES_CL_PATH) -> pd.DataFrame:
    vt = pd.read_parquet(verses_translated_path, columns=['p_id', 'pos', 'v_id'])
    vt['p_id'] = pd.to_numeric(vt['p_id'], errors='raise').astype('int64')
    vt['canonical_pos'] = pd.to_numeric(vt['pos'], errors='raise').astype('Int64')
    vt['canonical_v_id'] = pd.to_numeric(vt['v_id'], errors='raise').astype('int64')
    if vt.duplicated(['p_id', 'canonical_pos']).any():
        examples = vt[vt.duplicated(['p_id', 'canonical_pos'], keep=False)].head(10)[['p_id', 'canonical_pos']].to_dict('records')
        raise ValueError(f'Duplicate p_id/pos rows in verses_translated; examples={examples}')
    poems = pd.read_parquet(poems_path, columns=['p_id', 'nro'])
    poems['p_id'] = pd.to_numeric(poems['p_id'], errors='raise').astype('int64')
    poems['poem_id'] = poems['nro'].astype(str)
    if poems.duplicated(['poem_id']).any():
        examples = poems[poems.duplicated(['poem_id'], keep=False)].head(10)[['poem_id', 'p_id']].to_dict('records')
        raise ValueError(f'Duplicate poem_id rows in poems table; examples={examples}')
    verses_cl = pd.read_parquet(verses_cl_path, columns=['v_id', 'text'])
    verses_cl['canonical_v_id'] = pd.to_numeric(verses_cl['v_id'], errors='raise').astype('int64')
    verses_cl = verses_cl.rename(columns={'text': 'canonical_original_text'})[['canonical_v_id', 'canonical_original_text']]
    if verses_cl.duplicated(['canonical_v_id']).any():
        examples = verses_cl[verses_cl.duplicated(['canonical_v_id'], keep=False)].head(10)[['canonical_v_id']].to_dict('records')
        raise ValueError(f'Duplicate v_id rows in verses_cl table; examples={examples}')
    mapping = vt[['p_id', 'canonical_pos', 'canonical_v_id']].merge(poems[['p_id', 'poem_id']], on='p_id', how='left', validate='many_to_one').merge(verses_cl, on='canonical_v_id', how='left', validate='many_to_one')
    missing_poem = mapping['poem_id'].isna()
    if missing_poem.any():
        examples = mapping.loc[missing_poem].head(10)[['p_id', 'canonical_pos', 'canonical_v_id']].to_dict('records')
        raise ValueError(f'Missing poem_id for {int(missing_poem.sum())} translated rows; examples={examples}')
    if mapping.duplicated(['poem_id', 'canonical_pos']).any():
        examples = mapping[mapping.duplicated(['poem_id', 'canonical_pos'], keep=False)].head(10)[['poem_id', 'canonical_pos', 'canonical_v_id']].to_dict('records')
        raise ValueError(f'Duplicate poem_id/pos canonical rows; examples={examples}')
    return mapping[['poem_id', 'canonical_pos', 'p_id', 'canonical_v_id', 'canonical_original_text']].rename(columns={'p_id': 'canonical_p_id'})

def attach_canonical_verse_ids(frame: pd.DataFrame, *, verses_translated_path: str | Path=DEFAULT_VERSES_TRANSLATED_PATH, poems_path: str | Path=DEFAULT_POEMS_PATH, verses_cl_path: str | Path=DEFAULT_VERSES_CL_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = frame.copy()
    working['poem_id'] = working['poem_id'].astype(str)
    working['canonical_pos'] = pd.to_numeric(working['verse_position'], errors='coerce').astype('Int64')
    mapping = read_canonical_mapping(verses_translated_path, poems_path, verses_cl_path)
    merged = working.merge(mapping, on=['poem_id', 'canonical_pos'], how='left', validate='many_to_one')
    missing_pos = merged['canonical_pos'].isna()
    missing_v_id = merged['canonical_v_id'].isna()
    missing_original_text = merged['canonical_original_text'].isna()
    if 'filter_text' in merged.columns:
        merged['canonical_original_text'] = merged['canonical_original_text'].fillna(merged['filter_text'])
    merged['canonical_original_text'] = merged['canonical_original_text'].fillna('').astype(str)
    examples = merged.loc[missing_v_id, ['poem_id', 'verse_position', 'verse_id']].head(10).to_dict('records')
    summary: dict[str, Any] = {'verses_translated_path': str(verses_translated_path), 'poems_path': str(poems_path), 'verses_cl_path': str(verses_cl_path), 'total_rows': int(len(merged)), 'rows_missing_position': int(missing_pos.sum()), 'rows_with_canonical_v_id': int((~missing_v_id).sum()), 'rows_missing_canonical_v_id': int(missing_v_id.sum()), 'rows_missing_canonical_original_text_before_fallback': int(missing_original_text.sum()), 'missing_canonical_examples': examples}
    return (merged, summary)
