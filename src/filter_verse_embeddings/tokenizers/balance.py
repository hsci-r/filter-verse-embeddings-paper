"""Build dialect-balanced SentencePiece training text for Frozen."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
DEFAULT_METADATA_ROOT = Path('artifacts/data/metadata')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Sample train poems equally by inferred dialect and write tokenizer text.')
    parser.add_argument('--verses-path', type=Path, default=Path('artifacts/data/paper_verses.parquet'))
    parser.add_argument('--poems-path', type=Path, default=DEFAULT_METADATA_ROOT / 'poems.parquet')
    parser.add_argument('--p-pl-path', type=Path, default=DEFAULT_METADATA_ROOT / 'p_pl.parquet')
    parser.add_argument('--place-dialects-path', type=Path, default=DEFAULT_METADATA_ROOT / 'place_dialects.parquet')
    parser.add_argument('--split', default='train')
    parser.add_argument('--text-column', default='filter_text')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-dialects', type=int, nargs='*', default=[32], help='Dialect IDs to exclude from tokenizer balancing; defaults to 32 (Skolt Sami, 6 train poems).')
    parser.add_argument('--target-draws-per-dialect', type=int, default=None, help='Override the default per-dialect draw target. In poem mode this is poem draws; in verse mode this is verse-line draws.')
    parser.add_argument('--balance-unit', choices=['poem', 'verse'], default='poem', help='Balance tokenizer training text by sampled poems or by sampled verse lines.')
    parser.add_argument('--single-dialect-places-only', action='store_true', help='Before building dialect pools, discard places linked to more than one distinct dialect. This affects tokenizer training text only; downstream model training can still use the full corpus.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--text-output-name', default='train_filter_text.txt')
    parser.add_argument('--manifest-output-name', default='balanced_poem_draws.tsv')
    parser.add_argument('--metadata-output-name', default='balanced_sampling_metadata.json')
    parser.add_argument('--dry-run', action='store_true', help='Report sampling metadata without writing tokenizer text or manifests.')
    return parser.parse_args()

def sanitize_training_text(text: object) -> str:
    """Flatten one verse into a single SentencePiece training line."""
    return ' '.join(str(text).split())

def read_poem_split(verses_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(verses_path, columns=['poem_id', 'split'])
    frame['poem_id'] = frame['poem_id'].astype(str)
    poem_split = frame.drop_duplicates(['poem_id', 'split']).copy()
    split_counts = poem_split.groupby('poem_id')['split'].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        raise ValueError(f'Poems assigned to multiple splits: {leaking.index[:20].tolist()}')
    return poem_split.drop_duplicates('poem_id').reset_index(drop=True)

def build_dialect_pools(train_poem_ids: set[str], poems: pd.DataFrame, p_pl: pd.DataFrame, place_dialects: pd.DataFrame, exclude_dialects: set[int], single_dialect_places_only: bool=False) -> tuple[dict[int, list[str]], dict[str, Any]]:
    """Return multi-member dialect pools keyed by d_id."""
    poem_bridge = poems[['p_id', 'nro']].rename(columns={'nro': 'poem_id'}).copy()
    poem_bridge['poem_id'] = poem_bridge['poem_id'].astype(str)
    place_dialects = place_dialects.copy()
    place_dialects['d_id'] = pd.to_numeric(place_dialects['d_id'], errors='coerce')
    place_dialects_with_dialect = place_dialects.dropna(subset=['d_id']).copy()
    place_dialects_with_dialect['d_id'] = place_dialects_with_dialect['d_id'].astype(int)
    place_dialect_counts = place_dialects_with_dialect.groupby('pl_id')['d_id'].nunique()
    multi_dialect_place_ids = set(place_dialect_counts[place_dialect_counts > 1].index)
    single_dialect_place_ids = set(place_dialect_counts[place_dialect_counts == 1].index)
    eligible_place_dialects = place_dialects_with_dialect
    if single_dialect_places_only:
        eligible_place_dialects = eligible_place_dialects[eligible_place_dialects['pl_id'].isin(single_dialect_place_ids)].copy()
    linked_all_places = poem_bridge.merge(p_pl[['p_id', 'pl_id']], on='p_id', how='left').merge(place_dialects_with_dialect[['pl_id', 'd_id']], on='pl_id', how='left')
    linked = poem_bridge.merge(p_pl[['p_id', 'pl_id']], on='p_id', how='left').merge(eligible_place_dialects[['pl_id', 'd_id']], on='pl_id', how='left')
    train_linked_all_places = linked_all_places[linked_all_places['poem_id'].isin(train_poem_ids)].copy()
    train_linked = linked[linked['poem_id'].isin(train_poem_ids)].copy()
    metadata_poems = set(poem_bridge['poem_id'])
    missing_metadata = sorted(train_poem_ids - metadata_poems)
    membership = train_linked.dropna(subset=['d_id'])[['poem_id', 'd_id']].drop_duplicates()
    membership['d_id'] = membership['d_id'].astype(int)
    excluded_membership = membership[membership['d_id'].isin(exclude_dialects)].copy()
    membership = membership[~membership['d_id'].isin(exclude_dialects)].copy()
    pools = {int(d_id): sorted(group['poem_id'].astype(str).unique().tolist()) for (d_id, group) in membership.groupby('d_id', sort=True)}
    known_dialect_poems = set(membership['poem_id'].astype(str))
    poems_with_place = set(train_linked_all_places.dropna(subset=['pl_id'])['poem_id'].astype(str))
    eligible_place_poems = set(train_linked.dropna(subset=['pl_id'])['poem_id'].astype(str))
    single_dialect_place_poems = set(train_linked_all_places[train_linked_all_places['pl_id'].isin(single_dialect_place_ids)].dropna(subset=['pl_id'])['poem_id'].astype(str))
    multi_dialect_place_poems = set(train_linked_all_places[train_linked_all_places['pl_id'].isin(multi_dialect_place_ids)].dropna(subset=['pl_id'])['poem_id'].astype(str))
    stats = {'num_train_poems': int(len(train_poem_ids)), 'num_train_poems_missing_metadata': int(len(missing_metadata)), 'missing_metadata_poem_ids_preview': missing_metadata[:20], 'num_train_poems_with_place': int(len(poems_with_place)), 'num_train_poems_without_place': int(len(train_poem_ids - poems_with_place)), 'single_dialect_places_only': bool(single_dialect_places_only), 'num_single_dialect_places': int(len(single_dialect_place_ids)), 'num_multi_dialect_places': int(len(multi_dialect_place_ids)), 'num_train_poems_with_single_dialect_place': int(len(single_dialect_place_poems)), 'num_train_poems_with_multi_dialect_place': int(len(multi_dialect_place_poems)), 'num_train_poems_with_eligible_place': int(len(eligible_place_poems)), 'num_train_poems_with_dialect': int(len(known_dialect_poems)), 'num_train_poems_without_dialect': int(len(train_poem_ids) - len(known_dialect_poems)), 'num_dialect_groups': int(len(pools)), 'pool_sizes': {str(d_id): int(len(poems_in_pool)) for (d_id, poems_in_pool) in pools.items()}, 'multi_member_poems': int(membership.groupby('poem_id')['d_id'].nunique().gt(1).sum()), 'excluded_dialects': sorted((int(d_id) for d_id in exclude_dialects)), 'excluded_dialect_memberships': {str(d_id): int(count) for (d_id, count) in excluded_membership.groupby('d_id')['poem_id'].nunique().items()}, 'policy': {'membership': 'single_dialect_places_only' if single_dialect_places_only else 'multi_member', 'ambiguous_places': 'discarded_before_tokenizer_balancing' if single_dialect_places_only else 'preserved_as_multi_member_dialect_links', 'excluded_dialects': 'removed from tokenizer balancing only', 'missing_dialect_train_poems': 'excluded_from_tokenizer_training_only', 'dialect_source': 'place_dialects.parquet d_id via poems.parquet and p_pl.parquet'}}
    return (pools, stats)

def sample_balanced_poem_draws(pools: dict[int, list[str]], *, seed: int, target_draws_per_dialect: int | None=None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not pools:
        raise ValueError('No non-empty dialect pools available for sampling')
    known_poems = set().union(*(set(pool) for pool in pools.values()))
    if target_draws_per_dialect is None:
        target_draws_per_dialect = math.ceil(len(known_poems) / len(pools))
    if target_draws_per_dialect <= 0:
        raise ValueError('--target-draws-per-dialect must be positive')
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    dialect_stats: dict[str, dict[str, Any]] = {}
    for d_id in sorted(pools):
        pool = np.array(sorted(pools[d_id]), dtype=object)
        replace = len(pool) < target_draws_per_dialect
        sampled = rng.choice(pool, size=target_draws_per_dialect, replace=replace)
        unique_sampled = len(set((str(poem_id) for poem_id in sampled.tolist())))
        dialect_stats[str(d_id)] = {'pool_size': int(len(pool)), 'draws': int(target_draws_per_dialect), 'sample_with_replacement': bool(replace), 'unique_sampled_poems': int(unique_sampled), 'duplicate_draws_within_dialect': int(target_draws_per_dialect - unique_sampled)}
        for (draw_index, poem_id) in enumerate(sampled.tolist()):
            rows.append({'d_id': int(d_id), 'draw_index': int(draw_index), 'poem_id': str(poem_id), 'pool_size': int(len(pool)), 'sample_with_replacement': bool(replace)})
    draw_frame = pd.DataFrame(rows)
    total_draws = int(len(draw_frame))
    unique_poems = int(draw_frame['poem_id'].nunique())
    stats = {'seed': int(seed), 'target_draws_per_dialect': int(target_draws_per_dialect), 'total_poem_draws': total_draws, 'unique_sampled_poems': unique_poems, 'duplicate_poem_draws': int(total_draws - unique_poems), 'duplicate_draw_rate': 0.0 if total_draws == 0 else float((total_draws - unique_poems) / total_draws), 'dialects': dialect_stats}
    return (draw_frame, stats)

def validate_draws_train_only(draw_frame: pd.DataFrame, poem_split: pd.DataFrame, split: str) -> None:
    split_by_poem = poem_split.set_index('poem_id')['split'].astype(str)
    selected = set(draw_frame['poem_id'].astype(str))
    missing = sorted(selected - set(split_by_poem.index))
    if missing:
        raise ValueError(f'Sampled poems missing from split table: {missing[:20]}')
    bad = sorted((poem_id for poem_id in selected if split_by_poem.loc[poem_id] != split))
    if bad:
        raise ValueError(f'Sampled non-{split} poems for tokenizer training: {bad[:20]}')

def write_training_text(verses_path: Path, draw_frame: pd.DataFrame, *, split: str, text_column: str, output_path: Path) -> dict[str, Any]:
    columns = ['poem_id', 'split', text_column]
    verses = pd.read_parquet(verses_path, columns=columns)
    verses['poem_id'] = verses['poem_id'].astype(str)
    train_verses = verses[verses['split'] == split].copy()
    selected = set(draw_frame['poem_id'].astype(str))
    missing_text = sorted(selected - set(train_verses['poem_id'].unique()))
    if missing_text:
        raise ValueError(f'Sampled poems missing train text rows: {missing_text[:20]}')
    poem_texts: dict[str, list[str]] = {}
    for (poem_id, group) in train_verses[train_verses['poem_id'].isin(selected)].groupby('poem_id', sort=False):
        cleaned = [sanitize_training_text(text) for text in group[text_column].fillna('')]
        poem_texts[str(poem_id)] = [text for text in cleaned if text]
    num_lines = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as fout:
        for poem_id in draw_frame['poem_id'].astype(str):
            for text in poem_texts.get(poem_id, []):
                fout.write(text)
                fout.write('\n')
                num_lines += 1
    return {'training_text_path': str(output_path), 'num_nonempty_training_lines': int(num_lines), 'num_sampled_poems_with_nonempty_text': int(sum((bool(poem_texts.get(pid)) for pid in selected)))}

def read_nonempty_train_verses(verses_path: Path, *, split: str, text_column: str) -> pd.DataFrame:
    columns = ['poem_id', 'split', text_column]
    verses = pd.read_parquet(verses_path, columns=columns).reset_index(drop=False).rename(columns={'index': 'source_verse_row'})
    verses['poem_id'] = verses['poem_id'].astype(str)
    train_verses = verses[verses['split'] == split].copy()
    train_verses['training_text'] = train_verses[text_column].fillna('').map(sanitize_training_text)
    train_verses = train_verses[train_verses['training_text'].astype(bool)].copy()
    return train_verses.reset_index(drop=True)

def sample_balanced_verse_draws(pools: dict[int, list[str]], train_verses: pd.DataFrame, *, seed: int, target_draws_per_dialect: int | None=None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not pools:
        raise ValueError('No non-empty dialect pools available for verse sampling')
    required_columns = {'poem_id', 'source_verse_row', 'training_text'}
    missing_columns = required_columns - set(train_verses.columns)
    if missing_columns:
        raise ValueError(f'train_verses missing columns: {sorted(missing_columns)}')
    known_poems = set().union(*(set(pool) for pool in pools.values()))
    known_verses = train_verses[train_verses['poem_id'].isin(known_poems)].copy()
    if known_verses.empty:
        raise ValueError('No non-empty train verses available for dialect verse sampling')
    if target_draws_per_dialect is None:
        target_draws_per_dialect = math.ceil(known_verses['source_verse_row'].nunique() / len(pools))
    if target_draws_per_dialect <= 0:
        raise ValueError('--target-draws-per-dialect must be positive')
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    dialect_stats: dict[str, dict[str, Any]] = {}
    for d_id in sorted(pools):
        pool_poems = set((str(poem_id) for poem_id in pools[d_id]))
        pool_verses = train_verses[train_verses['poem_id'].isin(pool_poems)].sort_values(['poem_id', 'source_verse_row'], kind='stable').reset_index(drop=True)
        if pool_verses.empty:
            raise ValueError(f'Dialect {d_id} has no non-empty train verses')
        source_rows = pool_verses['source_verse_row'].to_numpy(dtype=np.int64)
        replace = len(source_rows) < target_draws_per_dialect
        sampled_source_rows = rng.choice(source_rows, size=target_draws_per_dialect, replace=replace)
        verse_lookup = pool_verses.drop_duplicates('source_verse_row').set_index('source_verse_row')
        unique_sampled_verses = len(set((int(row_id) for row_id in sampled_source_rows.tolist())))
        unique_sampled_poems = int(verse_lookup.loc[list(set(sampled_source_rows.tolist())), 'poem_id'].nunique())
        dialect_stats[str(d_id)] = {'pool_size_verses': int(len(source_rows)), 'pool_size_poems': int(pool_verses['poem_id'].nunique()), 'draws': int(target_draws_per_dialect), 'sample_with_replacement': bool(replace), 'unique_sampled_verses': int(unique_sampled_verses), 'unique_sampled_poems': int(unique_sampled_poems), 'duplicate_draws_within_dialect': int(target_draws_per_dialect - unique_sampled_verses)}
        for (draw_index, source_verse_row) in enumerate(sampled_source_rows.tolist()):
            verse_row = verse_lookup.loc[int(source_verse_row)]
            rows.append({'d_id': int(d_id), 'draw_index': int(draw_index), 'poem_id': str(verse_row['poem_id']), 'source_verse_row': int(source_verse_row), 'pool_size_verses': int(len(source_rows)), 'sample_with_replacement': bool(replace)})
    draw_frame = pd.DataFrame(rows)
    total_draws = int(len(draw_frame))
    unique_verses = int(draw_frame['source_verse_row'].nunique())
    unique_poems = int(draw_frame['poem_id'].nunique())
    stats = {'seed': int(seed), 'target_draws_per_dialect': int(target_draws_per_dialect), 'total_verse_draws': total_draws, 'unique_sampled_verses': unique_verses, 'unique_sampled_poems': unique_poems, 'duplicate_verse_draws': int(total_draws - unique_verses), 'duplicate_verse_draw_rate': 0.0 if total_draws == 0 else float((total_draws - unique_verses) / total_draws), 'dialects': dialect_stats}
    return (draw_frame, stats)

def write_verse_training_text(train_verses: pd.DataFrame, draw_frame: pd.DataFrame, *, output_path: Path) -> dict[str, Any]:
    if 'source_verse_row' not in draw_frame.columns:
        raise ValueError('Verse draw frame must include source_verse_row')
    verse_text = train_verses.drop_duplicates('source_verse_row').set_index('source_verse_row')['training_text']
    missing = sorted(set(draw_frame['source_verse_row'].astype(int)) - set(verse_text.index.astype(int)))
    if missing:
        raise ValueError(f'Sampled verse rows missing from train text: {missing[:20]}')
    num_lines = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as fout:
        for source_verse_row in draw_frame['source_verse_row'].astype(int):
            text = str(verse_text.loc[int(source_verse_row)])
            if not text:
                continue
            fout.write(text)
            fout.write('\n')
            num_lines += 1
    return {'training_text_path': str(output_path), 'num_nonempty_training_lines': int(num_lines), 'num_unique_sampled_verses_with_nonempty_text': int(draw_frame['source_verse_row'].nunique())}

def build_metadata(args: argparse.Namespace, pool_stats: dict, sample_stats: dict, text_stats: dict | None) -> dict:
    return {'builder': 'tokenizer/build_dialect_balanced_training_text.py', 'inputs': {'verses_path': str(args.verses_path), 'poems_path': str(args.poems_path), 'p_pl_path': str(args.p_pl_path), 'place_dialects_path': str(args.place_dialects_path), 'split': args.split, 'text_column': args.text_column, 'exclude_dialects': sorted((int(d_id) for d_id in args.exclude_dialects)), 'single_dialect_places_only': bool(args.single_dialect_places_only), 'balance_unit': str(args.balance_unit)}, 'sampling': {**pool_stats, **sample_stats, 'corpus_size_policy': 'ceil(num_unique_known_dialect_train_verses / num_dialect_groups) verse draws per dialect' if args.balance_unit == 'verse' else 'ceil(num_train_poems_with_dialect / num_dialect_groups) poem draws per dialect', 'replacement_policy': 'only dialect pools smaller than target_draws_per_dialect sample with replacement'}, 'training_text': text_stats or {'dry_run': True}}

def main() -> None:
    args = parse_args()
    poem_split = read_poem_split(args.verses_path)
    train_poem_ids = set(poem_split[poem_split['split'] == args.split]['poem_id'].astype(str))
    if not train_poem_ids:
        raise ValueError(f'No poems found for split={args.split!r}')
    poems = pd.read_parquet(args.poems_path, columns=['p_id', 'nro'])
    p_pl = pd.read_parquet(args.p_pl_path, columns=['p_id', 'pl_id'])
    place_dialects = pd.read_parquet(args.place_dialects_path, columns=['pl_id', 'd_id'])
    (pools, pool_stats) = build_dialect_pools(train_poem_ids, poems, p_pl, place_dialects, exclude_dialects=set(args.exclude_dialects), single_dialect_places_only=bool(args.single_dialect_places_only))
    train_verses = None
    if args.balance_unit == 'verse':
        train_verses = read_nonempty_train_verses(args.verses_path, split=args.split, text_column=args.text_column)
        (draw_frame, sample_stats) = sample_balanced_verse_draws(pools, train_verses, seed=args.seed, target_draws_per_dialect=args.target_draws_per_dialect)
    else:
        (draw_frame, sample_stats) = sample_balanced_poem_draws(pools, seed=args.seed, target_draws_per_dialect=args.target_draws_per_dialect)
    validate_draws_train_only(draw_frame, poem_split, args.split)
    if args.dry_run:
        metadata = build_metadata(args, pool_stats, sample_stats, text_stats=None)
        print(json.dumps(metadata, indent=2, ensure_ascii=True))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text_path = args.output_dir / args.text_output_name
    manifest_path = args.output_dir / args.manifest_output_name
    metadata_path = args.output_dir / args.metadata_output_name
    draw_frame.to_csv(manifest_path, sep='\t', index=False)
    if args.balance_unit == 'verse':
        if train_verses is None:
            raise RuntimeError('Internal error: train_verses was not prepared for verse balancing')
        text_stats = write_verse_training_text(train_verses, draw_frame, output_path=text_path)
    else:
        text_stats = write_training_text(args.verses_path, draw_frame, split=args.split, text_column=args.text_column, output_path=text_path)
    text_stats['poem_draw_manifest_path'] = str(manifest_path)
    metadata = build_metadata(args, pool_stats, sample_stats, text_stats)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps(metadata['sampling'], indent=2, ensure_ascii=True))
    print(f'Wrote {text_path}')
    print(f'Wrote {manifest_path}')
    print(f'Wrote {metadata_path}')
if __name__ == '__main__':
    main()
