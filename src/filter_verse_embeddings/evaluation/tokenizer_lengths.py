"""Summarize tokenizer tokens-per-verse by split."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable
import numpy as np
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.data import HFTokenizer, SentencePieceTokenizer, load_split_table, split_frame

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verses-path', type=Path, default=Path('artifacts/data/paper_verses.parquet'))
    parser.add_argument('--text-column', default='filter_text')
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    parser.add_argument('--standard-tokenizer', type=Path, required=True)
    parser.add_argument('--versebal-tokenizer', type=Path, required=True)
    parser.add_argument('--hf-model-name', default='EMBEDDIA/finest-bert')
    parser.add_argument('--chunk-size', type=int, default=8192)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    return parser.parse_args()

def quantile(lengths: np.ndarray, q: float) -> float:
    if lengths.size == 0:
        return float('nan')
    return float(np.quantile(lengths, q))

def token_length_stats(lengths: np.ndarray) -> dict[str, float | int]:
    if lengths.size == 0:
        return {'token_mean': float('nan'), 'token_median': float('nan'), 'token_p90': float('nan'), 'token_p95': float('nan'), 'token_p99': float('nan'), 'token_max': 0, 'empty_verses': 0}
    return {'token_mean': float(lengths.mean()), 'token_median': quantile(lengths, 0.5), 'token_p90': quantile(lengths, 0.9), 'token_p95': quantile(lengths, 0.95), 'token_p99': quantile(lengths, 0.99), 'token_max': int(lengths.max()), 'empty_verses': int((lengths == 0).sum())}

def encode_lengths(texts: list[str], tokenizer: SentencePieceTokenizer | HFTokenizer, *, chunk_size: int) -> np.ndarray:
    if isinstance(tokenizer, HFTokenizer):
        lengths = np.empty(len(texts), dtype=np.int32)
        hf_tokenizer = tokenizer.hf_tokenizer
        for start in range(0, len(texts), int(chunk_size)):
            end = min(start + int(chunk_size), len(texts))
            encoded = hf_tokenizer(texts[start:end], add_special_tokens=False, padding=False, truncation=False)
            lengths[start:end] = [len(input_ids) for input_ids in encoded['input_ids']]
        return lengths
    return np.fromiter((len(tokenizer.encode_text(text)) for text in texts), dtype=np.int32, count=len(texts))

def tokenizer_specs(standard_tokenizer: Path, versebal_tokenizer: Path, hf_model_name: str) -> list[tuple[str, Callable[[], SentencePieceTokenizer | HFTokenizer]]]:
    return [('standard_sp32k', lambda : SentencePieceTokenizer(str(standard_tokenizer))), ('versebal_sp32k', lambda : SentencePieceTokenizer(str(versebal_tokenizer))), ('finest_bert_wp', lambda : HFTokenizer(str(hf_model_name)))]

def format_number(value: object, digits: int=2) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ''
        return f'{value:.{digits}f}'
    if isinstance(value, int):
        return f'{value:,}'
    return str(value)

def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    columns = [('split', 'Split'), ('tokenizer', 'Tokenizer'), ('vocab_size', 'Vocab'), ('rows', 'Rows'), ('token_mean', 'Mean tok/verse'), ('token_median', 'Median'), ('token_p95', 'P95'), ('token_p99', 'P99'), ('token_max', 'Max'), ('empty_verses', 'Empty')]
    lines = ['# Paper Tokenizer Length Stats', '', 'Token counts are per verse and exclude BOS/EOS, padding, and packed-window separators.', '', '| ' + ' | '.join((label for (_, label) in columns)) + ' |', '| ' + ' | '.join(('---' for _ in columns)) + ' |']
    for row in rows:
        cells = []
        for (key, _) in columns:
            digits = 3 if key == 'token_mean' else 2
            cells.append(format_number(row[key], digits=digits))
        lines.append('| ' + ' | '.join(cells) + ' |')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main() -> None:
    args = parse_args()
    split_table = load_split_table(args.verses_path)
    rows: list[dict[str, object]] = []
    for (tokenizer_label, make_tokenizer) in tokenizer_specs(args.standard_tokenizer, args.versebal_tokenizer, str(args.hf_model_name)):
        tokenizer_start = time.time()
        tokenizer = make_tokenizer()
        if isinstance(tokenizer, HFTokenizer):
            tokenizer.hf_tokenizer.model_max_length = 10 ** 12
        vocab_size = int(tokenizer.vocab_size)
        for split_name in args.splits:
            split_start = time.time()
            split_df = split_frame(split_table, split_name)
            if args.text_column not in split_df.columns:
                raise KeyError(f'Missing text column {args.text_column!r} in {args.verses_path}')
            texts = ['' if value is None else str(value) for value in split_df[args.text_column].fillna('')]
            lengths = encode_lengths(texts, tokenizer, chunk_size=int(args.chunk_size))
            row: dict[str, object] = {'split': str(split_name), 'tokenizer': tokenizer_label, 'vocab_size': vocab_size, 'rows': int(len(split_df)), 'poems': int(split_df['poem_id'].nunique()), **token_length_stats(lengths), 'elapsed_sec': round(time.time() - split_start, 2)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        print(json.dumps({'tokenizer': tokenizer_label, 'tokenizer_total_elapsed_sec': round(time.time() - tokenizer_start, 2)}, ensure_ascii=True), flush=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding='utf-8')
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print(f'Wrote {args.output_json}')
    print(f'Wrote {args.output_csv}')
    print(f'Wrote {args.output_md}')
if __name__ == '__main__':
    main()
