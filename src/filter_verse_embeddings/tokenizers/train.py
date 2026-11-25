"""Train a SentencePiece tokenizer on the FILTER train split only."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import sentencepiece as spm
import yaml
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a SentencePiece tokenizer from the split parquet.')
    parser.add_argument('--config', default='config/tokenizer.yaml', help='Tokenizer YAML config.')
    return parser.parse_args()

def load_config(config_path: str | Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as fin:
        return yaml.safe_load(fin)

def ensure_output_dir(config: dict) -> Path:
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def sanitize_training_text(text: str) -> str:
    return ' '.join(str(text).split())

def build_training_text(config: dict, output_dir: Path) -> tuple[Path, dict]:
    data_cfg = config['data']
    output_cfg = config['output']
    prebuilt_text_path = data_cfg.get('training_text_path')
    if prebuilt_text_path:
        text_path = Path(prebuilt_text_path)
        if not text_path.exists():
            raise FileNotFoundError(f'Prebuilt tokenizer training text not found: {text_path}')
        num_nonempty_rows = 0
        with text_path.open(encoding='utf-8') as fin:
            for line in fin:
                if line.strip():
                    num_nonempty_rows += 1
        if num_nonempty_rows == 0:
            raise ValueError(f'Prebuilt tokenizer training text is empty: {text_path}')
        stats = {'mode': 'prebuilt', 'training_text_path': str(text_path), 'num_nonempty_training_lines': int(num_nonempty_rows)}
        log.info('Using prebuilt tokenizer training text with %d non-empty lines', num_nonempty_rows)
        return (text_path, stats)
    verses_path = Path(data_cfg['verses_path'])
    split_name = data_cfg['split']
    text_column = data_cfg['text_column']
    log.info('Loading %s from %s', split_name, verses_path)
    df = pd.read_parquet(verses_path, columns=['split', text_column])
    split_df = df[df['split'] == split_name].copy()
    if split_df.empty:
        raise ValueError(f'No rows found for split={split_name!r}')
    if text_column not in split_df.columns:
        raise ValueError(f'Missing text column {text_column!r}')
    text_path = output_dir / output_cfg['training_text_filename']
    num_rows = len(split_df)
    num_nonempty_rows = 0
    with open(text_path, 'w', encoding='utf-8') as fout:
        for raw_text in split_df[text_column].fillna(''):
            clean_text = sanitize_training_text(raw_text)
            if not clean_text:
                continue
            fout.write(clean_text)
            fout.write('\n')
            num_nonempty_rows += 1
    stats = {'mode': 'split_dump', 'verses_path': str(verses_path), 'split': split_name, 'text_column': text_column, 'num_rows_in_split': int(num_rows), 'num_nonempty_training_lines': int(num_nonempty_rows), 'training_text_path': str(text_path)}
    log.info('Prepared tokenizer training text with %d non-empty lines', num_nonempty_rows)
    return (text_path, stats)

def build_trainer_kwargs(config: dict, text_path: Path, output_dir: Path) -> tuple[dict, Path]:
    trainer_cfg = dict(config['trainer'])
    output_cfg = config['output']
    model_prefix = output_dir / output_cfg['model_prefix']
    user_defined_symbols = trainer_cfg.pop('user_defined_symbols', [])
    trainer_kwargs = {'input': str(text_path), 'model_prefix': str(model_prefix), **trainer_cfg}
    if user_defined_symbols:
        trainer_kwargs['user_defined_symbols'] = list(user_defined_symbols)
    return (trainer_kwargs, model_prefix.with_suffix('.model'))

def export_vocab_files(model_path: Path, output_dir: Path, output_cfg: dict) -> dict:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    rows: list[dict] = []
    txt_path = output_dir / output_cfg['vocab_txt_filename']
    tsv_path = output_dir / output_cfg['vocab_tsv_filename']
    with open(txt_path, 'w', encoding='utf-8') as txt_out:
        for piece_id in range(sp.vocab_size()):
            piece = sp.id_to_piece(piece_id)
            txt_out.write(piece)
            txt_out.write('\n')
            rows.append({'id': piece_id, 'piece': piece, 'score': float(sp.get_score(piece_id)), 'is_unknown': bool(sp.is_unknown(piece_id)), 'is_control': bool(sp.is_control(piece_id)), 'is_unused': bool(sp.is_unused(piece_id)), 'is_byte': bool(sp.is_byte(piece_id))})
    pd.DataFrame(rows).to_csv(tsv_path, sep='\t', index=False)
    return {'model_path': str(model_path), 'vocab_path': str(model_path.with_suffix('.vocab')), 'vocab_txt_path': str(txt_path), 'vocab_tsv_path': str(tsv_path), 'vocab_size': int(sp.vocab_size()), 'special_token_ids': {'unk_id': int(sp.unk_id()), 'bos_id': int(sp.bos_id()), 'eos_id': int(sp.eos_id()), 'pad_id': int(sp.pad_id())}}

def save_resolved_config(config: dict, output_dir: Path) -> None:
    output_cfg = config['output']
    resolved_config_path = output_dir / output_cfg['resolved_config_filename']
    with open(resolved_config_path, 'w', encoding='utf-8') as fout:
        yaml.safe_dump(config, fout, sort_keys=False)

def save_metadata(config: dict, output_dir: Path, training_stats: dict, vocab_stats: dict) -> None:
    metadata = {'tokenizer_type': 'sentencepiece', 'config': config, 'training_text': training_stats, 'artifacts': vocab_stats}
    metadata_path = output_dir / config['output']['metadata_filename']
    with open(metadata_path, 'w', encoding='utf-8') as fout:
        json.dump(metadata, fout, indent=2, ensure_ascii=True)

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    save_resolved_config(config, output_dir)
    (text_path, training_stats) = build_training_text(config, output_dir)
    (trainer_kwargs, model_path) = build_trainer_kwargs(config, text_path, output_dir)
    log.info('Training SentencePiece tokenizer: type=%s vocab_size=%s', trainer_kwargs['model_type'], trainer_kwargs['vocab_size'])
    spm.SentencePieceTrainer.Train(**trainer_kwargs)
    vocab_stats = export_vocab_files(model_path, output_dir, config['output'])
    save_metadata(config, output_dir, training_stats, vocab_stats)
    log.info('Tokenizer artifacts written to %s', output_dir)
    log.info('Readable vocabulary list: %s', vocab_stats['vocab_tsv_path'])
if __name__ == '__main__':
    main()
