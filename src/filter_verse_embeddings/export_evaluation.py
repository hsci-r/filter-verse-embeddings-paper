"""Export FILTER embeddings in the shared evaluator format."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.model import ScratchEncoderConfig, ScratchTransformerEncoder
from filter_verse_embeddings.teachers.sentence import SentenceTeacher, load_or_compute_teacher_embeddings
from filter_verse_embeddings.data import HFTokenizer, SentencePieceTokenizer, collate_token_sequences, load_config, load_split_table, save_json, split_frame
from filter_verse_embeddings.export_utils import l2_normalize, save_dense_embeddings, save_metadata

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export FILTER embeddings.')
    parser.add_argument('--config', default='configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--teacher-device', default='auto')
    parser.add_argument('--english-cache-dir', default=None, help='Directory with fixed English teacher embeddings named <split>.npy.')
    parser.add_argument('--no-compute-english', action='store_true', help='Fail instead of loading/running the English teacher if a cache file is missing.')
    return parser.parse_args()

def pick_device(device_name: str) -> torch.device:
    if device_name != 'auto':
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def resolve_checkpoint(checkpoint_override: str | None, run_dir: Path, checkpoint_dir: Path) -> Path:
    if checkpoint_override:
        path = Path(checkpoint_override)
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return path
    train_metadata = run_dir / 'train_metadata.json'
    if train_metadata.exists():
        payload = json.loads(train_metadata.read_text(encoding='utf-8'))
        last_checkpoint = payload.get('last_checkpoint')
        if last_checkpoint and Path(last_checkpoint).exists():
            return Path(last_checkpoint)
    candidates = sorted(checkpoint_dir.glob('epoch_*.pt'))
    if not candidates:
        raise FileNotFoundError(f'No checkpoints found in {checkpoint_dir}')
    return candidates[-1]

def encode_filter_texts(*, model: ScratchTransformerEncoder, projection: torch.nn.Linear, tokenizer: SentencePieceTokenizer, texts, max_length: int, batch_size: int, device: torch.device, add_bos: bool, add_eos: bool) -> np.ndarray:
    sequences: list[list[int]] = []
    reserved = int(add_bos) + int(add_eos)
    max_content_length = max_length - reserved
    for text in texts.fillna(''):
        piece_ids = tokenizer.encode_text(text)[:max_content_length]
        sequence: list[int] = []
        if add_bos:
            sequence.append(tokenizer.bos_id)
        sequence.extend(piece_ids)
        if add_eos:
            sequence.append(tokenizer.eos_id)
        sequences.append(sequence)
    loader = DataLoader(sequences, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_token_sequences(batch, tokenizer.pad_id))
    vectors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pooled = model(input_ids=input_ids, attention_mask=attention_mask, return_pooled=True)['pooled_embedding']
            pooled = projection(pooled)
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
    if not vectors:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return l2_normalize(np.concatenate(vectors, axis=0)).astype(np.float32)

def load_cached_embeddings(path: Path, *, expected_rows: int) -> np.ndarray | None:
    if not path.exists():
        return None
    embeddings = np.load(path).astype(np.float32, copy=False)
    if int(embeddings.shape[0]) != int(expected_rows):
        return None
    return embeddings

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = Path(config['output']['run_dir'])
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    device_name = args.device if args.device != 'auto' else str(config['training'].get('device', 'auto'))
    device = pick_device(device_name)
    checkpoint_path = resolve_checkpoint(args.checkpoint, run_dir, checkpoint_dir)
    if config['tokenizer'].get('hf_model_name'):
        tokenizer = HFTokenizer(config['tokenizer']['hf_model_name'])
    else:
        tokenizer = SentencePieceTokenizer(config['tokenizer']['model_path'])
    model_config = ScratchEncoderConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=int(config['tokenizer']['max_length']), num_layers=int(config['model']['num_layers']), hidden_size=int(config['model']['hidden_size']), num_heads=int(config['model']['num_heads']), ffn_dim=int(config['model']['ffn_dim']), dropout=float(config['model']['dropout']), pad_token_id=tokenizer.pad_id)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = ScratchTransformerEncoder(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    teacher_embedding_dim = int(checkpoint.get('teacher_embedding_dim', 768))
    projection = torch.nn.Linear(model_config.hidden_size, teacher_embedding_dim, bias=False)
    projection.load_state_dict(checkpoint['projection_state_dict'])
    projection.to(device)
    projection.eval()
    split_table = load_split_table(config['data']['verses_path'])
    export_summary: dict[str, dict[str, int]] = {}
    teacher_cache_dir = Path(args.english_cache_dir or str(config['teacher'].get('cache_dir') or run_dir / 'teacher_cache'))
    teacher: SentenceTeacher | None = None
    for split_name in config['output']['export_splits']:
        split_df = split_frame(split_table, split_name)
        filter_embeddings = encode_filter_texts(model=model, projection=projection, tokenizer=tokenizer, texts=split_df[config['data']['filter_text_column']], max_length=int(config['tokenizer']['max_length']), batch_size=int(config['output']['export_batch_size']), device=device, add_bos=bool(config['tokenizer']['add_bos']), add_eos=bool(config['tokenizer']['add_eos']))
        english_cache_path = teacher_cache_dir / f'{split_name}.npy'
        expected_english_rows = len(split_df)
        cached_english = load_cached_embeddings(english_cache_path, expected_rows=expected_english_rows)
        if cached_english is not None:
            english_embeddings = cached_english
        else:
            legacy_cache_path = teacher_cache_dir / f'{split_name}_english.npy'
            cached_legacy = load_cached_embeddings(legacy_cache_path, expected_rows=expected_english_rows)
            if cached_legacy is not None:
                english_embeddings = cached_legacy
            elif args.no_compute_english:
                raise FileNotFoundError(f'English embedding cache missing or wrong row count for split={split_name!r}; expected {expected_english_rows} rows at {english_cache_path} or {legacy_cache_path}')
            else:
                if teacher is None:
                    teacher = SentenceTeacher(config['teacher']['sentence_model_name'], device=args.teacher_device if args.teacher_device != 'auto' else str(config['teacher'].get('device', 'auto')))
                english_embeddings = load_or_compute_teacher_embeddings(english_cache_path, teacher, split_df[config['data']['english_text_column']], batch_size=int(config['teacher'].get('batch_size', 128)), show_progress_bar=True, expected_rows=expected_english_rows)
        save_dense_embeddings(run_dir / 'embeddings' / f'{split_name}_filter.npy', filter_embeddings)
        save_dense_embeddings(run_dir / 'embeddings' / f'{split_name}_english.npy', l2_normalize(english_embeddings))
        split_summary = {'num_rows': int(len(split_df))}
        save_metadata(run_dir / 'metadata', split_df.assign(split=split_name))
        export_summary[split_name] = split_summary
    save_json({'run_name': config['run_name'], 'model_type': 'scr_mlm_v', 'checkpoint_path': str(checkpoint_path), 'teacher_model_name': config['teacher']['sentence_model_name'], 'export_summary': export_summary}, run_dir / 'metadata.json')
if __name__ == '__main__':
    main()
