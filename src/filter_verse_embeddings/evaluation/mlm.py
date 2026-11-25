"""Cheap intrinsic MLM evaluation for FILTER checkpoints.

This is intentionally a stochastic fixed-seed test-loss probe, not a full
deterministic pseudo-perplexity pass. It samples BERT-style masks once over the
chosen split and reports mean MLM loss plus exp(loss) as a comparable proxy.
"""
from __future__ import annotations
import argparse
from functools import partial
import json
import math
from pathlib import Path
import random
import sys
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.losses import mlm_loss_and_count
from filter_verse_embeddings.masking import apply_mlm_mask
from filter_verse_embeddings.model import ScratchEncoderConfig, ScratchTransformerEncoder
from filter_verse_embeddings.data import HFTokenizer, SentencePieceTokenizer, TokenSequenceDataset, build_training_sequences, collate_token_sequences, load_config, load_split_table, split_frame

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=None, help='Training config. Defaults to the config embedded in the checkpoint.')
    parser.add_argument('--verses-path', type=Path, default=None, help='Override config data.verses_path.')
    parser.add_argument('--split', default='test')
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--mlm-probability', type=float, default=None)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    return parser.parse_args()

def resolve_device(device_name: str) -> torch.device:
    if device_name != 'auto':
        return torch.device(device_name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_eval_config(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if args.config is not None:
        config = load_config(args.config)
    else:
        if 'config' not in checkpoint:
            raise KeyError('Checkpoint has no embedded config; pass --config.')
        config = checkpoint['config']
    config.setdefault('training', {})
    config.setdefault('data', {})
    config.setdefault('tokenizer', {})
    config.setdefault('model', {})
    if args.verses_path is not None:
        config['data']['verses_path'] = str(args.verses_path)
    return config

def build_tokenizer(config: dict[str, Any]) -> HFTokenizer | SentencePieceTokenizer:
    tokenizer_config = config['tokenizer']
    if tokenizer_config.get('hf_model_name'):
        return HFTokenizer(str(tokenizer_config['hf_model_name']))
    return SentencePieceTokenizer(str(tokenizer_config['model_path']))

def build_model(config: dict[str, Any], tokenizer: HFTokenizer | SentencePieceTokenizer) -> ScratchTransformerEncoder:
    model_config = ScratchEncoderConfig(vocab_size=int(tokenizer.vocab_size), max_position_embeddings=int(config['tokenizer']['max_length']), num_layers=int(config['model']['num_layers']), hidden_size=int(config['model']['hidden_size']), num_heads=int(config['model']['num_heads']), ffn_dim=int(config['model']['ffn_dim']), dropout=float(config['model']['dropout']), pad_token_id=int(tokenizer.pad_id))
    return ScratchTransformerEncoder(model_config)

def special_token_ids(config: dict[str, Any], tokenizer: HFTokenizer | SentencePieceTokenizer) -> set[int]:
    return {int(tokenizer.pad_id), int(tokenizer.bos_id), int(tokenizer.eos_id), int(tokenizer.unk_id), int(tokenizer.piece_to_id(str(config['tokenizer']['mask_token'])))}

def main() -> None:
    args = parse_args()
    set_eval_seed(int(args.seed))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = load_eval_config(args, checkpoint)
    tokenizer = build_tokenizer(config)
    model = build_model(config, tokenizer)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    device = resolve_device(str(args.device))
    model.to(device)
    model.eval()
    split_table = load_split_table(config['data']['verses_path'])
    split_df = split_frame(split_table, str(args.split))
    total_split_rows = int(len(split_df))
    if args.max_rows is not None:
        split_df = split_df.head(int(args.max_rows)).reset_index(drop=True)
    (sequences, sequence_stats) = build_training_sequences(split_df=split_df, tokenizer=tokenizer, text_column=str(config['data']['filter_text_column']), sequence_mode=str(config['training'].get('sequence_mode', 'single_verse')), max_length=int(config['tokenizer']['max_length']), add_bos=bool(config['tokenizer'].get('add_bos', True)), add_eos=bool(config['tokenizer'].get('add_eos', True)), verse_sep_token=str(config['tokenizer'].get('verse_sep_token', '<VERSE_SEP>')))
    dataset = TokenSequenceDataset(sequences)
    batch_size = int(args.batch_size or config['training'].get('eval_batch_size', config['training'].get('batch_size', 64)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(args.num_workers), collate_fn=partial(collate_token_sequences, pad_id=int(tokenizer.pad_id)))
    mask_token_id = int(tokenizer.piece_to_id(str(config['tokenizer']['mask_token'])))
    mlm_probability = float(args.mlm_probability if args.mlm_probability is not None else config['training'].get('mlm_probability', 0.15))
    reserved_ids = special_token_ids(config, tokenizer)
    total_loss_sum = 0.0
    total_masked_tokens = 0
    total_batches = 0
    zero_mask_batches = 0
    with torch.no_grad():
        for (step_index, batch) in enumerate(loader, start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            (masked_inputs, labels) = apply_mlm_mask(input_ids=input_ids, attention_mask=attention_mask, mask_token_id=mask_token_id, vocab_size=int(tokenizer.vocab_size), mlm_probability=mlm_probability, special_token_ids=reserved_ids)
            outputs = model(masked_inputs, attention_mask, return_pooled=False)
            (loss_sum, token_count) = mlm_loss_and_count(outputs['mlm_logits'], labels)
            masked_count = int(token_count.item())
            if masked_count == 0:
                zero_mask_batches += 1
            total_loss_sum += float(loss_sum.item())
            total_masked_tokens += masked_count
            total_batches += 1
            if args.max_steps is not None and step_index >= int(args.max_steps):
                break
    if total_masked_tokens == 0:
        raise ValueError('No MLM tokens were sampled; increase rows/steps or mlm_probability.')
    mlm_loss = total_loss_sum / float(total_masked_tokens)
    pseudo_ppl = math.exp(mlm_loss) if mlm_loss < 700 else float('inf')
    report = {'metric_type': 'fixed_seed_stochastic_mlm_loss', 'checkpoint': str(args.checkpoint), 'checkpoint_epoch': int(checkpoint.get('epoch', -1)), 'checkpoint_global_step': int(checkpoint.get('global_step', -1)), 'config': str(args.config) if args.config is not None else 'checkpoint_embedded_config', 'verses_path': str(config['data']['verses_path']), 'split': str(args.split), 'total_split_rows': total_split_rows, 'evaluated_rows': int(len(split_df)), 'sequence_mode': str(config['training'].get('sequence_mode', 'single_verse')), 'num_sequences': int(len(sequences)), 'sequence_stats': sequence_stats, 'num_batches': int(total_batches), 'zero_mask_batches': int(zero_mask_batches), 'masked_token_count': int(total_masked_tokens), 'mlm_probability': mlm_probability, 'seed': int(args.seed), 'batch_size': int(batch_size), 'mlm_loss': float(mlm_loss), 'mlm_pseudo_ppl': float(pseudo_ppl)}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=True))
if __name__ == '__main__':
    main()
