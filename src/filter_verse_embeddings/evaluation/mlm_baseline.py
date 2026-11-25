"""Cheap intrinsic MLM evaluation for HuggingFace masked-LM baselines."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.losses import mlm_loss_and_count
from filter_verse_embeddings.masking import apply_mlm_mask
from filter_verse_embeddings.data import load_split_table, split_frame

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--verses-path', type=Path, default=Path('artifacts/data/paper_verses.parquet'))
    parser.add_argument('--split', default='test')
    parser.add_argument('--text-column', default='filter_text')
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--mlm-probability', type=float, default=0.15)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--allow-non-mlm-architecture', action='store_true', help='Allow models whose config does not advertise *ForMaskedLM. Use only for deliberate probes, because the MLM head may be randomly initialized.')
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

def validate_mlm_architecture(model_name: str, *, allow_non_mlm: bool) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(model_name)
    architectures = list(getattr(config, 'architectures', []) or [])
    has_declared_mlm = any(('MaskedLM' in architecture for architecture in architectures))
    if not has_declared_mlm and (not allow_non_mlm):
        raise ValueError(f'{model_name!r} does not advertise a masked-LM architecture (architectures={architectures}). Use --allow-non-mlm-architecture only if you accept that the MLM head may not correspond to the retrieval baseline.')
    return {'model_type': getattr(config, 'model_type', None), 'architectures': architectures, 'has_declared_mlm': bool(has_declared_mlm)}

def special_token_ids(tokenizer) -> set[int]:
    ids = {tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id}
    return {int(token_id) for token_id in ids if token_id is not None}

def load_texts(verses_path: Path, split_name: str, text_column: str, max_rows: int | None) -> tuple[pd.DataFrame, list[str], int]:
    split_table = load_split_table(verses_path)
    split_df = split_frame(split_table, split_name)
    total_rows = int(len(split_df))
    if text_column not in split_df.columns:
        raise KeyError(f'Missing text column {text_column!r} in {verses_path}')
    if max_rows is not None:
        split_df = split_df.head(int(max_rows)).reset_index(drop=True)
    texts = ['' if value is None else str(value) for value in split_df[text_column].fillna('')]
    return (split_df, texts, total_rows)

def main() -> None:
    args = parse_args()
    set_eval_seed(int(args.seed))
    model_info = validate_mlm_architecture(str(args.model_name), allow_non_mlm=bool(args.allow_non_mlm_architecture))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name))
    if tokenizer.mask_token_id is None:
        raise ValueError(f'{args.model_name!r} tokenizer has no mask token.')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.mask_token
    device = resolve_device(str(args.device))
    model = AutoModelForMaskedLM.from_pretrained(str(args.model_name)).to(device)
    model.eval()
    (split_df, texts, total_rows) = load_texts(args.verses_path, str(args.split), str(args.text_column), args.max_rows)
    loader = DataLoader(texts, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    total_loss_sum = 0.0
    total_masked_tokens = 0
    total_batches = 0
    zero_mask_batches = 0
    reserved_ids = special_token_ids(tokenizer)
    with torch.no_grad():
        for (step_index, batch_texts) in enumerate(loader, start=1):
            encoded = tokenizer(list(batch_texts), padding=True, truncation=True, max_length=int(args.max_length), return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            (masked_inputs, labels) = apply_mlm_mask(input_ids=input_ids, attention_mask=attention_mask, mask_token_id=int(tokenizer.mask_token_id), vocab_size=int(tokenizer.vocab_size), mlm_probability=float(args.mlm_probability), special_token_ids=reserved_ids)
            outputs = model(input_ids=masked_inputs, attention_mask=attention_mask)
            (loss_sum, token_count) = mlm_loss_and_count(outputs.logits, labels)
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
    report = {'metric_type': 'hf_fixed_seed_stochastic_mlm_loss', 'model_name': str(args.model_name), 'model_info': model_info, 'verses_path': str(args.verses_path), 'split': str(args.split), 'text_column': str(args.text_column), 'total_split_rows': total_rows, 'evaluated_rows': int(len(split_df)), 'num_batches': int(total_batches), 'zero_mask_batches': int(zero_mask_batches), 'masked_token_count': int(total_masked_tokens), 'max_length': int(args.max_length), 'mlm_probability': float(args.mlm_probability), 'seed': int(args.seed), 'batch_size': int(args.batch_size), 'mlm_loss': float(mlm_loss), 'mlm_pseudo_ppl': float(math.exp(mlm_loss) if mlm_loss < 700 else float('inf'))}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=True))
if __name__ == '__main__':
    main()
