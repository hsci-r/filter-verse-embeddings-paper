"""Export a raw HuggingFace encoder baseline for retrieval evals.

Uses AutoModel + AutoTokenizer with mean pooling over non-padding tokens.
Writes the standard evaluator run layout:
  run_dir/
    embeddings/{split}_filter.npy
    embeddings/{split}_english.npy
    metadata/{split}_metadata.parquet
    metadata.json"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.data import load_config, load_split_table, split_frame
from filter_verse_embeddings.export_utils import l2_normalize, save_dense_embeddings, save_metadata

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml')
    p.add_argument('--verses-path', type=Path, default=None, help='Override config data.verses_path, useful for Phase-specific cleaned splits.')
    p.add_argument('--model-name', required=True, help='HuggingFace model ID, e.g. EMBEDDIA/finest-bert')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--splits', nargs='+', default=['test'])
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--max-length', type=int, default=128)
    return p.parse_args()

def resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-08)

def encode_texts(texts, *, model: AutoModel, tokenizer: AutoTokenizer, batch_size: int, max_length: int, device: torch.device) -> np.ndarray:
    text_list = list(texts)
    all_embs: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(text_list), batch_size):
        batch = text_list[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc['attention_mask'])
        all_embs.append(emb.cpu().float().numpy())
        if start // batch_size % 50 == 0:
            done = min(start + batch_size, len(text_list))
            print(f'  encoded {done:,}/{len(text_list):,}', flush=True)
    return l2_normalize(np.vstack(all_embs).astype(np.float32))

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f'Loading {args.model_name} on {device} …')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    config = load_config(args.config)
    if args.verses_path is not None:
        config['data']['verses_path'] = str(args.verses_path)
    split_table = load_split_table(config['data']['verses_path'])
    filter_col = config['data']['filter_text_column']
    english_col = config['data']['english_text_column']
    export_summary: dict[str, dict[str, object]] = {}
    for split_name in args.splits:
        print(f'\n--- split: {split_name} ---')
        split_df = split_frame(split_table, split_name)
        side_summaries: dict[str, object] = {}
        for (side, column) in [('filter', filter_col), ('english', english_col)]:
            print(f'Encoding {side} side ({len(split_df):,} rows) …')
            embs = encode_texts(split_df[column], model=model, tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_length, device=device)
            if embs.shape[0] != len(split_df):
                raise ValueError(f'Row mismatch split={split_name} side={side}: embs={embs.shape[0]} rows={len(split_df)}')
            save_dense_embeddings(args.run_dir / 'embeddings' / f'{split_name}_{side}.npy', embs)
            side_summaries[side] = {'num_rows': int(embs.shape[0]), 'embedding_dim': int(embs.shape[1])}
        save_metadata(args.run_dir / 'metadata', split_df.assign(split=split_name))
        export_summary[split_name] = side_summaries
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'metadata.json').write_text(json.dumps({'run_name': args.run_dir.name, 'model_type': 'hf_mean_pool_baseline', 'model_name': str(args.model_name), 'baseline_interpretation': 'Raw HuggingFace encoder baseline: FILTER verses and English translations are embedded with the same frozen model using mean pooling over non-padding tokens. Local retrieval evaluates FILTER-side embeddings; translation retrieval evaluates cross-lingual FILTER-to-English alignment.', 'export_summary': export_summary}, indent=2, ensure_ascii=True), encoding='utf-8')
    print(f'\nWrote HF mean-pool baseline run to {args.run_dir}')
if __name__ == '__main__':
    main()
