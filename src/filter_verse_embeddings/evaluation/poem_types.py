"""Frozen-representation multi-label poem-type probes."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import random
import re
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.evaluation.poem_type_data import DEFAULT_EXCLUDED_TYPE_PREFIXES, DEFAULT_POEM_TYPES_PATH, DEFAULT_TYPES_PATH, DEFAULT_VERSES_PATH, LabelBundle, choose_threshold, evaluate_scores, l2_normalize, load_primary_top_labels, write_json
from filter_verse_embeddings.model import ScratchEncoderConfig, ScratchTransformerEncoder
from filter_verse_embeddings.data import HFTokenizer, SentencePieceTokenizer, collate_token_sequences, load_config
from filter_verse_embeddings.dataset import PackedContrastiveWindowDataset, collate_packed_contrastive_batch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=None)
    parser.add_argument('--run-index', type=int, default=None)
    parser.add_argument('--target', choices=('trained', 'baselines', 'english_baselines', 'all'), default='all')
    parser.add_argument('--representations', nargs='+', default=['isolated', 'contextual'])
    parser.add_argument('--verses-path', type=Path, default=DEFAULT_VERSES_PATH)
    parser.add_argument('--types-path', type=Path, default=DEFAULT_TYPES_PATH)
    parser.add_argument('--poem-types-path', type=Path, default=DEFAULT_POEM_TYPES_PATH)
    parser.add_argument('--min-train-support', type=int, default=20)
    parser.add_argument('--exclude-type-prefixes', nargs='*', default=list(DEFAULT_EXCLUDED_TYPE_PREFIXES))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--feature-cache-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--classifier-max-iter', type=int, default=100)
    parser.add_argument('--classifier-alpha', type=float, default=0.0001)
    parser.add_argument('--classifier-n-jobs', type=int, default=8)
    parser.add_argument('--max-train-poems', type=int, default=None)
    parser.add_argument('--max-eval-poems', type=int, default=None)
    parser.add_argument('--hf-model-name', default='EMBEDDIA/finest-bert')
    parser.add_argument('--hf-max-length', type=int, default=512)
    parser.add_argument('--tfidf-min-df', type=int, default=2)
    parser.add_argument('--tfidf-ngram-min', type=int, default=3)
    parser.add_argument('--tfidf-ngram-max', type=int, default=5)
    parser.add_argument('--english-text-column', default='english_verse')
    parser.add_argument('--english-sentence-transformer-models', nargs='*', default=[], help='SentenceTransformer model names for translation-assisted English-side type probes.')
    parser.add_argument('--english-hf-models', nargs='*', default=[], help='HuggingFace encoder model names for translation-assisted English-side type probes.')
    parser.add_argument('--english-hf-max-length', type=int, default=128)
    parser.add_argument('--english-hf-poem-context-models', nargs='*', default=[], help='HuggingFace encoder model names for contextual English poem-level probes (all verses of a poem packed into one window, like the Finnish contextual baseline).')
    parser.add_argument('--english-hf-poem-context-max-length', type=int, default=512)
    parser.add_argument('--english-sentence-cache-dir', type=Path, default=None)
    return parser.parse_args()

def safe_id(value: str) -> str:
    return re.sub('[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')

def resolve_device(device_name: str) -> torch.device:
    if device_name != 'auto':
        return torch.device(device_name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def maybe_limit_bundle(bundle: LabelBundle, max_train_poems: int | None, max_eval_poems: int | None) -> LabelBundle:
    for split_name in ('train', 'dev', 'test'):
        limit = max_train_poems if split_name == 'train' else max_eval_poems
        if limit is None:
            continue
        bundle.poem_frames[split_name] = bundle.poem_frames[split_name].head(int(limit)).reset_index(drop=True)
        bundle.y[split_name] = bundle.y[split_name][:int(limit)]
    return bundle

def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    return rows

def load_split_verses(verses_path: Path, split_name: str, text_column: str) -> pd.DataFrame:
    verses = pd.read_parquet(verses_path, columns=['poem_id', 'verse_id', 'verse_position', 'split', 'source_collection', text_column])
    frame = verses[verses['split'] == split_name].copy()
    frame['poem_id'] = frame['poem_id'].astype(str)
    frame[text_column] = frame[text_column].fillna('').astype(str)
    return frame.sort_values(['poem_id', 'verse_position', 'verse_id'], kind='stable').reset_index(drop=True)

def load_relevant_verses(verses_path: Path, split_name: str, poem_ids: pd.Series, text_column: str='filter_text') -> pd.DataFrame:
    frame = load_split_verses(verses_path, split_name, text_column)
    keep = set(poem_ids.astype(str).tolist())
    return frame[frame['poem_id'].astype(str).isin(keep)].reset_index(drop=True)

def aggregate_by_poem(vectors: np.ndarray, poem_ids: list[str], target_poem_ids: list[str]) -> np.ndarray:
    dim = int(vectors.shape[1])
    index = {poem_id: row for (row, poem_id) in enumerate(target_poem_ids)}
    sums = np.zeros((len(target_poem_ids), dim), dtype=np.float32)
    counts = np.zeros(len(target_poem_ids), dtype=np.float32)
    for (vector, poem_id) in zip(vectors, poem_ids):
        row = index.get(str(poem_id))
        if row is None:
            continue
        sums[row] += vector.astype(np.float32, copy=False)
        counts[row] += 1.0
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        examples = [target_poem_ids[int(idx)] for idx in missing[:5]]
        raise ValueError(f'Missing encoded rows for {len(missing)} poems, examples={examples}')
    sums /= counts[:, None]
    return l2_normalize(sums)

def load_scratch_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = load_config(config_path)
    if config['tokenizer'].get('hf_model_name'):
        tokenizer = HFTokenizer(config['tokenizer']['hf_model_name'])
    else:
        tokenizer = SentencePieceTokenizer(config['tokenizer']['model_path'])
    model_config = ScratchEncoderConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=int(config['tokenizer']['max_length']), num_layers=int(config['model']['num_layers']), hidden_size=int(config['model']['hidden_size']), num_heads=int(config['model']['num_heads']), ffn_dim=int(config['model']['ffn_dim']), dropout=float(config['model']['dropout']), pad_token_id=tokenizer.pad_id)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = ScratchTransformerEncoder(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    teacher_dim = int(checkpoint.get('teacher_embedding_dim', 768))
    projection = torch.nn.Linear(model_config.hidden_size, teacher_dim, bias=False)
    projection.load_state_dict(checkpoint['projection_state_dict'])
    projection.to(device)
    projection.eval()
    return (config, tokenizer, model, projection)

def encode_scratch_isolated(*, verses: pd.DataFrame, target_poem_ids: list[str], config: dict, tokenizer, model: ScratchTransformerEncoder, projection: torch.nn.Linear, device: torch.device, batch_size: int) -> np.ndarray:
    reserved = int(bool(config['tokenizer']['add_bos'])) + int(bool(config['tokenizer']['add_eos']))
    max_content = int(config['tokenizer']['max_length']) - reserved
    sequences: list[list[int]] = []
    for text in verses[config['data']['filter_text_column']].fillna('').astype(str):
        token_ids = tokenizer.encode_text(text)[:max_content]
        sequence: list[int] = []
        if bool(config['tokenizer']['add_bos']):
            sequence.append(tokenizer.bos_id)
        sequence.extend(token_ids)
        if bool(config['tokenizer']['add_eos']):
            sequence.append(tokenizer.eos_id)
        sequences.append(sequence)
    loader = DataLoader(sequences, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_token_sequences(batch, tokenizer.pad_id))
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), return_pooled=True)
            vectors = projection(outputs['pooled_embedding'])
            chunks.append(vectors.detach().cpu().numpy().astype(np.float32))
    verse_vectors = l2_normalize(np.concatenate(chunks, axis=0))
    return aggregate_by_poem(verse_vectors, verses['poem_id'].astype(str).tolist(), target_poem_ids)

def pool_spans(hidden_states: torch.Tensor, verse_spans: torch.Tensor, verse_mask: torch.Tensor) -> torch.Tensor:
    prefix = torch.cat([hidden_states.new_zeros(hidden_states.size(0), 1, hidden_states.size(-1)), hidden_states.cumsum(dim=1)], dim=1)
    starts = verse_spans[..., 0]
    ends = verse_spans[..., 1]
    batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device).unsqueeze(1)
    span_sums = prefix[batch_index, ends] - prefix[batch_index, starts]
    lengths = (ends - starts).clamp_min(1).unsqueeze(-1)
    return (span_sums / lengths)[verse_mask]

def encode_scratch_contextual(*, verses: pd.DataFrame, target_poem_ids: list[str], config: dict, tokenizer, model: ScratchTransformerEncoder, projection: torch.nn.Linear, device: torch.device, batch_size: int, num_workers: int) -> np.ndarray:
    dummy_teacher = np.zeros((len(verses), 1), dtype=np.float32)
    dataset = PackedContrastiveWindowDataset(verses.reset_index(drop=True), tokenizer=tokenizer, teacher_embeddings=dummy_teacher, max_length=int(config['tokenizer']['max_length']), add_bos=bool(config['tokenizer']['add_bos']), add_eos=bool(config['tokenizer']['add_eos']), verse_sep_token=str(config['tokenizer']['verse_sep_token']), filter_text_column=str(config['data']['filter_text_column']))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda batch: collate_packed_contrastive_batch(batch, pad_id=tokenizer.pad_id))
    vectors: list[np.ndarray] = []
    poem_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), return_pooled=False)
            span_vectors = pool_spans(outputs['last_hidden_state'], batch['verse_spans'].to(device), batch['verse_mask'].to(device))
            projected = projection(span_vectors).detach().cpu().numpy().astype(np.float32)
            vectors.append(projected)
            verse_counts = batch['verse_mask'].sum(dim=1).cpu().numpy().astype(int).tolist()
            for (poem_id, count) in zip(batch['poem_ids'], verse_counts):
                poem_ids.extend([str(poem_id)] * int(count))
    verse_vectors = l2_normalize(np.concatenate(vectors, axis=0))
    return aggregate_by_poem(verse_vectors, poem_ids, target_poem_ids)

def cache_path(feature_cache_dir: Path, feature_id: str, split_name: str) -> Path:
    return feature_cache_dir / safe_id(feature_id) / f'{split_name}.npy'

def load_or_build_scratch_features(*, feature_id: str, representation: str, run_info: dict[str, str], bundle: LabelBundle, verses_path: Path, feature_cache_dir: Path, device: torch.device, batch_size: int, num_workers: int) -> dict[str, np.ndarray]:
    by_split: dict[str, np.ndarray] = {}
    missing_splits = [split for split in ('train', 'dev', 'test') if not cache_path(feature_cache_dir, feature_id, split).exists()]
    config = tokenizer = model = projection = None
    if missing_splits:
        (config, tokenizer, model, projection) = load_scratch_model(Path(run_info['source_config']), Path(run_info['checkpoint']), device)
    for split_name in ('train', 'dev', 'test'):
        path = cache_path(feature_cache_dir, feature_id, split_name)
        if path.exists():
            by_split[split_name] = np.load(path).astype(np.float32, copy=False)
            continue
        assert config is not None and tokenizer is not None and (model is not None) and (projection is not None)
        poem_ids = bundle.poem_frames[split_name]['poem_id'].astype(str).tolist()
        verses = load_relevant_verses(verses_path, split_name, bundle.poem_frames[split_name]['poem_id'])
        if representation == 'isolated':
            features = encode_scratch_isolated(verses=verses, target_poem_ids=poem_ids, config=config, tokenizer=tokenizer, model=model, projection=projection, device=device, batch_size=batch_size)
        elif representation == 'contextual':
            features = encode_scratch_contextual(verses=verses, target_poem_ids=poem_ids, config=config, tokenizer=tokenizer, model=model, projection=projection, device=device, batch_size=batch_size, num_workers=num_workers)
        else:
            raise ValueError(f'Unknown scratch representation: {representation}')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features)
        by_split[split_name] = features
    return by_split

def build_lexical_features(bundle: LabelBundle, args: argparse.Namespace):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer='char', lowercase=True, ngram_range=(int(args.tfidf_ngram_min), int(args.tfidf_ngram_max)), min_df=int(args.tfidf_min_df), dtype=np.float32, sublinear_tf=True)
    vectorizer.fit(bundle.poem_frames['train']['poem_text'].fillna('').astype(str))
    return ({split: vectorizer.transform(bundle.poem_frames[split]['poem_text'].fillna('').astype(str)) for split in ('train', 'dev', 'test')}, {'num_features': int(len(vectorizer.vocabulary_))})

def pack_hf_windows(poem_frame: pd.DataFrame, tokenizer, max_length: int) -> list[dict[str, object]]:
    cls_id = int(tokenizer.cls_token_id or tokenizer.bos_token_id)
    sep_id = int(tokenizer.sep_token_id or tokenizer.eos_token_id)
    reserved = 2
    max_content = int(max_length) - reserved
    windows: list[dict[str, object]] = []
    for row in poem_frame.itertuples(index=False):
        poem_id = str(getattr(row, 'poem_id'))
        verses = str(getattr(row, 'poem_text')).split('\n')
        current: list[int] = []
        current_pool: list[int] = []
        for verse in verses:
            tokens = tokenizer.encode(verse, add_special_tokens=False)[:max_content]
            addition = tokens if not current else [sep_id] + tokens
            addition_pool = [1] * len(tokens) if not current else [0] + [1] * len(tokens)
            if current and len(current) + len(addition) > max_content:
                windows.append({'poem_id': poem_id, 'input_ids': [cls_id] + current + [sep_id], 'pool_mask': [0] + current_pool + [0]})
                current = []
                current_pool = []
                addition = tokens
                addition_pool = [1] * len(tokens)
            current.extend(addition)
            current_pool.extend(addition_pool)
        if current:
            windows.append({'poem_id': poem_id, 'input_ids': [cls_id] + current + [sep_id], 'pool_mask': [0] + current_pool + [0]})
        else:
            windows.append({'poem_id': poem_id, 'input_ids': [cls_id, sep_id], 'pool_mask': [0, 0]})
    return windows

def collate_hf_windows(batch: list[dict[str, object]], pad_id: int) -> dict[str, object]:
    max_len = max((len(item['input_ids']) for item in batch))
    input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    pool_mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    poem_ids = []
    for (row, item) in enumerate(batch):
        ids = list(item['input_ids'])
        mask = list(item['pool_mask'])
        input_ids[row, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        attention_mask[row, :len(ids)] = 1
        pool_mask[row, :len(mask)] = torch.as_tensor(mask, dtype=torch.float32)
        poem_ids.append(str(item['poem_id']))
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'pool_mask': pool_mask, 'poem_ids': poem_ids}

def encode_frozen_hf_poems(*, poem_frame: pd.DataFrame, model_name: str, max_length: int, batch_size: int, num_workers: int, device: torch.device) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    windows = pack_hf_windows(poem_frame, tokenizer, max_length)
    loader = DataLoader(windows, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda batch: collate_hf_windows(batch, int(tokenizer.pad_token_id)))
    target_poem_ids = poem_frame['poem_id'].astype(str).tolist()
    index = {poem_id: row for (row, poem_id) in enumerate(target_poem_ids)}
    sums: np.ndarray | None = None
    counts = np.zeros(len(target_poem_ids), dtype=np.float32)
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            mask = batch['pool_mask'].to(device).unsqueeze(-1)
            hidden = outputs.last_hidden_state
            token_counts = mask.sum(dim=1).clamp_min(1.0)
            vectors = (hidden * mask).sum(dim=1) / token_counts
            vectors_np = vectors.detach().cpu().numpy().astype(np.float32)
            if sums is None:
                sums = np.zeros((len(target_poem_ids), vectors_np.shape[1]), dtype=np.float32)
            for (vector, poem_id) in zip(vectors_np, batch['poem_ids']):
                row = index[str(poem_id)]
                sums[row] += vector
                counts[row] += 1.0
    if sums is None:
        raise ValueError('No HF windows were encoded.')
    sums /= np.clip(counts[:, None], a_min=1.0, a_max=None)
    return l2_normalize(sums)

def load_or_build_frozen_hf_features(*, feature_id: str, bundle: LabelBundle, args: argparse.Namespace, device: torch.device) -> dict[str, np.ndarray]:
    by_split: dict[str, np.ndarray] = {}
    for split_name in ('train', 'dev', 'test'):
        path = cache_path(args.feature_cache_dir, feature_id, split_name)
        if path.exists():
            by_split[split_name] = np.load(path).astype(np.float32, copy=False)
            continue
        features = encode_frozen_hf_poems(poem_frame=bundle.poem_frames[split_name], model_name=str(args.hf_model_name), max_length=int(args.hf_max_length), batch_size=int(args.batch_size), num_workers=int(args.num_workers), device=device)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features)
        by_split[split_name] = features
    return by_split

def sentence_cache_path(cache_dir: Path | None, split_name: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f'{split_name}.npy'

def load_cached_verse_embeddings(cache_path: Path | None, expected_rows: int) -> np.ndarray | None:
    if cache_path is None or not cache_path.exists():
        return None
    cached = np.load(cache_path).astype(np.float32, copy=False)
    if int(cached.shape[0]) != int(expected_rows):
        return None
    return cached

def encode_sentence_transformer_english_verses(*, verses: pd.DataFrame, target_poem_ids: list[str], text_column: str, model_name: str, device: torch.device, batch_size: int) -> np.ndarray:
    from filter_verse_embeddings.teachers.sentence import SentenceTeacher
    teacher = SentenceTeacher(model_name, device=str(device))
    vectors = teacher.encode_texts(verses[text_column].fillna('').astype(str), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    verse_vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
    return aggregate_by_poem(verse_vectors, verses['poem_id'].astype(str).tolist(), target_poem_ids)

def encode_hf_english_verses(*, verses: pd.DataFrame, target_poem_ids: list[str], text_column: str, model_name: str, max_length: int, batch_size: int, device: torch.device) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    try:
        model = AutoModel.from_pretrained(model_name, attn_implementation='eager').to(device)
    except TypeError:
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    texts = verses[text_column].fillna('').astype(str).tolist()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
            outputs = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
            mask = enc['attention_mask'].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
            if start // batch_size % 100 == 0:
                done = min(start + batch_size, len(texts))
                print(f'  encoded {done:,}/{len(texts):,} English verses with {model_name}', flush=True)
    verse_vectors = l2_normalize(np.concatenate(chunks, axis=0))
    return aggregate_by_poem(verse_vectors, verses['poem_id'].astype(str).tolist(), target_poem_ids)

def build_english_poem_frame(verses: pd.DataFrame, target_poem_ids: list[str], text_column: str) -> pd.DataFrame:
    """Group verse rows into one row per poem with English text joined by newline."""
    verses = verses.copy()
    verses['poem_id'] = verses['poem_id'].astype(str)
    grouped: dict[str, list[str]] = {pid: [] for pid in target_poem_ids}
    for row in verses.sort_values(['poem_id', 'verse_position', 'verse_id'], kind='stable').itertuples(index=False):
        pid = str(row.poem_id)
        if pid in grouped:
            val = getattr(row, text_column, '') or ''
            grouped[pid].append(str(val))
    rows = [{'poem_id': pid, 'poem_text': '\n'.join(grouped[pid])} for pid in target_poem_ids]
    return pd.DataFrame(rows)

def encode_hf_english_poem_context(*, verses: pd.DataFrame, target_poem_ids: list[str], text_column: str, model_name: str, max_length: int, batch_size: int, device: torch.device) -> np.ndarray:
    """Contextual poem-level encoding: pack all English verses of a poem into one
    sliding window, pool over verse tokens only, aggregate windows by mean."""
    from transformers import AutoModel, AutoTokenizer
    poem_frame = build_english_poem_frame(verses, target_poem_ids, text_column)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    try:
        model = AutoModel.from_pretrained(model_name, attn_implementation='eager').to(device)
    except TypeError:
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    windows = pack_hf_windows(poem_frame, tokenizer, max_length)
    loader = DataLoader(windows, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: collate_hf_windows(batch, int(tokenizer.pad_token_id)))
    index = {poem_id: row for (row, poem_id) in enumerate(target_poem_ids)}
    sums: np.ndarray | None = None
    counts = np.zeros(len(target_poem_ids), dtype=np.float32)
    done = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            mask = batch['pool_mask'].to(device).unsqueeze(-1)
            hidden = outputs.last_hidden_state
            token_counts = mask.sum(dim=1).clamp_min(1.0)
            vectors = (hidden * mask).sum(dim=1) / token_counts
            vectors_np = vectors.detach().cpu().numpy().astype(np.float32)
            if sums is None:
                sums = np.zeros((len(target_poem_ids), vectors_np.shape[1]), dtype=np.float32)
            for (vector, poem_id) in zip(vectors_np, batch['poem_ids']):
                row = index[str(poem_id)]
                sums[row] += vector
                counts[row] += 1.0
            done += len(batch['poem_ids'])
            if done // batch_size % 50 == 0:
                print(f'  encoded {done:,}/{len(windows):,} windows with {model_name}', flush=True)
    if sums is None:
        raise ValueError('No poem windows were encoded.')
    sums /= np.clip(counts[:, None], a_min=1.0, a_max=None)
    return l2_normalize(sums)

def load_or_build_english_hf_poem_context_features(*, feature_id: str, model_name: str, bundle: 'LabelBundle', verses_path: Path, feature_cache_dir: Path, text_column: str, max_length: int, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    by_split: dict[str, np.ndarray] = {}
    for split_name in ('train', 'dev', 'test'):
        path = cache_path(feature_cache_dir, feature_id, split_name)
        if path.exists():
            by_split[split_name] = np.load(path).astype(np.float32, copy=False)
            continue
        poem_ids = bundle.poem_frames[split_name]['poem_id'].astype(str).tolist()
        verses = load_relevant_verses(verses_path, split_name, bundle.poem_frames[split_name]['poem_id'], text_column)
        features = encode_hf_english_poem_context(verses=verses, target_poem_ids=poem_ids, text_column=text_column, model_name=model_name, max_length=max_length, batch_size=batch_size, device=device)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features)
        by_split[split_name] = features
    return by_split

def build_english_poem_frame(verses: pd.DataFrame, target_poem_ids: list[str], text_column: str) -> pd.DataFrame:
    """Group verse rows into one row per poem with English text joined by newline."""
    verses = verses.copy()
    verses['poem_id'] = verses['poem_id'].astype(str)
    grouped: dict[str, list[str]] = {pid: [] for pid in target_poem_ids}
    for row in verses.sort_values(['poem_id', 'verse_position', 'verse_id'], kind='stable').itertuples(index=False):
        pid = str(row.poem_id)
        if pid in grouped:
            val = getattr(row, text_column, '') or ''
            grouped[pid].append(str(val))
    rows = [{'poem_id': pid, 'poem_text': '\n'.join(grouped[pid])} for pid in target_poem_ids]
    return pd.DataFrame(rows)

def encode_hf_english_poem_context(*, verses: pd.DataFrame, target_poem_ids: list[str], text_column: str, model_name: str, max_length: int, batch_size: int, device: torch.device) -> np.ndarray:
    """Contextual poem-level encoding: pack all English verses of a poem into sliding
    windows (verse tokens only are pooled), then aggregate windows per poem by mean."""
    from transformers import AutoModel, AutoTokenizer
    poem_frame = build_english_poem_frame(verses, target_poem_ids, text_column)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    try:
        model = AutoModel.from_pretrained(model_name, attn_implementation='eager').to(device)
    except TypeError:
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    windows = pack_hf_windows(poem_frame, tokenizer, max_length)
    loader = DataLoader(windows, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: collate_hf_windows(batch, int(tokenizer.pad_token_id)))
    index = {poem_id: row for (row, poem_id) in enumerate(target_poem_ids)}
    sums: np.ndarray | None = None
    counts = np.zeros(len(target_poem_ids), dtype=np.float32)
    done = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            mask = batch['pool_mask'].to(device).unsqueeze(-1)
            hidden = outputs.last_hidden_state
            token_counts = mask.sum(dim=1).clamp_min(1.0)
            vectors = (hidden * mask).sum(dim=1) / token_counts
            vectors_np = vectors.detach().cpu().numpy().astype(np.float32)
            if sums is None:
                sums = np.zeros((len(target_poem_ids), vectors_np.shape[1]), dtype=np.float32)
            for (vector, poem_id) in zip(vectors_np, batch['poem_ids']):
                row = index[str(poem_id)]
                sums[row] += vector
                counts[row] += 1.0
            done += len(batch['poem_ids'])
            if done // batch_size % 50 == 0:
                print(f'  encoded {done:,}/{len(windows):,} windows with {model_name}', flush=True)
    if sums is None:
        raise ValueError('No poem windows were encoded.')
    sums /= np.clip(counts[:, None], a_min=1.0, a_max=None)
    return l2_normalize(sums)

def load_or_build_english_hf_poem_context_features(*, feature_id: str, model_name: str, bundle: 'LabelBundle', verses_path: Path, feature_cache_dir: Path, text_column: str, max_length: int, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    by_split: dict[str, np.ndarray] = {}
    for split_name in ('train', 'dev', 'test'):
        path = cache_path(feature_cache_dir, feature_id, split_name)
        if path.exists():
            by_split[split_name] = np.load(path).astype(np.float32, copy=False)
            continue
        poem_ids = bundle.poem_frames[split_name]['poem_id'].astype(str).tolist()
        verses = load_relevant_verses(verses_path, split_name, bundle.poem_frames[split_name]['poem_id'], text_column)
        features = encode_hf_english_poem_context(verses=verses, target_poem_ids=poem_ids, text_column=text_column, model_name=model_name, max_length=max_length, batch_size=batch_size, device=device)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features)
        by_split[split_name] = features
    return by_split

def load_or_build_english_sentence_transformer_features(*, feature_id: str, model_name: str, bundle: LabelBundle, verses_path: Path, feature_cache_dir: Path, sentence_cache_dir: Path | None, text_column: str, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    by_split: dict[str, np.ndarray] = {}
    for split_name in ('train', 'dev', 'test'):
        path = cache_path(feature_cache_dir, feature_id, split_name)
        if path.exists():
            by_split[split_name] = np.load(path).astype(np.float32, copy=False)
            continue
        poem_ids = bundle.poem_frames[split_name]['poem_id'].astype(str).tolist()
        split_verses = load_split_verses(verses_path, split_name, text_column)
        keep = split_verses['poem_id'].astype(str).isin(set(poem_ids)).to_numpy()
        verses = split_verses.loc[keep].reset_index(drop=True)
        cached = load_cached_verse_embeddings(sentence_cache_path(sentence_cache_dir, split_name), len(split_verses))
        if cached is not None:
            features = aggregate_by_poem(l2_normalize(cached[keep]), verses['poem_id'].astype(str).tolist(), poem_ids)
        else:
            features = encode_sentence_transformer_english_verses(verses=verses, target_poem_ids=poem_ids, text_column=text_column, model_name=model_name, device=device, batch_size=batch_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features)
        by_split[split_name] = features
    return by_split

def load_or_build_english_hf_features(*, feature_id: str, model_name: str, bundle: LabelBundle, verses_path: Path, feature_cache_dir: Path, text_column: str, max_length: int, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    by_split: dict[str, np.ndarray] = {}
    for split_name in ('train', 'dev', 'test'):
        path = cache_path(feature_cache_dir, feature_id, split_name)
        if path.exists():
            by_split[split_name] = np.load(path).astype(np.float32, copy=False)
            continue
        poem_ids = bundle.poem_frames[split_name]['poem_id'].astype(str).tolist()
        verses = load_relevant_verses(verses_path, split_name, bundle.poem_frames[split_name]['poem_id'], text_column)
        features = encode_hf_english_verses(verses=verses, target_poem_ids=poem_ids, text_column=text_column, model_name=model_name, max_length=max_length, batch_size=batch_size, device=device)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features)
        by_split[split_name] = features
    return by_split

def train_probe(feature_id: str, feature_kind: str, features: dict[str, object], bundle: LabelBundle, args: argparse.Namespace, metadata: dict) -> dict:
    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsRestClassifier
    classifier = OneVsRestClassifier(SGDClassifier(loss='log_loss', penalty='l2', alpha=float(args.classifier_alpha), max_iter=int(args.classifier_max_iter), tol=0.001, class_weight='balanced', random_state=int(args.seed)), n_jobs=int(args.classifier_n_jobs))
    classifier.fit(features['train'], bundle.y['train'])
    scores = {split: np.asarray(classifier.decision_function(features[split]), dtype=np.float32) for split in ('dev', 'test')}
    (threshold, dev_metrics) = choose_threshold(bundle.y['dev'], scores['dev'])
    test_metrics = evaluate_scores(y_true=bundle.y['test'], scores=scores['test'], threshold=threshold)
    report = {'feature_id': feature_id, 'feature_kind': feature_kind, 'classifier': {'type': 'OneVsRestClassifier(SGDClassifier(log_loss))', 'alpha': float(args.classifier_alpha), 'max_iter': int(args.classifier_max_iter), 'class_weight': 'balanced'}, 'metadata': metadata, 'label_coverage': bundle.coverage, 'dev': dev_metrics, 'test': test_metrics}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f'{safe_id(feature_id)}.json'
    csv_path = args.output_dir / f'{safe_id(feature_id)}.csv'
    write_json(report, report_path)
    pd.DataFrame([{'feature_id': feature_id, 'feature_kind': feature_kind, **{f'dev_{key}': value for (key, value) in dev_metrics.items()}, **{f'test_{key}': value for (key, value) in test_metrics.items()}, **metadata}]).to_csv(csv_path, index=False)
    return report

def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    bundle = load_primary_top_labels(verses_path=args.verses_path, types_path=args.types_path, poem_types_path=args.poem_types_path, min_train_support=int(args.min_train_support), exclude_type_prefixes=args.exclude_type_prefixes)
    bundle = maybe_limit_bundle(bundle, args.max_train_poems, args.max_eval_poems)
    reports = []
    if args.target in ('baselines', 'all'):
        (lexical_features, lexical_meta) = build_lexical_features(bundle, args)
        reports.append(train_probe('baseline_lexical_tfidf_poem_text', 'lexical_tfidf', lexical_features, bundle, args, {'model': 'char_tfidf', **lexical_meta}))
        hf_feature_id = f'baseline_frozen_{safe_id(args.hf_model_name)}_packed_poem'
        hf_features = load_or_build_frozen_hf_features(feature_id=hf_feature_id, bundle=bundle, args=args, device=device)
        reports.append(train_probe(hf_feature_id, 'frozen_hf_packed_poem', hf_features, bundle, args, {'model_name': str(args.hf_model_name), 'hf_max_length': int(args.hf_max_length)}))
    if args.target in ('english_baselines', 'all'):
        for model_name in args.english_sentence_transformer_models:
            feature_id = f'baseline_english_sentence_transformer_{safe_id(model_name)}_verse_mean'
            features = load_or_build_english_sentence_transformer_features(feature_id=feature_id, model_name=str(model_name), bundle=bundle, verses_path=args.verses_path, feature_cache_dir=args.feature_cache_dir, sentence_cache_dir=args.english_sentence_cache_dir, text_column=str(args.english_text_column), device=device, batch_size=int(args.batch_size))
            reports.append(train_probe(feature_id, 'english_sentence_transformer_verse_mean', features, bundle, args, {'model_name': str(model_name), 'input_side': 'english_translation', 'text_column': str(args.english_text_column), 'representation': 'verse_mean', 'translation_assisted': True}))
        for model_name in args.english_hf_models:
            feature_id = f'baseline_english_hf_{safe_id(model_name)}_verse_mean'
            features = load_or_build_english_hf_features(feature_id=feature_id, model_name=str(model_name), bundle=bundle, verses_path=args.verses_path, feature_cache_dir=args.feature_cache_dir, text_column=str(args.english_text_column), max_length=int(args.english_hf_max_length), device=device, batch_size=int(args.batch_size))
            reports.append(train_probe(feature_id, 'english_hf_verse_mean', features, bundle, args, {'model_name': str(model_name), 'input_side': 'english_translation', 'text_column': str(args.english_text_column), 'representation': 'verse_mean', 'english_hf_max_length': int(args.english_hf_max_length), 'translation_assisted': True}))
        for model_name in args.english_hf_poem_context_models:
            feature_id = f'baseline_english_hf_{safe_id(model_name)}_poem_context'
            features = load_or_build_english_hf_poem_context_features(feature_id=feature_id, model_name=str(model_name), bundle=bundle, verses_path=args.verses_path, feature_cache_dir=args.feature_cache_dir, text_column=str(args.english_text_column), max_length=int(args.english_hf_poem_context_max_length), device=device, batch_size=int(args.batch_size))
            reports.append(train_probe(feature_id, 'english_hf_poem_context', features, bundle, args, {'model_name': str(model_name), 'input_side': 'english_translation', 'text_column': str(args.english_text_column), 'representation': 'poem_context', 'english_hf_max_length': int(args.english_hf_poem_context_max_length), 'translation_assisted': True}))
    if args.target in ('trained', 'all'):
        if args.manifest is None:
            raise SystemExit('--manifest is required for trained probes.')
        rows = read_manifest(args.manifest)
        if args.run_index is not None:
            rows = [rows[int(args.run_index)]]
        for run_info in rows:
            for representation in args.representations:
                feature_id = f"trained_{run_info['label']}_{representation}"
                features = load_or_build_scratch_features(feature_id=feature_id, representation=str(representation), run_info=run_info, bundle=bundle, verses_path=args.verses_path, feature_cache_dir=args.feature_cache_dir, device=device, batch_size=int(args.batch_size), num_workers=int(args.num_workers))
                reports.append(train_probe(feature_id, f'trained_{representation}', features, bundle, args, {'label': run_info.get('label', ''), 'surface': run_info.get('surface', ''), 'sem_config': run_info.get('sem_config', ''), 'epoch': run_info.get('epoch', ''), 'task_id': run_info.get('task_id', '')}))
    write_json({'num_reports': len(reports), 'reports': [report['feature_id'] for report in reports], 'label_coverage': bundle.coverage}, args.output_dir / 'last_run_index.json')
if __name__ == '__main__':
    main()
