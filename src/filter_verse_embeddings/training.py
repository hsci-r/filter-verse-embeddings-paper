"""Train FILTER with MLM plus verse-level contrastive distillation."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import csv
from functools import partial
import logging
import math
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter as _TBSummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from filter_verse_embeddings.losses import info_nce_logits, mlm_loss_and_count, symmetric_info_nce_loss, weighted_cosine_alignment_loss
from filter_verse_embeddings.masking import apply_mlm_mask
from filter_verse_embeddings.model import ScratchEncoderConfig, ScratchTransformerEncoder
from filter_verse_embeddings.data import HFTokenizer, SentencePieceTokenizer, load_config, load_split_table, save_json, save_yaml, split_frame
from filter_verse_embeddings.dataset import PackedContrastiveWindowDataset, build_word_rows_by_poem, collate_packed_contrastive_batch, token_teacher_cache_path
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train FILTER.')
    parser.add_argument('--config', default='configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml')
    parser.add_argument('--init-checkpoint', default=None)
    parser.add_argument('--init-hf-model', default=None, metavar='HF_MODEL_NAME', help="Initialise encoder weights from a HuggingFace BERT-compatible model (e.g. 'EMBEDDIA/finest-bert'). Token embeddings and the MLM head are left at random init due to vocabulary mismatch. Ignored when --resume-checkpoint is set.")
    parser.add_argument('--resume-checkpoint', default=None)
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--checkpoint-dir', default=None)
    parser.add_argument('--teacher-cache-dir', default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--teacher-device', default='auto')
    parser.add_argument('--max-train-steps', type=int, default=None)
    parser.add_argument('--max-dev-steps', type=int, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--scheduler-num-epochs', type=int, default=None, help='Use this epoch horizon for LR scheduling while --max-epochs controls the actual stop epoch. Useful for short resumed canaries that should preserve the original scheduler.')
    parser.add_argument('--max-train-examples', type=int, default=None)
    parser.add_argument('--max-dev-examples', type=int, default=None)
    parser.add_argument('--override-lr', type=float, default=None)
    parser.add_argument('--override-use-scheduler', choices=('true', 'false'), default=None)
    parser.add_argument('--override-warmup-ratio', type=float, default=None)
    parser.add_argument('--override-min-lr-ratio', type=float, default=None)
    parser.add_argument('--override-lambda-v', type=float, default=None)
    parser.add_argument('--override-lambda-w', type=float, default=None)
    parser.add_argument('--override-word-loss-scale', type=float, default=None)
    parser.add_argument('--override-temperature', type=float, default=None)
    parser.add_argument('--override-verse-warmup-fraction', type=float, default=None)
    parser.add_argument('--contrastive-direction', choices=('symmetric', 'forward_only'), default=None)
    parser.add_argument('--projection-lr-multiplier', type=float, default=None)
    parser.add_argument('--log-contrastive-stats-every-steps', type=int, default=0)
    parser.add_argument('--module-debug-every-steps', type=int, default=0)
    parser.add_argument('--reset-optimizer-state', action='store_true')
    parser.add_argument('--resume-lr-scale', type=float, default=None)
    parser.add_argument('--resume-absolute-lr', type=float, default=None, help='After restoring a checkpoint, set the resumed model LR to this absolute value and adjust scheduler base_lrs so future scheduler steps preserve the current phase.')
    parser.add_argument('--abort-on-nonfinite', action='store_true')
    parser.add_argument('--detect-anomaly', action='store_true')
    parser.add_argument('--tb-log-dir', type=str, default=None, help='TensorBoard log directory. If omitted, falls back to config output.tensorboard_dir, or {run_dir}/tensorboard if neither is set.')
    parser.add_argument('--tb-log-every-steps', type=int, default=500, help='Log scalar summaries to TensorBoard every N global steps (default: 500).')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(device_name: str) -> torch.device:
    if device_name != 'auto':
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def ensure_run_dirs(run_dir: Path, checkpoint_dir: Path, *, resume: bool) -> None:
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f'Run directory {run_dir} already exists. Choose a new run_name/run_dir to avoid overwriting.')
    for path in [run_dir, checkpoint_dir, run_dir / 'logs', run_dir / 'teacher_cache']:
        path.mkdir(parents=True, exist_ok=True)

def load_optional_checkpoint(model: ScratchTransformerEncoder, checkpoint_path: str | None) -> str | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return str(checkpoint_path)

def load_hf_bert_weights(model: ScratchTransformerEncoder, hf_model_name: str) -> str:
    """Transfer encoder weights from a HuggingFace BERT model into ScratchTransformerEncoder.

    Token embeddings are intentionally skipped because the vocabularies differ.
    Layer norms are copied despite the architectural pre-norm vs post-norm difference —
    they serve as a warm-start that the model can rapidly adapt.
    The MLM head (mlm_transform / mlm_decoder / mlm_bias) is left at random init
    because the output vocabulary is ours, not the source model's.

    Returns hf_model_name for logging.
    """
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("The 'transformers' package is required to load HF BERT weights.") from exc
    log.info("Loading HF BERT encoder weights from '%s' …", hf_model_name)
    hf_model = AutoModel.from_pretrained(hf_model_name)
    hf_sd = {k: v.cpu() for (k, v) in hf_model.state_dict().items()}
    del hf_model
    if 'bert.embeddings.position_embeddings.weight' in hf_sd:
        pfx = 'bert.'
    elif 'embeddings.position_embeddings.weight' in hf_sd:
        pfx = ''
    else:
        log.warning("Cannot detect HF BERT key prefix in '%s'; weight transfer aborted. Keys sample: %s", hf_model_name, list(hf_sd.keys())[:5])
        return hf_model_name
    cfg = model.config
    num_layers = cfg.num_layers
    our_sd = model.state_dict()
    updated: list[str] = []
    skipped: list[str] = []

    def _assign(dst: str, src: torch.Tensor) -> None:
        if dst not in our_sd:
            skipped.append(f'{dst} [not in model]')
            return
        if our_sd[dst].shape != src.shape:
            skipped.append(f'{dst} [shape mismatch: model={tuple(our_sd[dst].shape)} hf={tuple(src.shape)}]')
            return
        our_sd[dst] = src.clone()
        updated.append(dst)
    hf_pos = hf_sd.get(f'{pfx}embeddings.position_embeddings.weight')
    if hf_pos is not None:
        our_len = our_sd['position_embeddings.weight'].shape[0]
        _assign('position_embeddings.weight', hf_pos[:our_len])
    for sfx in ('weight', 'bias'):
        key = f'{pfx}embeddings.LayerNorm.{sfx}'
        if key in hf_sd:
            _assign(f'embedding_layer_norm.{sfx}', hf_sd[key])
    for i in range(num_layers):
        hp = f'{pfx}encoder.layer.{i}'
        dp = f'encoder.layers.{i}'
        q_w = hf_sd.get(f'{hp}.attention.self.query.weight')
        k_w = hf_sd.get(f'{hp}.attention.self.key.weight')
        v_w = hf_sd.get(f'{hp}.attention.self.value.weight')
        if q_w is not None and k_w is not None and (v_w is not None):
            _assign(f'{dp}.self_attn.in_proj_weight', torch.cat([q_w, k_w, v_w], dim=0))
        q_b = hf_sd.get(f'{hp}.attention.self.query.bias')
        k_b = hf_sd.get(f'{hp}.attention.self.key.bias')
        v_b = hf_sd.get(f'{hp}.attention.self.value.bias')
        if q_b is not None and k_b is not None and (v_b is not None):
            _assign(f'{dp}.self_attn.in_proj_bias', torch.cat([q_b, k_b, v_b], dim=0))
        for sfx in ('weight', 'bias'):
            key = f'{hp}.attention.output.dense.{sfx}'
            if key in hf_sd:
                _assign(f'{dp}.self_attn.out_proj.{sfx}', hf_sd[key])
        for sfx in ('weight', 'bias'):
            key = f'{hp}.attention.output.LayerNorm.{sfx}'
            if key in hf_sd:
                _assign(f'{dp}.norm1.{sfx}', hf_sd[key])
        for sfx in ('weight', 'bias'):
            key = f'{hp}.intermediate.dense.{sfx}'
            if key in hf_sd:
                _assign(f'{dp}.linear1.{sfx}', hf_sd[key])
            key = f'{hp}.output.dense.{sfx}'
            if key in hf_sd:
                _assign(f'{dp}.linear2.{sfx}', hf_sd[key])
        for sfx in ('weight', 'bias'):
            key = f'{hp}.output.LayerNorm.{sfx}'
            if key in hf_sd:
                _assign(f'{dp}.norm2.{sfx}', hf_sd[key])
    last = num_layers - 1
    for sfx in ('weight', 'bias'):
        key = f'{pfx}encoder.layer.{last}.output.LayerNorm.{sfx}'
        if key in hf_sd:
            _assign(f'final_layer_norm.{sfx}', hf_sd[key])
    model.load_state_dict(our_sd, strict=False)
    log.info('HF BERT weight transfer complete: %d tensors updated, %d skipped.', len(updated), len(skipped))
    if skipped:
        log.debug('Skipped during HF BERT transfer: %s', skipped)
    return hf_model_name

def append_metrics_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def ensure_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f'Non-finite values detected in {name}.')

def build_optimizer(model: ScratchTransformerEncoder, projection: torch.nn.Linear, word_projection: torch.nn.Linear | None=None, *, lr: float, weight_decay: float, projection_lr_multiplier: float) -> AdamW:
    param_groups = [{'params': list(model.parameters()), 'lr': lr}, {'params': list(projection.parameters()), 'lr': lr * projection_lr_multiplier}]
    if word_projection is not None:
        param_groups.append({'params': list(word_projection.parameters()), 'lr': lr * projection_lr_multiplier})
    return AdamW(param_groups, lr=lr, weight_decay=weight_decay)

def restore_optimizer_state(optimizer: AdamW, optimizer_state_dict: dict[str, object]) -> None:
    try:
        optimizer.load_state_dict(optimizer_state_dict)
        return
    except ValueError:
        current_state = optimizer.state_dict()
        saved_groups = optimizer_state_dict.get('param_groups', [])
        current_groups = current_state.get('param_groups', [])
        if len(saved_groups) != 1 or len(current_groups) < 2:
            raise
        old_group = dict(saved_groups[0])
        adapted_state = {'state': optimizer_state_dict['state'], 'param_groups': []}
        for current_group in current_groups:
            merged_group = dict(current_group)
            for (key, value) in old_group.items():
                if key in {'params', 'lr', 'initial_lr'}:
                    continue
                merged_group[key] = value
            adapted_state['param_groups'].append(merged_group)
        optimizer.load_state_dict(adapted_state)
        log.warning('Adapted optimizer state from a single param-group checkpoint to the current split optimizer.')

def current_lrs(optimizer: AdamW) -> tuple[float, float]:
    model_lr = float(optimizer.param_groups[0]['lr'])
    projection_lr = float(optimizer.param_groups[-1]['lr'])
    return (model_lr, projection_lr)

def projection_weight_norm(projection: torch.nn.Linear) -> float:
    return float(projection.weight.detach().norm().item())
TRACKED_MODULE_GROUPS = ('token_embeddings', 'position_embeddings', 'embedding_layer_norm', 'encoder.layers.0')

def tracked_module_group(parameter_name: str) -> str | None:
    if parameter_name.startswith('token_embeddings.'):
        return 'token_embeddings'
    if parameter_name.startswith('position_embeddings.'):
        return 'position_embeddings'
    if parameter_name.startswith('embedding_layer_norm.'):
        return 'embedding_layer_norm'
    if parameter_name.startswith('encoder.layers.0.'):
        return 'encoder.layers.0'
    return None

def named_trainable_parameters(model: ScratchTransformerEncoder, projection: torch.nn.Linear, word_projection: torch.nn.Linear | None=None) -> list[tuple[str, torch.nn.Parameter]]:
    named_parameters = list(model.named_parameters())
    named_parameters.extend(((f'projection.{name}', param) for (name, param) in projection.named_parameters()))
    if word_projection is not None:
        named_parameters.extend(((f'word_projection.{name}', param) for (name, param) in word_projection.named_parameters()))
    return named_parameters

def optimizer_group_lookup(optimizer: AdamW) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            lookup[id(param)] = group
    return lookup

def scalar_from_state_step(step_value: Any) -> float | None:
    if step_value is None:
        return None
    if isinstance(step_value, torch.Tensor):
        return float(step_value.detach().cpu().item())
    return float(step_value)

def update_tensor_stats(*, tensor: torch.Tensor | None, sq_acc: dict[str, float], min_acc: dict[str, float], max_acc: dict[str, float], key: str) -> None:
    if tensor is None:
        return
    detached = tensor.detach()
    sq_acc[key] += float(detached.pow(2).sum().item())
    min_acc[key] = min(min_acc[key], float(detached.min().item()))
    max_acc[key] = max(max_acc[key], float(detached.max().item()))

def final_tensor_stats(*, sq_acc: dict[str, float], min_acc: dict[str, float], max_acc: dict[str, float], key: str) -> tuple[float, float, float]:
    norm = math.sqrt(sq_acc[key]) if sq_acc[key] > 0.0 else 0.0
    min_value = min_acc[key] if min_acc[key] != float('inf') else float('nan')
    max_value = max_acc[key] if max_acc[key] != float('-inf') else float('nan')
    return (float(norm), float(min_value), float(max_value))

def estimate_adamw_update_components(*, param: torch.nn.Parameter, optimizer_group: dict[str, Any], optimizer_state: dict[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor]:
    lr = float(optimizer_group['lr'])
    weight_decay = float(optimizer_group.get('weight_decay', 0.0))
    decay_term = lr * weight_decay * param.detach()
    exp_avg = optimizer_state.get('exp_avg')
    exp_avg_sq = optimizer_state.get('exp_avg_sq')
    step = scalar_from_state_step(optimizer_state.get('step'))
    if exp_avg is None or exp_avg_sq is None or step is None or (step <= 0):
        return (None, decay_term)
    (beta1, beta2) = optimizer_group.get('betas', (0.9, 0.999))
    eps = float(optimizer_group.get('eps', 1e-08))
    bias_correction1 = 1.0 - float(beta1) ** step
    bias_correction2 = 1.0 - float(beta2) ** step
    if bias_correction1 <= 0.0 or bias_correction2 <= 0.0:
        return (None, decay_term)
    m_hat = exp_avg.detach() / bias_correction1
    v_hat = exp_avg_sq.detach() / bias_correction2
    adam_term = lr * (m_hat / (v_hat.sqrt() + eps))
    return (adam_term, decay_term)

def collect_tracked_module_debug_stats(model: ScratchTransformerEncoder, optimizer: AdamW) -> dict[str, float]:
    optimizer_lookup = optimizer_group_lookup(optimizer)
    sq_acc = {key: 0.0 for key in ('param', 'grad', 'exp_avg', 'exp_avg_sq', 'adam_update', 'decay_update', 'combined_update')}
    rows: dict[str, dict[str, float]] = {}
    min_acc_template = {key: float('inf') for key in ('exp_avg_sq',)}
    max_acc_template = {key: float('-inf') for key in ('exp_avg_sq',)}
    grouped: dict[str, dict[str, Any]] = {}
    for group_name in TRACKED_MODULE_GROUPS:
        grouped[group_name] = {'param_sq': 0.0, 'grad_sq': 0.0, 'exp_avg_sq': 0.0, 'exp_avg_sq_stats': {'exp_avg_sq': 0.0}, 'adam_update_sq': 0.0, 'decay_update_sq': 0.0, 'combined_update_sq': 0.0, 'max_abs_weight': 0.0, 'max_abs_grad': 0.0, 'exp_avg_sq_min': float('inf'), 'exp_avg_sq_max': float('-inf')}
    for (name, param) in model.named_parameters():
        group_name = tracked_module_group(name)
        if group_name is None:
            continue
        bucket = grouped[group_name]
        param_detached = param.detach()
        bucket['param_sq'] += float(param_detached.pow(2).sum().item())
        bucket['max_abs_weight'] = max(bucket['max_abs_weight'], float(param_detached.abs().max().item()))
        if param.grad is not None and torch.isfinite(param.grad).all():
            grad_detached = param.grad.detach()
            bucket['grad_sq'] += float(grad_detached.pow(2).sum().item())
            bucket['max_abs_grad'] = max(bucket['max_abs_grad'], float(grad_detached.abs().max().item()))
        elif param.grad is not None:
            bucket['grad_sq'] = float('nan')
            bucket['max_abs_grad'] = float('nan')
        optimizer_group = optimizer_lookup.get(id(param))
        optimizer_state = optimizer.state.get(param, {})
        exp_avg = optimizer_state.get('exp_avg')
        exp_avg_sq = optimizer_state.get('exp_avg_sq')
        if exp_avg is not None:
            bucket['exp_avg_sq'] += float(exp_avg.detach().pow(2).sum().item())
        if exp_avg_sq is not None:
            exp_avg_sq_detached = exp_avg_sq.detach()
            bucket['exp_avg_sq_stats']['exp_avg_sq'] += float(exp_avg_sq_detached.pow(2).sum().item())
            bucket['exp_avg_sq_min'] = min(bucket['exp_avg_sq_min'], float(exp_avg_sq_detached.min().item()))
            bucket['exp_avg_sq_max'] = max(bucket['exp_avg_sq_max'], float(exp_avg_sq_detached.max().item()))
        if optimizer_group is not None:
            (adam_term, decay_term) = estimate_adamw_update_components(param=param, optimizer_group=optimizer_group, optimizer_state=optimizer_state)
            if adam_term is not None:
                bucket['adam_update_sq'] += float(adam_term.pow(2).sum().item())
                combined = adam_term + decay_term
            else:
                combined = decay_term
            bucket['decay_update_sq'] += float(decay_term.pow(2).sum().item())
            bucket['combined_update_sq'] += float(combined.pow(2).sum().item())
    row: dict[str, float] = {}
    for (group_name, bucket) in grouped.items():
        param_norm = math.sqrt(bucket['param_sq']) if bucket['param_sq'] > 0.0 else 0.0
        grad_sq = bucket['grad_sq']
        grad_norm = float('nan') if isinstance(grad_sq, float) and math.isnan(grad_sq) else math.sqrt(grad_sq)
        exp_avg_norm = math.sqrt(bucket['exp_avg_sq']) if bucket['exp_avg_sq'] > 0.0 else 0.0
        exp_avg_sq_norm = math.sqrt(bucket['exp_avg_sq_stats']['exp_avg_sq']) if bucket['exp_avg_sq_stats']['exp_avg_sq'] > 0.0 else 0.0
        adam_update_norm = math.sqrt(bucket['adam_update_sq']) if bucket['adam_update_sq'] > 0.0 else 0.0
        decay_update_norm = math.sqrt(bucket['decay_update_sq']) if bucket['decay_update_sq'] > 0.0 else 0.0
        combined_update_norm = math.sqrt(bucket['combined_update_sq']) if bucket['combined_update_sq'] > 0.0 else 0.0
        prefix = group_name.replace('.', '_')
        row[f'{prefix}_param_norm'] = float(param_norm)
        row[f'{prefix}_max_abs_weight'] = float(bucket['max_abs_weight'])
        row[f'{prefix}_grad_norm'] = float(grad_norm)
        row[f'{prefix}_max_abs_grad'] = float(bucket['max_abs_grad'])
        row[f'{prefix}_exp_avg_norm'] = float(exp_avg_norm)
        row[f'{prefix}_exp_avg_sq_norm'] = float(exp_avg_sq_norm)
        row[f'{prefix}_exp_avg_sq_min'] = float(bucket['exp_avg_sq_min']) if bucket['exp_avg_sq_min'] != float('inf') else float('nan')
        row[f'{prefix}_exp_avg_sq_max'] = float(bucket['exp_avg_sq_max']) if bucket['exp_avg_sq_max'] != float('-inf') else float('nan')
        row[f'{prefix}_adam_update_norm'] = float(adam_update_norm)
        row[f'{prefix}_decay_update_norm'] = float(decay_update_norm)
        row[f'{prefix}_combined_update_norm'] = float(combined_update_norm)
        row[f'{prefix}_update_to_weight_ratio'] = float(combined_update_norm / param_norm) if param_norm > 0.0 else float('nan')
    return row

def build_module_debug_row(*, epoch_index: int, global_step: int, step_index: int, model_lr: float, projection_lr: float, total_grad_norm: float, mlm_loss: float, masked_token_count: int, scheduler: torch.optim.lr_scheduler.LambdaLR | None, warmup_steps: int, module_stats: dict[str, float]) -> dict[str, object]:
    scheduler_last_epoch = None if scheduler is None else int(scheduler.state_dict().get('last_epoch', -1))
    scheduler_phase = 'disabled'
    if scheduler is not None:
        scheduler_phase = 'warmup' if scheduler_last_epoch < warmup_steps else 'cosine'
    return {'epoch': int(epoch_index), 'global_step': int(global_step), 'step': int(step_index), 'model_lr': float(model_lr), 'projection_lr': float(projection_lr), 'scheduler_last_epoch': scheduler_last_epoch, 'scheduler_phase': scheduler_phase, 'mlm_loss': float(mlm_loss), 'masked_token_count': int(masked_token_count), 'total_grad_norm': float(total_grad_norm), **module_stats}

def effective_contrastive_weight(*, target_weight: float, candidate_global_step: int, warmup_steps: int) -> float:
    weight = float(target_weight)
    if warmup_steps > 0:
        weight *= min(candidate_global_step / float(warmup_steps), 1.0)
    return weight

def reset_optimizer_state(optimizer: AdamW) -> None:
    optimizer.state.clear()

def sync_optimizer_lrs_from_scheduler(optimizer: AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR | None) -> None:
    if scheduler is None:
        return
    last_lrs = scheduler.get_last_lr()
    for (group, lr) in zip(optimizer.param_groups, last_lrs):
        group['lr'] = float(lr)

def scale_resumed_lr(optimizer: AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR | None, scale: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = float(group['lr']) * scale
        if 'initial_lr' in group:
            group['initial_lr'] = float(group['initial_lr']) * scale
    if scheduler is not None:
        scheduler.base_lrs = [float(lr) * scale for lr in scheduler.base_lrs]
        if hasattr(scheduler, '_last_lr'):
            scheduler._last_lr = [float(lr) * scale for lr in scheduler._last_lr]

def set_resumed_absolute_lr(optimizer: AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR | None, model_lr: float, projection_lr_multiplier: float) -> None:
    target_lrs = [float(model_lr)]
    target_lrs.extend((float(model_lr) * float(projection_lr_multiplier) for _ in optimizer.param_groups[1:]))
    if scheduler is not None:
        old_base_lrs = [float(lr) for lr in scheduler.base_lrs]
        old_last_lrs = [float(lr) for lr in scheduler.get_last_lr()]
        new_base_lrs = []
        for (target_lr, old_base_lr, old_last_lr) in zip(target_lrs, old_base_lrs, old_last_lrs):
            phase_factor = old_last_lr / old_base_lr if old_base_lr > 0.0 else 1.0
            phase_factor = max(phase_factor, 1e-12)
            new_base_lrs.append(float(target_lr) / phase_factor)
        scheduler.base_lrs = new_base_lrs
        if hasattr(scheduler, '_last_lr'):
            scheduler._last_lr = [float(lr) for lr in target_lrs]
    for (index, (group, target_lr)) in enumerate(zip(optimizer.param_groups, target_lrs)):
        group['lr'] = float(target_lr)
        if scheduler is not None and 'initial_lr' in group:
            group['initial_lr'] = float(scheduler.base_lrs[index])

def gradient_group_name(parameter_name: str, *, top_layer_index: int) -> str | None:
    if parameter_name.startswith('word_projection.'):
        return 'word_projection'
    if parameter_name.startswith('projection.'):
        return 'projection'
    if parameter_name.startswith(f'encoder.layers.{top_layer_index}.'):
        return 'top_transformer_layer'
    if parameter_name.startswith('mlm_transform.') or parameter_name.startswith('mlm_layer_norm.') or parameter_name.startswith('mlm_decoder.') or (parameter_name == 'mlm_bias'):
        return 'mlm_head'
    if parameter_name.startswith('token_embeddings.') or parameter_name.startswith('position_embeddings.') or parameter_name.startswith('embedding_layer_norm.'):
        return 'embeddings'
    return None

def collect_gradient_stats(model: ScratchTransformerEncoder, projection: torch.nn.Linear, word_projection: torch.nn.Linear | None=None) -> tuple[float, dict[str, float], list[str]]:
    group_names = ('projection', 'word_projection', 'top_transformer_layer', 'mlm_head', 'embeddings')
    group_sq_norms = {group: 0.0 for group in group_names}
    group_has_grad = {group: False for group in group_names}
    group_has_nonfinite = {group: False for group in group_names}
    nonfinite_grad_names: list[str] = []
    top_layer_index = len(model.encoder.layers) - 1
    total_sq_norm = 0.0
    total_has_grad = False
    total_has_nonfinite = False
    for (name, parameter) in named_trainable_parameters(model, projection, word_projection):
        if parameter.grad is None:
            continue
        group_name = gradient_group_name(name, top_layer_index=top_layer_index)
        if not torch.isfinite(parameter.grad).all():
            nonfinite_grad_names.append(name)
            total_has_nonfinite = True
            if group_name is not None:
                group_has_nonfinite[group_name] = True
            continue
        grad_norm = float(parameter.grad.norm().item())
        total_sq_norm += grad_norm * grad_norm
        total_has_grad = True
        if group_name is not None:
            group_sq_norms[group_name] += grad_norm * grad_norm
            group_has_grad[group_name] = True
    total_grad_norm = float('nan') if total_has_nonfinite else math.sqrt(total_sq_norm) if total_has_grad else 0.0
    grouped_grad_norms: dict[str, float] = {}
    for group_name in group_names:
        if group_has_nonfinite[group_name]:
            grouped_grad_norms[group_name] = float('nan')
        elif group_has_grad[group_name]:
            grouped_grad_norms[group_name] = math.sqrt(group_sq_norms[group_name])
        else:
            grouped_grad_norms[group_name] = 0.0
    return (total_grad_norm, grouped_grad_norms, nonfinite_grad_names)

def tensor_stats(values: torch.Tensor | None, *, prefix: str) -> dict[str, float]:
    if values is None or values.numel() == 0:
        nan = float('nan')
        return {f'{prefix}_min': nan, f'{prefix}_mean': nan, f'{prefix}_max': nan}
    detached = values.detach()
    return {f'{prefix}_min': float(detached.min().item()), f'{prefix}_mean': float(detached.mean().item()), f'{prefix}_max': float(detached.max().item())}

def empty_contrastive_debug_stats() -> dict[str, float]:
    stats = {}
    for prefix in ('student_norm', 'projected_norm', 'teacher_norm', 'positive_logit', 'allowed_negative_logit'):
        stats.update(tensor_stats(None, prefix=prefix))
    stats['mean_positive_margin'] = float('nan')
    stats['max_abs_logit'] = float('nan')
    return stats

def build_contrastive_debug_stats(student_verse_embeddings: torch.Tensor, projected_embeddings: torch.Tensor, flat_teacher_embeddings: torch.Tensor, logits: torch.Tensor, allowed_mask: torch.Tensor) -> dict[str, float]:
    student_norms = student_verse_embeddings.norm(dim=-1)
    projected_norms = projected_embeddings.norm(dim=-1)
    teacher_norms = flat_teacher_embeddings.norm(dim=-1)
    stats = {}
    stats.update(tensor_stats(student_norms, prefix='student_norm'))
    stats.update(tensor_stats(projected_norms, prefix='projected_norm'))
    stats.update(tensor_stats(teacher_norms, prefix='teacher_norm'))
    diag = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    positive_logits = logits.diagonal()
    allowed_negative_mask = allowed_mask & ~diag
    allowed_negative_logits = logits[allowed_negative_mask]
    stats.update(tensor_stats(positive_logits, prefix='positive_logit'))
    stats.update(tensor_stats(allowed_negative_logits, prefix='allowed_negative_logit'))
    stats['max_abs_logit'] = float(logits.detach().abs().max().item()) if logits.numel() > 0 else float('nan')
    if allowed_negative_mask.any():
        row_has_negative = allowed_negative_mask.any(dim=1)
        max_negative_logits = logits.masked_fill(~allowed_negative_mask, float('-inf')).max(dim=1).values
        margins = positive_logits[row_has_negative] - max_negative_logits[row_has_negative]
        stats['mean_positive_margin'] = float(margins.detach().mean().item())
    else:
        stats['mean_positive_margin'] = float('nan')
    return stats

def build_contrastive_debug_row(*, epoch_index: int, global_step: int, step_index: int, verse_weight: float, temperature: float, contrastive_direction: str, model_lr: float, projection_lr: float, projection_weight_norm_value: float, total_grad_norm: float, grouped_grad_norms: dict[str, float], contrastive_debug_stats: dict[str, float]) -> dict[str, object]:
    row = {'epoch': int(epoch_index), 'global_step': int(global_step), 'step': int(step_index), 'verse_weight': float(verse_weight), 'temperature': float(temperature), 'contrastive_direction': contrastive_direction, 'model_lr': float(model_lr), 'projection_lr': float(projection_lr), 'projection_weight_norm': float(projection_weight_norm_value), 'total_grad_norm': float(total_grad_norm), 'projection_grad_norm': float(grouped_grad_norms['projection']), 'word_projection_grad_norm': float(grouped_grad_norms.get('word_projection', 0.0)), 'top_transformer_layer_grad_norm': float(grouped_grad_norms['top_transformer_layer']), 'mlm_head_grad_norm': float(grouped_grad_norms['mlm_head']), 'embeddings_grad_norm': float(grouped_grad_norms['embeddings'])}
    row.update(contrastive_debug_stats)
    return row

def build_nonfinite_failure_row(*, epoch_index: int, global_step: int, step_index: int, verse_weight: float, temperature: float, contrastive_direction: str, error_message: str, failing_batch_path: Path, model_lr: float, projection_lr: float, projection_weight_norm_value: float, nonfinite_grad_names: list[str], total_grad_norm: float, grouped_grad_norms: dict[str, float], contrastive_debug_stats: dict[str, float]) -> dict[str, object]:
    row = {'epoch': int(epoch_index), 'global_step': int(global_step), 'step': int(step_index), 'verse_weight': float(verse_weight), 'temperature': float(temperature), 'contrastive_direction': contrastive_direction, 'error': error_message, 'failing_batch_path': str(failing_batch_path), 'model_lr': float(model_lr), 'projection_lr': float(projection_lr), 'projection_weight_norm': float(projection_weight_norm_value), 'nonfinite_grad_param_count': int(len(nonfinite_grad_names)), 'nonfinite_grad_param_names': '|'.join(nonfinite_grad_names[:16]), 'total_grad_norm': float(total_grad_norm), 'projection_grad_norm': float(grouped_grad_norms['projection']), 'word_projection_grad_norm': float(grouped_grad_norms.get('word_projection', 0.0)), 'top_transformer_layer_grad_norm': float(grouped_grad_norms['top_transformer_layer']), 'mlm_head_grad_norm': float(grouped_grad_norms['mlm_head']), 'embeddings_grad_norm': float(grouped_grad_norms['embeddings'])}
    row.update(contrastive_debug_stats)
    return row

def persist_nonfinite_batch(*, abort_on_nonfinite: bool, batch_loss_rows: list[dict[str, object]], batch_losses_path: Path, skipped_batches_path: Path, nonfinite_failure_path: Path, failing_path: Path, save_failing_batch_kwargs: dict[str, object], failure_row: dict[str, object], skipped_row: dict[str, object], abort_message: str) -> None:
    save_failing_batch(failing_path, **save_failing_batch_kwargs)
    if abort_on_nonfinite:
        append_metrics_csv(batch_loss_rows, batch_losses_path)
        append_metrics_csv([failure_row], nonfinite_failure_path)
        raise RuntimeError(abort_message)
    append_metrics_csv([skipped_row], skipped_batches_path)

def save_failure_training_state(output_path: Path, *, epoch_index: int, step_index: int, global_step: int, model: ScratchTransformerEncoder, projection: torch.nn.Linear, word_projection: torch.nn.Linear | None, optimizer: AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR | None, teacher_dim: int, config: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': int(epoch_index), 'step': int(step_index), 'global_step': int(global_step), 'model_state_dict': model.state_dict(), 'projection_state_dict': projection.state_dict(), 'word_projection_state_dict': None if word_projection is None else word_projection.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(), 'teacher_embedding_dim': int(teacher_dim), 'config': config}, output_path)

def prune_old_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(checkpoint_dir.glob('epoch_*.pt'), key=lambda path: path.stat().st_mtime, reverse=True)
    for old_checkpoint in checkpoints[keep_last:]:
        old_checkpoint.unlink(missing_ok=True)

def save_failing_batch(output_path: Path, *, epoch_index: int, step_index: int, global_step: int, batch: dict[str, object], masked_inputs: torch.Tensor, labels: torch.Tensor, mlm_loss: torch.Tensor | None, contrastive_loss: torch.Tensor | None, total_loss: torch.Tensor | None, verse_weight: float, error_message: str, word_loss: torch.Tensor | None=None, word_loss_scaled: torch.Tensor | None=None, model_lr: float | None=None, projection_lr: float | None=None, projection_weight_norm_value: float | None=None, temperature: float | None=None, contrastive_direction: str | None=None, total_grad_norm: float | None=None, grouped_grad_norms: dict[str, float] | None=None, nonfinite_grad_names: list[str] | None=None, contrastive_debug_stats: dict[str, float] | None=None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'epoch': int(epoch_index), 'step': int(step_index), 'global_step': int(global_step), 'verse_weight': float(verse_weight), 'error_message': error_message, 'input_ids': batch['input_ids'].detach().cpu(), 'attention_mask': batch['attention_mask'].detach().cpu(), 'teacher_embeddings': batch['teacher_embeddings'].detach().cpu(), 'verse_spans': batch['verse_spans'].detach().cpu(), 'verse_mask': batch['verse_mask'].detach().cpu(), 'word_teacher_embeddings': batch['word_teacher_embeddings'].detach().cpu(), 'word_spans': batch['word_spans'].detach().cpu(), 'word_mask': batch['word_mask'].detach().cpu(), 'masked_inputs': masked_inputs.detach().cpu(), 'labels': labels.detach().cpu(), 'poem_ids': list(batch['poem_ids']), 'window_indices': batch['window_indices'].detach().cpu(), 'window_ids': list(batch['window_ids']), 'mlm_loss': None if mlm_loss is None else float(mlm_loss.detach().cpu().item()), 'contrastive_loss': None if contrastive_loss is None else float(contrastive_loss.detach().cpu().item()), 'word_loss': None if word_loss is None else float(word_loss.detach().cpu().item()), 'word_loss_scaled': None if word_loss_scaled is None else float(word_loss_scaled.detach().cpu().item()), 'total_loss': None if total_loss is None else float(total_loss.detach().cpu().item()), 'model_lr': None if model_lr is None else float(model_lr), 'projection_lr': None if projection_lr is None else float(projection_lr), 'projection_weight_norm': None if projection_weight_norm_value is None else float(projection_weight_norm_value), 'temperature': None if temperature is None else float(temperature), 'contrastive_direction': contrastive_direction, 'total_grad_norm': None if total_grad_norm is None else float(total_grad_norm), 'grouped_grad_norms': grouped_grad_norms or {}, 'nonfinite_grad_names': list(nonfinite_grad_names or []), 'contrastive_debug_stats': contrastive_debug_stats or {}}
    torch.save(payload, output_path)

def pool_verse_embeddings(hidden_states: torch.Tensor, verse_spans: torch.Tensor, verse_mask: torch.Tensor) -> torch.Tensor:
    seq_len = hidden_states.size(1)
    span_starts = verse_spans[..., 0]
    span_ends = verse_spans[..., 1]
    if (span_starts < 0).any() or (span_ends < 0).any() or (span_starts > seq_len).any() or (span_ends > seq_len).any() or (span_starts > span_ends).any():
        raise ValueError(f'verse_spans contains invalid boundaries for sequence length {seq_len}: expected 0 <= start <= end <= seq_len')
    span_lengths = span_ends - span_starts
    valid_span_lengths = span_lengths[verse_mask]
    if valid_span_lengths.numel() > 0 and torch.any(valid_span_lengths <= 0):
        raise ValueError('Encountered an empty or invalid verse span while mean-pooling verse embeddings; valid spans must satisfy end > start.')
    prefix = torch.cat([hidden_states.new_zeros(hidden_states.size(0), 1, hidden_states.size(-1)), hidden_states.cumsum(dim=1)], dim=1)
    batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device).unsqueeze(1)
    span_sums = prefix[batch_index, span_ends] - prefix[batch_index, span_starts]
    span_lengths = span_lengths.unsqueeze(-1).clamp_min(1)
    pooled = span_sums / span_lengths
    return pooled[verse_mask]

def compose_interpolated_loss(*, mlm_loss: torch.Tensor, contrastive_loss: torch.Tensor, word_loss: torch.Tensor, mlm_weight: float, verse_weight: float, word_weight: float, word_loss_scale: float=1.0, mlm_available: bool=True, verse_available: bool=True, word_available: bool=True) -> tuple[torch.Tensor, dict[str, float]]:
    word_loss_scale = max(0.0, float(word_loss_scale))
    requested = {'mlm': max(0.0, float(mlm_weight)), 'verse': max(0.0, float(verse_weight)), 'word': max(0.0, float(word_weight))}
    available = {'mlm': bool(mlm_available), 'verse': bool(verse_available), 'word': bool(word_available)}
    active_total = sum((weight for (name, weight) in requested.items() if available[name]))
    if active_total <= 0.0:
        zero = mlm_loss * 0.0 + contrastive_loss * 0.0 + word_loss * 0.0
        return (zero, {'mlm': 0.0, 'verse': 0.0, 'word': 0.0})
    effective = {name: weight / active_total if available[name] else 0.0 for (name, weight) in requested.items()}
    total = effective['mlm'] * mlm_loss + effective['verse'] * contrastive_loss + effective['word'] * word_loss_scale * word_loss
    return (total, effective)

def cross_window_negative_mask(verse_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    (batch_size, max_verses) = verse_mask.shape
    sample_ids = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, max_verses)
    flat_sample_ids = sample_ids[verse_mask.to(device=device)]
    if flat_sample_ids.numel() == 0:
        return torch.empty((0, 0), dtype=torch.bool, device=device)
    return flat_sample_ids.unsqueeze(1) != flat_sample_ids.unsqueeze(0)

def evaluate_epoch(*, model: ScratchTransformerEncoder, projection: torch.nn.Linear, word_projection: torch.nn.Linear | None, loader: DataLoader, tokenizer: SentencePieceTokenizer, config: dict, device: torch.device, max_steps: int | None, contrastive_weight: float, word_weight: float, contrastive_direction: str) -> dict[str, float]:
    model.eval()
    total_mlm_loss = 0.0
    total_contrastive_loss = 0.0
    total_word_loss = 0.0
    total_word_loss_scaled = 0.0
    total_loss = 0.0
    total_tokens = 0.0
    total_aligned_words = 0
    total_batches = 0
    special_token_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id, tokenizer.piece_to_id(config['tokenizer']['mask_token'])}
    temperature = float(config['loss']['temperature'])
    word_loss_scale = float(config['loss'].get('word_loss_scale', 1.0))
    with torch.no_grad():
        for (step_index, batch) in enumerate(loader, start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            teacher_embeddings = batch['teacher_embeddings'].to(device)
            verse_spans = batch['verse_spans'].to(device)
            verse_mask = batch['verse_mask'].to(device)
            word_teacher_embeddings = batch['word_teacher_embeddings'].to(device)
            word_spans = batch['word_spans'].to(device)
            word_mask = batch['word_mask'].to(device)
            (masked_inputs, labels) = apply_mlm_mask(input_ids=input_ids, attention_mask=attention_mask, mask_token_id=tokenizer.piece_to_id(config['tokenizer']['mask_token']), vocab_size=tokenizer.vocab_size, mlm_probability=float(config['training']['mlm_probability']), special_token_ids=special_token_ids)
            outputs = model(input_ids=masked_inputs, attention_mask=attention_mask, return_pooled=False)
            (mlm_loss_sum, token_count) = mlm_loss_and_count(outputs['mlm_logits'], labels)
            mlm_loss = mlm_loss_sum / token_count.clamp_min(1)
            if contrastive_weight > 0.0:
                student_verse_embeddings = pool_verse_embeddings(outputs['last_hidden_state'], verse_spans, verse_mask)
                projected_embeddings = projection(student_verse_embeddings)
                flat_teacher_embeddings = teacher_embeddings[verse_mask]
                contrastive_loss = symmetric_info_nce_loss(projected_embeddings, flat_teacher_embeddings, temperature=temperature, negative_mask=cross_window_negative_mask(verse_mask, device=device), direction=contrastive_direction)
            else:
                contrastive_loss = mlm_loss.new_zeros(())
            aligned_word_count = int(word_mask.sum().item())
            if word_weight > 0.0 and word_projection is not None and (aligned_word_count > 0):
                student_word_embeddings = pool_verse_embeddings(outputs['last_hidden_state'], word_spans, word_mask)
                projected_word_embeddings = word_projection(student_word_embeddings)
                flat_word_teacher_embeddings = word_teacher_embeddings[word_mask].to(projected_word_embeddings.dtype)
                word_loss = weighted_cosine_alignment_loss(projected_word_embeddings, flat_word_teacher_embeddings)
            else:
                word_loss = mlm_loss.new_zeros(())
            mlm_weight = max(0.0, 1.0 - contrastive_weight - word_weight)
            (total, _) = compose_interpolated_loss(mlm_loss=mlm_loss, contrastive_loss=contrastive_loss, word_loss=word_loss, mlm_weight=mlm_weight, verse_weight=contrastive_weight, word_weight=word_weight, word_loss_scale=word_loss_scale, mlm_available=True, verse_available=contrastive_weight > 0.0, word_available=aligned_word_count > 0 and word_projection is not None)
            total_mlm_loss += float(mlm_loss.item())
            total_contrastive_loss += float(contrastive_loss.item())
            total_word_loss += float(word_loss.item())
            total_word_loss_scaled += float((word_loss * word_loss_scale).item())
            total_loss += float(total.item())
            total_tokens += float(token_count.item())
            total_aligned_words += aligned_word_count
            total_batches += 1
            if max_steps is not None and step_index >= max_steps:
                break
    denom = max(total_batches, 1)
    return {'mlm_loss': total_mlm_loss / denom, 'contrastive_loss': total_contrastive_loss / denom, 'word_loss': total_word_loss / denom, 'word_loss_scaled': total_word_loss_scaled / denom, 'word_loss_scale': word_loss_scale, 'total_loss': total_loss / denom, 'masked_tokens': total_tokens, 'aligned_word_count': total_aligned_words, 'num_steps': total_batches}

def main() -> None:
    args = parse_args()
    if args.resume_lr_scale is not None and args.resume_absolute_lr is not None:
        raise SystemExit('Use only one of --resume-lr-scale or --resume-absolute-lr.')
    config = load_config(args.config)
    config.setdefault('loss', {})
    config.setdefault('training', {})
    config['loss'].setdefault('contrastive_direction', 'symmetric')
    config['loss'].setdefault('lambda_v', 0.0)
    config['loss'].setdefault('lambda_w', 0.0)
    config['loss'].setdefault('word_loss_scale', 1.0)
    config['loss'].setdefault('word_warmup_fraction', 0.0)
    config['training'].setdefault('projection_lr_multiplier', 1.0)
    config.setdefault('teacher', {})
    config['teacher'].setdefault('token_model_name', config['teacher'].get('sentence_model_name', 'sentence-transformers/all-mpnet-base-v2'))
    if args.run_name is not None:
        config['run_name'] = args.run_name
    if args.run_dir is not None:
        config['output']['run_dir'] = args.run_dir
    if args.checkpoint_dir is not None:
        config['output']['checkpoint_dir'] = args.checkpoint_dir
    if args.teacher_cache_dir is not None:
        config.setdefault('teacher', {})['cache_dir'] = args.teacher_cache_dir
    if args.max_epochs is not None:
        config['training']['num_epochs'] = min(int(config['training']['num_epochs']), int(args.max_epochs))
    if args.override_lr is not None:
        config['training']['lr'] = float(args.override_lr)
    if args.override_use_scheduler is not None:
        config['training']['use_scheduler'] = args.override_use_scheduler == 'true'
    if args.override_warmup_ratio is not None:
        config['training']['warmup_ratio'] = float(args.override_warmup_ratio)
    if args.override_min_lr_ratio is not None:
        config['training']['min_lr_ratio'] = float(args.override_min_lr_ratio)
    if args.override_lambda_v is not None:
        config['loss']['lambda_v'] = float(args.override_lambda_v)
    if args.override_lambda_w is not None:
        config['loss']['lambda_w'] = float(args.override_lambda_w)
    if args.override_word_loss_scale is not None:
        config['loss']['word_loss_scale'] = float(args.override_word_loss_scale)
    if args.override_temperature is not None:
        config['loss']['temperature'] = float(args.override_temperature)
    if args.override_verse_warmup_fraction is not None:
        config['loss']['verse_warmup_fraction'] = float(args.override_verse_warmup_fraction)
    if args.contrastive_direction is not None:
        config['loss']['contrastive_direction'] = str(args.contrastive_direction)
    if args.projection_lr_multiplier is not None:
        config['training']['projection_lr_multiplier'] = float(args.projection_lr_multiplier)
    target_verse_weight = float(config['loss'].get('lambda_v', 0.0))
    target_word_weight = float(config['loss'].get('lambda_w', 0.0))
    if target_verse_weight < 0.0 or target_word_weight < 0.0 or target_verse_weight + target_word_weight > 1.0:
        raise ValueError('loss.lambda_v and loss.lambda_w must be non-negative and sum to <= 1.')
    if float(config['loss'].get('word_loss_scale', 1.0)) < 0.0:
        raise ValueError('loss.word_loss_scale must be non-negative.')
    run_dir = Path(config['output']['run_dir'])
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    resume = args.resume_checkpoint is not None
    ensure_run_dirs(run_dir, checkpoint_dir, resume=resume)
    save_yaml(config, run_dir / 'config.yaml')
    set_seed(int(config['seed']))
    device_name = args.device if args.device != 'auto' else str(config['training'].get('device', 'auto'))
    device = pick_device(device_name)
    if config['tokenizer'].get('hf_model_name'):
        tokenizer = HFTokenizer(config['tokenizer']['hf_model_name'])
    else:
        tokenizer = SentencePieceTokenizer(config['tokenizer']['model_path'])
    model_config = ScratchEncoderConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=int(config['tokenizer']['max_length']), num_layers=int(config['model']['num_layers']), hidden_size=int(config['model']['hidden_size']), num_heads=int(config['model']['num_heads']), ffn_dim=int(config['model']['ffn_dim']), dropout=float(config['model']['dropout']), pad_token_id=tokenizer.pad_id)
    model = ScratchTransformerEncoder(model_config).to(device)
    init_checkpoint = args.init_checkpoint or config.get('initialization', {}).get('checkpoint_path')
    init_hf_model = args.init_hf_model or config.get('initialization', {}).get('hf_model_name')
    resume_checkpoint = args.resume_checkpoint
    initialized_from = None
    if resume_checkpoint:
        initialized_from = load_optional_checkpoint(model, resume_checkpoint)
    elif init_checkpoint:
        initialized_from = load_optional_checkpoint(model, init_checkpoint)
    elif init_hf_model:
        initialized_from = load_hf_bert_weights(model, init_hf_model)
    split_table = load_split_table(config['data']['verses_path'])
    train_df = split_frame(split_table, config['data']['train_split'])
    dev_df = split_frame(split_table, config['data']['dev_split'])
    if args.max_train_examples is not None:
        train_df = train_df.head(int(args.max_train_examples)).reset_index(drop=True)
    if args.max_dev_examples is not None:
        dev_df = dev_df.head(int(args.max_dev_examples)).reset_index(drop=True)
    from filter_verse_embeddings.teachers.sentence import SentenceTeacher, load_or_compute_teacher_embeddings
    teacher = SentenceTeacher(config['teacher']['sentence_model_name'], device=args.teacher_device if args.teacher_device != 'auto' else str(config['teacher'].get('device', 'auto')))
    teacher_batch_size = int(config['teacher'].get('batch_size', 128))
    teacher_cache_dir = Path(str(config['teacher'].get('cache_dir') or run_dir / 'teacher_cache'))
    train_teacher = load_or_compute_teacher_embeddings(teacher_cache_dir / 'train.npy', teacher, train_df[config['data']['english_text_column']], batch_size=teacher_batch_size, show_progress_bar=True, expected_rows=len(train_df))
    dev_teacher = load_or_compute_teacher_embeddings(teacher_cache_dir / 'dev.npy', teacher, dev_df[config['data']['english_text_column']], batch_size=teacher_batch_size, show_progress_bar=True, expected_rows=len(dev_df))
    teacher_dim = int(train_teacher.shape[1])
    projection = torch.nn.Linear(model_config.hidden_size, teacher_dim, bias=False).to(device)
    word_projection: torch.nn.Linear | None = None
    train_word_rows_by_poem = None
    dev_word_rows_by_poem = None
    train_word_teacher = None
    dev_word_teacher = None
    token_teacher_dim = 0
    if target_word_weight > 0.0:
        from filter_verse_embeddings.teachers.word import TokenTeacher, load_or_compute_token_teacher_embeddings
        word_analysis_path = Path(str(config['data'].get('word_analysis_path', 'artifacts/data/word_analysis_training.parquet')))
        log.info('Loading word-level alignment table from %s', word_analysis_path)
        word_columns = ['poem_id', 'word_id', 'word_position', 'original_form', 'english_translation', 'normalized_orthography', 'lemma_original']
        word_df = pd.read_parquet(word_analysis_path, columns=word_columns)
        train_poems = set(train_df['poem_id'].astype(str))
        dev_poems = set(dev_df['poem_id'].astype(str))
        word_df['poem_id'] = word_df['poem_id'].astype(str)
        train_word_df = word_df[word_df['poem_id'].isin(train_poems)].reset_index(drop=True)
        dev_word_df = word_df[word_df['poem_id'].isin(dev_poems)].reset_index(drop=True)
        token_teacher = TokenTeacher(str(config['teacher']['token_model_name']), device=args.teacher_device if args.teacher_device != 'auto' else str(config['teacher'].get('device', 'auto')))
        token_teacher_dim = token_teacher.hidden_size
        token_cache_dir = Path(str(config['teacher'].get('token_cache_dir') or teacher_cache_dir.parent / 'scr_mlm_vw_token_all_mpnet_base_v2'))
        token_max_length = int(config['teacher'].get('token_max_length', 512))
        token_chunk_rows = int(config['teacher'].get('token_chunk_rows', 128))
        train_token_cache_path = token_teacher_cache_path(token_cache_dir, 'train', train_word_df)
        dev_token_cache_path = token_teacher_cache_path(token_cache_dir, 'dev', dev_word_df)
        train_word_teacher = load_or_compute_token_teacher_embeddings(train_token_cache_path, token_teacher, train_word_df['english_translation'], max_length=token_max_length, chunk_rows=token_chunk_rows, expected_rows=len(train_word_df))
        dev_word_teacher = load_or_compute_token_teacher_embeddings(dev_token_cache_path, token_teacher, dev_word_df['english_translation'], max_length=token_max_length, chunk_rows=token_chunk_rows, expected_rows=len(dev_word_df))
        train_word_rows_by_poem = build_word_rows_by_poem(train_word_df, tokenizer)
        dev_word_rows_by_poem = build_word_rows_by_poem(dev_word_df, tokenizer)
        word_projection = torch.nn.Linear(model_config.hidden_size, token_teacher_dim, bias=False).to(device)
        log.info('Prepared word alignment rows: train=%d dev=%d token_teacher_dim=%d', sum((len(rows) for rows in train_word_rows_by_poem.values())), sum((len(rows) for rows in dev_word_rows_by_poem.values())), token_teacher_dim)
    train_dataset = PackedContrastiveWindowDataset(train_df, tokenizer, train_teacher, word_rows_by_poem=train_word_rows_by_poem, word_teacher_embeddings=train_word_teacher, max_length=int(config['tokenizer']['max_length']), add_bos=bool(config['tokenizer']['add_bos']), add_eos=bool(config['tokenizer']['add_eos']), verse_sep_token=config['tokenizer']['verse_sep_token'], filter_text_column=config['data']['filter_text_column'])
    dev_dataset = PackedContrastiveWindowDataset(dev_df, tokenizer, dev_teacher, word_rows_by_poem=dev_word_rows_by_poem, word_teacher_embeddings=dev_word_teacher, max_length=int(config['tokenizer']['max_length']), add_bos=bool(config['tokenizer']['add_bos']), add_eos=bool(config['tokenizer']['add_eos']), verse_sep_token=config['tokenizer']['verse_sep_token'], filter_text_column=config['data']['filter_text_column'])
    if target_word_weight > 0.0:
        log.info('Train word alignment stats: %s', train_dataset.alignment_stats)
        log.info('Dev word alignment stats: %s', dev_dataset.alignment_stats)
    train_loader = DataLoader(train_dataset, batch_size=int(config['training']['batch_size']), shuffle=True, num_workers=int(config['training'].get('num_workers', 0)), collate_fn=partial(collate_packed_contrastive_batch, pad_id=tokenizer.pad_id))
    dev_loader = DataLoader(dev_dataset, batch_size=int(config['training'].get('eval_batch_size', config['training']['batch_size'])), shuffle=False, num_workers=int(config['training'].get('num_workers', 0)), collate_fn=partial(collate_packed_contrastive_batch, pad_id=tokenizer.pad_id))
    optimizer = build_optimizer(model, projection, word_projection, lr=float(config['training']['lr']), weight_decay=float(config['training']['weight_decay']), projection_lr_multiplier=float(config['training'].get('projection_lr_multiplier', 1.0)))
    scheduler = None
    num_epochs = int(config['training']['num_epochs'])
    scheduler_num_epochs = int(args.scheduler_num_epochs) if args.scheduler_num_epochs is not None else num_epochs
    steps_per_epoch = len(train_loader)
    if args.max_train_steps is not None:
        steps_per_epoch = min(steps_per_epoch, int(args.max_train_steps))
    total_steps = steps_per_epoch * max(num_epochs, 1)
    scheduler_total_steps = steps_per_epoch * max(scheduler_num_epochs, 1)
    if args.scheduler_num_epochs is not None:
        log.info('Using scheduler_num_epochs=%d (%d scheduler steps) with stop num_epochs=%d (%d training steps)', scheduler_num_epochs, scheduler_total_steps, num_epochs, total_steps)
    warmup_steps = 0
    verse_warmup_fraction = float(config['loss'].get('verse_warmup_fraction', 0.0))
    word_warmup_fraction = float(config['loss'].get('word_warmup_fraction', 0.0))
    verse_warmup_steps = int(total_steps * verse_warmup_fraction)
    word_warmup_steps = int(total_steps * word_warmup_fraction)
    gradient_clip = float(config['training']['gradient_clip'])
    temperature = float(config['loss']['temperature'])
    word_loss_scale = float(config['loss'].get('word_loss_scale', 1.0))
    contrastive_direction = str(config['loss'].get('contrastive_direction', 'symmetric'))
    contrastive_stats_every_steps = max(0, int(args.log_contrastive_stats_every_steps))
    module_debug_every_steps = max(0, int(args.module_debug_every_steps))
    metrics_rows: list[dict[str, object]] = []
    global_step = 0
    start_epoch = 1
    skipped_batches_path = run_dir / 'logs' / 'skipped_batches.csv'
    failing_batch_dir = run_dir / 'failing_batches'
    nonfinite_failure_path = run_dir / 'logs' / 'nonfinite_failure.csv'
    contrastive_debug_path = run_dir / 'logs' / 'contrastive_debug.csv'
    module_debug_path = run_dir / 'logs' / 'module_state_debug.csv'
    resume_training_state: dict[str, object] | None = None
    if resume_checkpoint:
        resume_training_state = torch.load(resume_checkpoint, map_location='cpu')
        projection.load_state_dict(resume_training_state['projection_state_dict'])
        if word_projection is not None and resume_training_state.get('word_projection_state_dict') is not None:
            word_projection.load_state_dict(resume_training_state['word_projection_state_dict'])
        if not args.reset_optimizer_state:
            try:
                restore_optimizer_state(optimizer, resume_training_state['optimizer_state_dict'])
            except ValueError:
                log.warning('Optimizer param groups changed; starting with fresh optimizer state.')
        start_epoch = int(resume_training_state['epoch']) + 1
        global_step = int(resume_training_state.get('global_step', 0))
    if bool(config['training'].get('use_scheduler', False)):
        warmup_ratio = float(config['training'].get('warmup_ratio', 0.0))
        min_lr_ratio = float(config['training'].get('min_lr_ratio', 0.0))
        warmup_steps = max(1, int(scheduler_total_steps * warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(1e-08, step / float(warmup_steps))
            progress = (step - warmup_steps) / max(1, scheduler_total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return max(min_lr_ratio, cosine)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if resume_training_state is not None and resume_training_state.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(resume_training_state['scheduler_state_dict'])
            sync_optimizer_lrs_from_scheduler(optimizer, scheduler)
    if resume_checkpoint and args.reset_optimizer_state:
        reset_optimizer_state(optimizer)
    if args.resume_lr_scale is not None:
        scale_resumed_lr(optimizer, scheduler, float(args.resume_lr_scale))
    if args.resume_absolute_lr is not None:
        set_resumed_absolute_lr(optimizer, scheduler, model_lr=float(args.resume_absolute_lr), projection_lr_multiplier=float(config['training'].get('projection_lr_multiplier', 1.0)))
    special_token_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id, tokenizer.piece_to_id(config['tokenizer']['mask_token'])}
    _tb_log_dir = args.tb_log_dir or config.get('output', {}).get('tensorboard_dir') or str(run_dir / 'tensorboard')
    tb_writer = None
    if _TB_AVAILABLE:
        Path(_tb_log_dir).mkdir(parents=True, exist_ok=True)
        tb_writer = _TBSummaryWriter(log_dir=_tb_log_dir)
        log.info('TensorBoard logging enabled: %s', _tb_log_dir)
    else:
        log.warning('torch.utils.tensorboard not available; TensorBoard logging disabled.')
    tb_log_every_steps = max(1, int(args.tb_log_every_steps))
    if args.detect_anomaly:
        log.warning('Autograd anomaly detection enabled for FILTER debug run.')
    last_checkpoint_path = resume_checkpoint
    for epoch_index in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_train_mlm = 0.0
        epoch_train_contrastive = 0.0
        epoch_train_word = 0.0
        epoch_train_word_scaled = 0.0
        epoch_train_total = 0.0
        epoch_train_aligned_words = 0
        epoch_steps = 0
        epoch_skipped_batches = 0
        batch_loss_rows = []
        for (step_index, batch) in enumerate(train_loader, start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            teacher_embeddings = batch['teacher_embeddings'].to(device)
            verse_spans = batch['verse_spans'].to(device)
            verse_mask = batch['verse_mask'].to(device)
            word_teacher_embeddings = batch['word_teacher_embeddings'].to(device)
            word_spans = batch['word_spans'].to(device)
            word_mask = batch['word_mask'].to(device)
            (masked_inputs, labels) = apply_mlm_mask(input_ids=input_ids, attention_mask=attention_mask, mask_token_id=tokenizer.piece_to_id(config['tokenizer']['mask_token']), vocab_size=tokenizer.vocab_size, mlm_probability=float(config['training']['mlm_probability']), special_token_ids=special_token_ids)
            candidate_global_step = global_step + 1
            verse_weight = effective_contrastive_weight(target_weight=target_verse_weight, candidate_global_step=candidate_global_step, warmup_steps=verse_warmup_steps)
            word_weight = effective_contrastive_weight(target_weight=target_word_weight, candidate_global_step=candidate_global_step, warmup_steps=word_warmup_steps)
            mlm_weight = max(0.0, 1.0 - verse_weight - word_weight)
            (model_lr, projection_lr) = current_lrs(optimizer)
            optimizer.zero_grad(set_to_none=True)
            mlm_loss = None
            contrastive_loss = None
            word_loss = None
            word_loss_scaled = None
            total_loss = None
            student_verse_embeddings = None
            projected_embeddings = None
            flat_teacher_embeddings = None
            contrastive_debug_stats = empty_contrastive_debug_stats()
            projection_weight_norm_value = projection_weight_norm(projection)
            total_grad_norm = 0.0
            grouped_grad_norms = {'projection': 0.0, 'word_projection': 0.0, 'top_transformer_layer': 0.0, 'mlm_head': 0.0, 'embeddings': 0.0}
            effective_weights = {'mlm': mlm_weight, 'verse': verse_weight, 'word': word_weight}
            aligned_word_count = int(word_mask.sum().item())
            nonfinite_grad_names: list[str] = []
            failing_path = failing_batch_dir / 'first_failure.pt' if args.abort_on_nonfinite else failing_batch_dir / f'epoch_{epoch_index:04d}_step_{step_index:05d}.pt'
            try:
                outputs = model(input_ids=masked_inputs, attention_mask=attention_mask, return_pooled=False)
                (mlm_loss_sum, token_count) = mlm_loss_and_count(outputs['mlm_logits'], labels)
                mlm_loss = mlm_loss_sum / token_count.clamp_min(1)
                if verse_weight > 0.0:
                    student_verse_embeddings = pool_verse_embeddings(outputs['last_hidden_state'], verse_spans, verse_mask)
                    ensure_finite('student_verse_embeddings', student_verse_embeddings)
                    projected_embeddings = projection(student_verse_embeddings)
                    ensure_finite('projected_embeddings', projected_embeddings)
                    flat_teacher_embeddings = teacher_embeddings[verse_mask]
                    if flat_teacher_embeddings.shape[0] == 0:
                        raise ValueError('Empty flattened teacher embedding batch.')
                    ensure_finite('flat_teacher_embeddings', flat_teacher_embeddings)
                    negative_mask = cross_window_negative_mask(verse_mask, device=device)
                    (contrastive_logits, allowed_mask) = info_nce_logits(projected_embeddings, flat_teacher_embeddings, temperature=temperature, negative_mask=negative_mask)
                    contrastive_debug_stats = build_contrastive_debug_stats(student_verse_embeddings, projected_embeddings, flat_teacher_embeddings, contrastive_logits, allowed_mask)
                    contrastive_loss = symmetric_info_nce_loss(projected_embeddings, flat_teacher_embeddings, temperature=temperature, negative_mask=negative_mask, direction=contrastive_direction)
                else:
                    contrastive_loss = mlm_loss.new_zeros(())
                if word_weight > 0.0 and word_projection is not None and (aligned_word_count > 0):
                    student_word_embeddings = pool_verse_embeddings(outputs['last_hidden_state'], word_spans, word_mask)
                    ensure_finite('student_word_embeddings', student_word_embeddings)
                    projected_word_embeddings = word_projection(student_word_embeddings)
                    ensure_finite('projected_word_embeddings', projected_word_embeddings)
                    flat_word_teacher_embeddings = word_teacher_embeddings[word_mask].to(projected_word_embeddings.dtype)
                    ensure_finite('flat_word_teacher_embeddings', flat_word_teacher_embeddings)
                    word_loss = weighted_cosine_alignment_loss(projected_word_embeddings, flat_word_teacher_embeddings)
                else:
                    word_loss = mlm_loss.new_zeros(())
                word_loss_scaled = word_loss * word_loss_scale
                (total_loss, effective_weights) = compose_interpolated_loss(mlm_loss=mlm_loss, contrastive_loss=contrastive_loss, word_loss=word_loss, mlm_weight=mlm_weight, verse_weight=verse_weight, word_weight=word_weight, word_loss_scale=word_loss_scale, mlm_available=True, verse_available=verse_weight > 0.0, word_available=word_projection is not None and aligned_word_count > 0)
                ensure_finite('mlm_loss', mlm_loss)
                ensure_finite('contrastive_loss', contrastive_loss)
                ensure_finite('word_loss', word_loss)
                ensure_finite('word_loss_scaled', word_loss_scaled)
                ensure_finite('total_loss', total_loss)
                backward_context = torch.autograd.detect_anomaly() if args.detect_anomaly else nullcontext()
                with backward_context:
                    total_loss.backward()
                (total_grad_norm, grouped_grad_norms, nonfinite_grad_names) = collect_gradient_stats(model, projection, word_projection)
                if nonfinite_grad_names:
                    raise ValueError('Non-finite gradients detected after backward().')
                if contrastive_stats_every_steps > 0 and candidate_global_step % contrastive_stats_every_steps == 0:
                    append_metrics_csv([build_contrastive_debug_row(epoch_index=epoch_index, global_step=candidate_global_step, step_index=step_index, verse_weight=verse_weight, temperature=temperature, contrastive_direction=contrastive_direction, model_lr=model_lr, projection_lr=projection_lr, projection_weight_norm_value=projection_weight_norm_value, total_grad_norm=total_grad_norm, grouped_grad_norms=grouped_grad_norms, contrastive_debug_stats=contrastive_debug_stats)], contrastive_debug_path)
                if module_debug_every_steps > 0 and candidate_global_step % module_debug_every_steps == 0:
                    append_metrics_csv([build_module_debug_row(epoch_index=epoch_index, global_step=candidate_global_step, step_index=step_index, model_lr=model_lr, projection_lr=projection_lr, total_grad_norm=total_grad_norm, mlm_loss=float(mlm_loss.item()), masked_token_count=int((labels != -100).sum().item()), scheduler=scheduler, warmup_steps=warmup_steps, module_stats=collect_tracked_module_debug_stats(model, optimizer))], module_debug_path)
                clip_parameters = list(model.parameters()) + list(projection.parameters())
                if word_projection is not None:
                    clip_parameters += list(word_projection.parameters())
                torch.nn.utils.clip_grad_norm_(clip_parameters, gradient_clip, error_if_nonfinite=False)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            except Exception as exc:
                epoch_skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                failure_row = build_nonfinite_failure_row(epoch_index=epoch_index, step_index=step_index, global_step=candidate_global_step, verse_weight=verse_weight, temperature=temperature, contrastive_direction=contrastive_direction, error_message=str(exc), failing_batch_path=failing_path, model_lr=model_lr, projection_lr=projection_lr, projection_weight_norm_value=projection_weight_norm_value, nonfinite_grad_names=nonfinite_grad_names, total_grad_norm=total_grad_norm, grouped_grad_norms=grouped_grad_norms, contrastive_debug_stats=contrastive_debug_stats)
                save_failing_batch_kwargs = {'epoch_index': epoch_index, 'step_index': step_index, 'global_step': candidate_global_step, 'batch': batch, 'masked_inputs': masked_inputs, 'labels': labels, 'mlm_loss': mlm_loss, 'contrastive_loss': contrastive_loss, 'word_loss': word_loss, 'word_loss_scaled': word_loss_scaled, 'total_loss': total_loss, 'verse_weight': verse_weight, 'error_message': str(exc), 'model_lr': model_lr, 'projection_lr': projection_lr, 'projection_weight_norm_value': projection_weight_norm_value, 'temperature': temperature, 'contrastive_direction': contrastive_direction, 'total_grad_norm': total_grad_norm, 'grouped_grad_norms': grouped_grad_norms, 'nonfinite_grad_names': nonfinite_grad_names, 'contrastive_debug_stats': contrastive_debug_stats}
                failure_state_path = failing_path.with_suffix('.state.pt')
                save_failure_training_state(failure_state_path, epoch_index=epoch_index, step_index=step_index, global_step=candidate_global_step, model=model, projection=projection, word_projection=word_projection, optimizer=optimizer, scheduler=scheduler, teacher_dim=teacher_dim, config=config)
                if nonfinite_grad_names:
                    preview = ', '.join(nonfinite_grad_names[:8])
                    suffix = '' if len(nonfinite_grad_names) <= 8 else ', ...'
                    log.error('Non-finite gradients found in %d parameters: %s%s', len(nonfinite_grad_names), preview, suffix)
                skipped_row = {'epoch': epoch_index, 'global_step': candidate_global_step, 'step': step_index, 'verse_weight': float(verse_weight), 'word_weight': float(word_weight), 'model_lr': float(model_lr), 'projection_lr': float(projection_lr), 'projection_weight_norm': float(projection_weight_norm_value), 'nonfinite_grad_param_count': int(len(nonfinite_grad_names)), 'nonfinite_grad_param_names': '|'.join(nonfinite_grad_names[:16]), 'error': str(exc), 'failing_batch_path': str(failing_path)}
                persist_nonfinite_batch(abort_on_nonfinite=args.abort_on_nonfinite, batch_loss_rows=batch_loss_rows, batch_losses_path=run_dir / 'logs' / 'batch_losses.csv', skipped_batches_path=skipped_batches_path, nonfinite_failure_path=nonfinite_failure_path, failing_path=failing_path, save_failing_batch_kwargs=save_failing_batch_kwargs, failure_row=failure_row, skipped_row=skipped_row, abort_message=f'Aborting on first non-finite batch at epoch={epoch_index} step={step_index} global_step={candidate_global_step}: {exc}')
                log.warning('Skipping non-finite batch at epoch=%d step=%d global_step=%d: %s', epoch_index, step_index, candidate_global_step, exc)
                if args.max_train_steps is not None and step_index >= args.max_train_steps:
                    break
                continue
            global_step = candidate_global_step
            if tb_writer is not None and global_step % tb_log_every_steps == 0:
                tb_writer.add_scalar('train_step/total_loss', float(total_loss.item()), global_step)
                tb_writer.add_scalar('train_step/mlm_loss', float(mlm_loss.item()), global_step)
                tb_writer.add_scalar('train_step/contrastive_loss', float(contrastive_loss.item()), global_step)
                tb_writer.add_scalar('train_step/word_loss', float(word_loss.item()), global_step)
                tb_writer.add_scalar('train_step/lr', float(model_lr), global_step)
            batch_loss_rows.append({'epoch': epoch_index, 'global_step': global_step, 'step': step_index, 'mlm_loss': float(mlm_loss.item()), 'contrastive_loss': float(contrastive_loss.item()), 'word_loss': float(word_loss.item()), 'word_loss_scaled': float(word_loss_scaled.item()), 'word_loss_scale': float(word_loss_scale), 'total_loss': float(total_loss.item()), 'verse_weight': float(verse_weight), 'word_weight': float(word_weight), 'w_mlm': float(effective_weights['mlm']), 'w_verse': float(effective_weights['verse']), 'w_word': float(effective_weights['word']), 'aligned_word_count': int(aligned_word_count)})
            epoch_train_mlm += float(mlm_loss.item())
            epoch_train_contrastive += float(contrastive_loss.item())
            epoch_train_word += float(word_loss.item())
            epoch_train_word_scaled += float(word_loss_scaled.item())
            epoch_train_total += float(total_loss.item())
            epoch_train_aligned_words += int(aligned_word_count)
            epoch_steps += 1
            if args.max_train_steps is not None and step_index >= args.max_train_steps:
                break
        append_metrics_csv(batch_loss_rows, run_dir / 'logs' / 'batch_losses.csv')
        dev_metrics = evaluate_epoch(model=model, projection=projection, word_projection=word_projection, loader=dev_loader, tokenizer=tokenizer, config=config, device=device, max_steps=args.max_dev_steps, contrastive_weight=float(verse_weight), word_weight=float(word_weight), contrastive_direction=contrastive_direction)
        train_denom = max(epoch_steps, 1)
        metrics_rows.extend([{'epoch': epoch_index, 'global_step': global_step, 'split': 'train', 'mlm_loss': epoch_train_mlm / train_denom, 'contrastive_loss': epoch_train_contrastive / train_denom, 'word_loss': epoch_train_word / train_denom, 'word_loss_scaled': epoch_train_word_scaled / train_denom, 'word_loss_scale': float(word_loss_scale), 'total_loss': epoch_train_total / train_denom, 'verse_weight': float(verse_weight), 'word_weight': float(word_weight), 'aligned_word_count': int(epoch_train_aligned_words)}, {'epoch': epoch_index, 'global_step': global_step, 'split': 'dev', 'mlm_loss': dev_metrics['mlm_loss'], 'contrastive_loss': dev_metrics['contrastive_loss'], 'word_loss': dev_metrics['word_loss'], 'word_loss_scaled': dev_metrics['word_loss_scaled'], 'word_loss_scale': dev_metrics['word_loss_scale'], 'total_loss': dev_metrics['total_loss'], 'verse_weight': float(verse_weight), 'word_weight': float(word_weight), 'aligned_word_count': int(dev_metrics['aligned_word_count'])}])
        append_metrics_csv(metrics_rows[-2:], run_dir / 'logs' / 'train_metrics.csv')
        if tb_writer is not None:
            train_row = metrics_rows[-2]
            dev_row = metrics_rows[-1]
            for metric in ('total_loss', 'mlm_loss', 'contrastive_loss', 'word_loss'):
                tb_writer.add_scalar(f'train_epoch/{metric}', float(train_row[metric]), epoch_index)
                tb_writer.add_scalar(f'dev_epoch/{metric}', float(dev_row[metric]), epoch_index)
            tb_writer.add_scalar('train_epoch/lr', float(model_lr), epoch_index)
            tb_writer.flush()
        checkpoint_path = checkpoint_dir / f'epoch_{epoch_index:02d}.pt'
        torch.save({'epoch': epoch_index, 'global_step': global_step, 'model_state_dict': model.state_dict(), 'projection_state_dict': projection.state_dict(), 'word_projection_state_dict': None if word_projection is None else word_projection.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(), 'teacher_embedding_dim': int(teacher_dim), 'token_teacher_embedding_dim': int(token_teacher_dim), 'config': config}, checkpoint_path)
        last_checkpoint_path = str(checkpoint_path)
        prune_old_checkpoints(checkpoint_dir, keep_last=int(config['training'].get('keep_last_checkpoints', 0)))
    if tb_writer is not None:
        tb_writer.close()
    train_metadata = {'run_name': config['run_name'], 'model_type': 'scr_mlm_v', 'device': str(device), 'tokenizer_model_path': config['tokenizer'].get('model_path') or config['tokenizer'].get('hf_model_name'), 'teacher_model_name': config['teacher']['sentence_model_name'], 'teacher_embedding_dim': int(teacher_dim), 'initialized_from': initialized_from, 'training_config': config['training'], 'loss_config': config['loss'], 'last_checkpoint': last_checkpoint_path}
    save_yaml(config, run_dir / 'train_config.yaml')
    save_json(train_metadata, run_dir / 'train_metadata.json')
if __name__ == '__main__':
    main()
