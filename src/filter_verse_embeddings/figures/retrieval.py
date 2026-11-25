"""Paper-style plots for English-mined formulaic-set retrieval results."""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from typing import Iterable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
plt.rcParams.update({'axes.labelweight': 'bold', 'axes.linewidth': 1.35, 'font.weight': 'bold', 'hatch.linewidth': 1.45, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'xtick.major.size': 4.0, 'xtick.major.width': 1.25, 'ytick.major.size': 4.0, 'ytick.major.width': 1.25})
METRIC_MAIN = 'set_p_at_1'
METRIC_SECONDARY = 'set_uap'
METRIC_CROSS_MAIN = 'cross_subset_p_at_1'
SURFACE_ORDER = ['standard', 'versebal', 'finest']
SURFACE_LABELS = {'standard': 'standard', 'versebal': 'verse-balanced', 'finest': 'FinEst WP'}
SURFACE_STYLES = {'standard': {'facecolor': 'white', 'hatch': '', 'edgecolor': 'black', 'linewidth': 1.35}, 'versebal': {'facecolor': '#d9d9d9', 'hatch': '///', 'edgecolor': 'black', 'linewidth': 1.35}, 'finest': {'facecolor': '#969696', 'hatch': '\\\\', 'edgecolor': 'black', 'linewidth': 1.35}}
BASELINE_ORDER = ['char_ngram_tfidf', 'frozen_finestbert_mean_pool', 'multilingual_mpnet_original']
BASELINE_LABELS = {'char_ngram_tfidf': 'char n-gram\nTF-IDF', 'frozen_finestbert_mean_pool': 'frozen\nFinEstBERT', 'multilingual_mpnet_original': 'multilingual\nMPNet'}
BASELINE_LEGEND_LABELS = {'char_ngram_tfidf': 'char TF-IDF', 'frozen_finestbert_mean_pool': 'frozen FinEstBERT', 'multilingual_mpnet_original': 'multilingual MPNet'}
BASELINE_LINE_STYLES = {'char_ngram_tfidf': {'color': '#111111', 'linestyle': '-'}, 'frozen_finestbert_mean_pool': {'color': '#666666', 'linestyle': '--'}, 'multilingual_mpnet_original': {'color': '#999999', 'linestyle': ':'}}
OBJECTIVE_ORDER = ['v0p45_w0p45_mlm0p10', 'v0p33_w0p33_mlm0p34', 'v0p75_w0_mlm0p25', 'v0_w0p75_mlm0p25', 'puremlm']
OBJECTIVE_LABELS = {'v0p45_w0p45_mlm0p10': '.45/.45/.10', 'v0p33_w0p33_mlm0p34': '.33/.33/.34', 'v0p75_w0_mlm0p25': '.75/0/.25', 'v0_w0p75_mlm0p25': '0/.75/.25', 'puremlm': '0/0/1'}
OBJECTIVE_INLINE_LABELS = {'v0p45_w0p45_mlm0p10': '.45/.45/.10', 'v0p33_w0p33_mlm0p34': '.33/.33/.34', 'v0p75_w0_mlm0p25': '.75/0/.25', 'v0_w0p75_mlm0p25': '0/.75/.25', 'puremlm': '0/0/1'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prefix', default='formulaic_set_p99')
    parser.add_argument('--formats', default='jpg,pdf')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--top-n', type=int, default=None)
    return parser.parse_args()

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def fv(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return math.nan
    value = row.get(key, '')
    if value in ('', None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def save_all(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        suffix = fmt.strip().lstrip('.')
        if not suffix:
            continue
        path = output_dir / f'{stem}.{suffix}'
        save_kwargs = {'dpi': dpi, 'bbox_inches': 'tight'}
        if suffix.lower() in {'jpg', 'jpeg'}:
            save_kwargs['pil_kwargs'] = {'quality': 98, 'subsampling': 0}
        fig.savefig(path, **save_kwargs)
        print(f'saved {path}')
        paths.append(path)
    plt.close(fig)
    return paths

def row_label(row: dict[str, str]) -> str:
    if row.get('surface') == 'baseline':
        return BASELINE_LABELS.get(row.get('label', ''), row.get('label', '')).replace('\n', ' ')
    surface = SURFACE_LABELS.get(row.get('surface', ''), row.get('surface', ''))
    objective = OBJECTIVE_INLINE_LABELS.get(row.get('sem_config', ''), row.get('sem_config', ''))
    return f'{surface} {objective}'

def bar_style(row: dict[str, str]) -> dict[str, object]:
    if row.get('surface') == 'baseline':
        return {'facecolor': '#f0f0f0', 'hatch': '..', 'edgecolor': 'black', 'linewidth': 1.35}
    return SURFACE_STYLES.get(row.get('surface', ''), {'facecolor': '#d0d0d0', 'hatch': '', 'edgecolor': 'black', 'linewidth': 1.35})

def sorted_metric_rows(rows: list[dict[str, str]], metric: str, top_n: int | None=None) -> list[dict[str, str]]:
    selected = [row for row in rows if not math.isnan(fv(row, metric))]
    selected.sort(key=lambda row: fv(row, metric), reverse=True)
    if top_n is not None:
        selected = selected[:int(top_n)]
    return selected

def plot_ranked(rows: list[dict[str, str]], output_dir: Path, prefix: str, formats: Iterable[str], dpi: int, top_n: int | None) -> list[Path]:
    ranked = sorted_metric_rows(rows, METRIC_MAIN, top_n=top_n)
    if not ranked:
        raise SystemExit('No rows with formulaic-set P@1 found.')
    y = np.arange(len(ranked))[::-1]
    (fig, ax) = plt.subplots(figsize=(6.35, max(4.8, 0.32 * len(ranked) + 1.2)))
    values = [fv(row, METRIC_MAIN) for row in ranked]
    bars = ax.barh(y, values, height=0.66, linewidth=1.35, zorder=2)
    for (bar, row, value) in zip(bars, ranked, values):
        style = bar_style(row)
        bar.set_facecolor(str(style['facecolor']))
        bar.set_edgecolor(str(style['edgecolor']))
        bar.set_hatch(str(style['hatch']))
        bar.set_linewidth(float(style['linewidth']))
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f'{value:.3f}', va='center', ha='left', fontsize=7.4, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels([row_label(row) for row in ranked], fontsize=8.1, fontweight='bold')
    ax.set_xlabel('Formulaic-set P@1', fontsize=10.6, fontweight='bold')
    ax.set_xlim(0.0, max(0.5, max(values) + 0.045))
    ax.grid(axis='x', linewidth=0.65, alpha=0.38, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.35)
    ax.spines['bottom'].set_linewidth(1.35)
    ax.tick_params(axis='x', labelsize=8.2, width=1.25)
    legend_items = [Patch(**SURFACE_STYLES['standard'], label='standard'), Patch(**SURFACE_STYLES['versebal'], label='verse-balanced'), Patch(**SURFACE_STYLES['finest'], label='FinEst WP'), Patch(facecolor='#f0f0f0', hatch='..', edgecolor='black', linewidth=1.35, label='baseline')]
    fig.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.56, 1.01), ncol=4, fontsize=8.2, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save_all(fig, output_dir, f'{prefix}_ranked', formats, dpi)

def find_row(rows: list[dict[str, str]], surface: str, objective: str) -> dict[str, str] | None:
    for row in rows:
        if row.get('surface') == surface and row.get('sem_config') == objective:
            return row
    return None

def present_surfaces(rows: list[dict[str, str]]) -> list[str]:
    seen = {row.get('surface', '') for row in rows}
    return [surface for surface in SURFACE_ORDER if surface in seen]

def present_objectives(rows: list[dict[str, str]]) -> list[str]:
    seen = {row.get('sem_config', '') for row in rows}
    return [objective for objective in OBJECTIVE_ORDER if objective in seen]

def baseline_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_label = {row.get('label', ''): row for row in rows if row.get('surface') == 'baseline'}
    return [by_label[label] for label in BASELINE_ORDER if label in by_label]

def plot_by_objective(rows: list[dict[str, str]], output_dir: Path, prefix: str, formats: Iterable[str], dpi: int) -> list[Path]:
    surfaces = present_surfaces(rows)
    objectives = present_objectives(rows)
    if not surfaces or not objectives:
        raise SystemExit('No trained surface/objective rows found.')
    x = np.arange(len(objectives), dtype=float)
    width = 0.8 / max(len(surfaces), 1)
    (fig, ax) = plt.subplots(figsize=(3.45, 1.95))
    for (idx, surface) in enumerate(surfaces):
        offset = (idx - (len(surfaces) - 1) / 2) * width
        values = [fv(find_row(rows, surface, objective), METRIC_MAIN) for objective in objectives]
        style = SURFACE_STYLES[surface]
        ax.bar(x + offset, values, width=width * 0.9, facecolor=style['facecolor'], edgecolor=style['edgecolor'], hatch=style['hatch'], linewidth=1.35, label=SURFACE_LABELS[surface], zorder=2)
    baselines = baseline_rows(rows)
    for row in baselines:
        label = row.get('label', '')
        style = BASELINE_LINE_STYLES[label]
        value = fv(row, METRIC_MAIN)
        ax.axhline(value, color=style['color'], linestyle=style['linestyle'], linewidth=1.75, alpha=0.94, zorder=1)
    ax.set_xlim(-0.58, len(objectives) - 0.02)
    ax.set_xticks(x)
    ax.set_xticklabels([OBJECTIVE_LABELS[objective] for objective in objectives], fontsize=6.0, rotation=20, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Objective weights (v / w / m)', fontsize=7.3)
    ax.set_ylabel('Formulaic-set P@1', fontsize=7.7)
    ax.set_ylim(0.0, 0.5)
    ax.grid(axis='y', linewidth=0.65, alpha=0.38, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.35)
    ax.spines['bottom'].set_linewidth(1.35)
    ax.tick_params(axis='y', labelsize=7.2, width=1.25)
    ax.tick_params(axis='x', length=0, pad=2)
    handles: list[object] = [Patch(**SURFACE_STYLES[surface], label=SURFACE_LABELS[surface]) for surface in surfaces]
    handles.extend([Line2D([0], [0], color=BASELINE_LINE_STYLES[row.get('label', '')]['color'], linestyle=BASELINE_LINE_STYLES[row.get('label', '')]['linestyle'], linewidth=1.75, label=BASELINE_LEGEND_LABELS.get(row.get('label', ''), row.get('label', ''))) for row in baselines])
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=3, fontsize=6.3, frameon=True, handlelength=1.8, columnspacing=0.9, borderpad=0.35)
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.28)
    return save_all(fig, output_dir, f'{prefix}_by_objective', formats, dpi)

def plot_cross_subset_delta(rows: list[dict[str, str]], output_dir: Path, prefix: str, formats: Iterable[str], dpi: int) -> list[Path]:
    ranked = sorted_metric_rows(rows, METRIC_MAIN)
    if not ranked:
        raise SystemExit('No rows with formulaic-set P@1 found.')
    top_trained = [row for row in ranked if row.get('surface') != 'baseline'][:3]
    selected = top_trained + baseline_rows(rows)
    selected = [row for row in selected if not math.isnan(fv(row, METRIC_CROSS_MAIN))]
    if not selected:
        raise SystemExit('No rows with cross-subset metrics found.')
    y = np.arange(len(selected))[::-1]
    (fig, ax) = plt.subplots(figsize=(3.55, max(2.05, 0.27 * len(selected) + 0.72)))
    all_scores = [fv(row, METRIC_MAIN) for row in selected]
    cross_scores = [fv(row, METRIC_CROSS_MAIN) for row in selected]
    for (yi, row, all_v, cross_v) in zip(y, selected, all_scores, cross_scores):
        style = '#3f3f3f' if row.get('surface') != 'baseline' else '#8a8a8a'
        ax.plot([cross_v, all_v], [yi, yi], color=style, linewidth=1.5, zorder=1)
        ax.scatter(cross_v, yi, marker='o', s=24, facecolor='white', edgecolor='#111111', linewidth=1.0, zorder=3)
        ax.scatter(all_v, yi, marker='o', s=24, facecolor='#111111', edgecolor='#111111', linewidth=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([row_label(row) for row in selected], fontsize=7.0)
    ax.set_xlabel('P@1', fontsize=8.2)
    ax.set_xlim(0.05, max(all_scores) + 0.04)
    ax.grid(axis='x', linewidth=0.6, alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.25)
    ax.spines['bottom'].set_linewidth(1.25)
    ax.tick_params(axis='x', labelsize=7.1, width=1.2)
    legend_items = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#111111', markeredgecolor='#111111', markersize=5, label='P@1 all'), Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor='#111111', markersize=5, label='P@1 cross-lingual')]
    fig.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=2, fontsize=6.8, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.84], pad=0.25)
    return save_all(fig, output_dir, f'{prefix}_cross_subset_delta', formats, dpi)

def write_summary(rows: list[dict[str, str]], output_dir: Path, prefix: str, paths: list[Path]) -> Path:
    ranked_main = sorted_metric_rows(rows, METRIC_MAIN)
    ranked_uap = sorted_metric_rows(rows, METRIC_SECONDARY)
    baselines = baseline_rows(rows)
    best = ranked_main[0]
    best_uap = ranked_uap[0]
    best_baseline = sorted(baselines, key=lambda row: fv(row, METRIC_MAIN), reverse=True)[0]
    lines = ['# English-Mined Formulaic-Set Plot Summary', '', 'This plot set summarizes the `english_mined_formulaic_set_p99` benchmark.', '', f'- Rows plotted: `{len(ranked_main)}`', f"- Best by P@1: `{best.get('label')}` P@1 `{fv(best, METRIC_MAIN):.4f}`, UAP `{fv(best, METRIC_SECONDARY):.4f}`", f"- Best by UAP: `{best_uap.get('label')}` P@1 `{fv(best_uap, METRIC_MAIN):.4f}`, UAP `{fv(best_uap, METRIC_SECONDARY):.4f}`", f"- Best baseline by P@1: `{best_baseline.get('label')}` P@1 `{fv(best_baseline, METRIC_MAIN):.4f}`, UAP `{fv(best_baseline, METRIC_SECONDARY):.4f}`", '', 'Output files:']
    for path in paths:
        lines.append(f'- `{path}`')
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f'{prefix}_summary.md'
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'saved {summary_path}')
    return summary_path

def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary_csv)
    if not rows:
        raise SystemExit(f'No rows found in {args.summary_csv}')
    formats = [item.strip() for item in str(args.formats).split(',') if item.strip()]
    paths: list[Path] = []
    paths.extend(plot_ranked(rows, args.output_dir, args.prefix, formats, int(args.dpi), args.top_n))
    paths.extend(plot_by_objective(rows, args.output_dir, args.prefix, formats, int(args.dpi)))
    paths.extend(plot_cross_subset_delta(rows, args.output_dir, args.prefix, formats, int(args.dpi)))
    write_summary(rows, args.output_dir, args.prefix, paths)
if __name__ == '__main__':
    main()
