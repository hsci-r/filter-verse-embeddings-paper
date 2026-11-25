"""Paper-style intrinsic MLM diagnostic plots for Paper summaries."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
SURFACE_ORDER = ['standard', 'versebal', 'finest', 'finestscratch']
SURFACE_LABELS = {'standard': 'standard', 'versebal': 'verse-balanced', 'finest': 'FinEst-init', 'finestscratch': 'FinEst-scratch'}
SURFACE_STYLES = {'standard': {'color': '#111111', 'marker': 'o', 'facecolor': 'white', 'linestyle': '-'}, 'versebal': {'color': '#555555', 'marker': 's', 'facecolor': '#d9d9d9', 'linestyle': '--'}, 'finest': {'color': '#222222', 'marker': '^', 'facecolor': '#969696', 'linestyle': ':'}, 'finestscratch': {'color': '#777777', 'marker': 'D', 'facecolor': 'white', 'linestyle': '-.'}}
OBJECTIVE_ORDER = ['v0p45_w0p45_mlm0p10', 'v0p33_w0p33_mlm0p34', 'v0p75_w0_mlm0p25', 'v0_w0p75_mlm0p25', 'puremlm']
OBJECTIVE_LABELS = {'v0p45_w0p45_mlm0p10': '.45/.45/.10', 'v0p33_w0p33_mlm0p34': '.33/.33/.34', 'v0p75_w0_mlm0p25': '.75/0/.25', 'v0_w0p75_mlm0p25': '0/.75/.25', 'puremlm': '0/0/1'}
OBJECTIVE_MARKERS = {'v0p45_w0p45_mlm0p10': 'o', 'v0p33_w0p33_mlm0p34': 's', 'v0p75_w0_mlm0p25': '^', 'v0_w0p75_mlm0p25': 'D', 'puremlm': 'X'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-csv', type=Path, required=True)
    parser.add_argument('--additional-summary-csv', type=Path, action='append', default=[], help='Append rows from another compatible summary CSV; may be repeated.')
    parser.add_argument('--hf-baseline-json', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prefix', default='intrinsic_mlm_paper')
    parser.add_argument('--formats', default='jpg,pdf')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--only-objective', action='store_true', help='Write only the by-objective paper figure.')
    return parser.parse_args()

def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def read_hf_baseline(path: Path | None) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding='utf-8') as handle:
        data = json.load(handle)
    out = {}
    for key in ('mlm_loss', 'mlm_pseudo_ppl'):
        try:
            out[key] = float(data[key])
        except (KeyError, TypeError, ValueError):
            pass
    return out

def fv(row: Dict[str, str] | None, key: str) -> float:
    if row is None:
        return math.nan
    value = row.get(key, '')
    if value in ('', None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def find_row(rows: List[Dict[str, str]], surface: str, objective: str) -> Dict[str, str] | None:
    for row in rows:
        if row.get('surface') == surface and row.get('sem_config') == objective:
            return row
    return None

def present_surfaces(rows: List[Dict[str, str]]) -> List[str]:
    seen = {row.get('surface', '') for row in rows}
    return [surface for surface in SURFACE_ORDER if surface in seen]

def present_objectives(rows: List[Dict[str, str]]) -> List[str]:
    seen = {row.get('sem_config', '') for row in rows}
    return [objective for objective in OBJECTIVE_ORDER if objective in seen]

def save_all(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        suffix = fmt.strip().lstrip('.')
        if not suffix:
            continue
        path = output_dir / f'{stem}.{suffix}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'saved {path}')
    plt.close(fig)

def surface_handles(surfaces: List[str]) -> List[Line2D]:
    handles = []
    for surface in surfaces:
        style = SURFACE_STYLES[surface]
        handles.append(Line2D([0], [0], color=style['color'], marker=style['marker'], markerfacecolor=style['facecolor'], markeredgecolor='black', linestyle=style['linestyle'], linewidth=1.8, markersize=5.5, label=SURFACE_LABELS[surface]))
    return handles

def surface_marker_handles(surfaces: List[str]) -> List[Line2D]:
    handles = []
    for surface in surfaces:
        style = SURFACE_STYLES[surface]
        handles.append(Line2D([0], [0], color='none', marker=style['marker'], markerfacecolor=style['facecolor'], markeredgecolor='black', markersize=6, label=SURFACE_LABELS[surface]))
    return handles

def objective_handles(objectives: List[str]) -> List[Line2D]:
    return [Line2D([0], [0], color='none', marker=OBJECTIVE_MARKERS[objective], markerfacecolor='#d0d0d0', markeredgecolor='black', markersize=6, label=OBJECTIVE_LABELS[objective]) for objective in objectives]

def plot_objective_loss(rows: List[Dict[str, str]], output_dir: Path, prefix: str, formats: Iterable[str], dpi: int, hf_baseline: Dict[str, float]) -> None:
    surfaces = present_surfaces(rows)
    objectives = present_objectives(rows)
    if not surfaces or not objectives:
        raise SystemExit('No surface/objective rows found for intrinsic MLM plot.')
    x = np.arange(len(objectives), dtype=float)
    (fig, ax) = plt.subplots(figsize=(3.55, 2.55))
    all_values: List[float] = []
    for surface in surfaces:
        style = SURFACE_STYLES[surface]
        values = [fv(find_row(rows, surface, objective), 'mlm_loss') for objective in objectives]
        all_values.extend([value for value in values if not math.isnan(value)])
        ax.plot(x, values, color=style['color'], linestyle=style['linestyle'], linewidth=1.8, marker=style['marker'], markersize=5.5, markerfacecolor=style['facecolor'], markeredgecolor='black', markeredgewidth=0.9, label=SURFACE_LABELS[surface], zorder=3)
    if all_values:
        ymin = min(all_values) - 0.035
        ymax = max(all_values) + 0.04
        ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([OBJECTIVE_LABELS[objective] for objective in objectives], fontsize=7.2)
    ax.set_xlabel('Objective weights (v/w/m)')
    ax.set_ylabel('MLM loss (lower is better)')
    ax.grid(axis='y', linewidth=0.5, alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=surface_handles(surfaces), loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=6.8, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_all(fig, output_dir, f'{prefix}_by_objective', formats, dpi)

def compact_point_label(row: Dict[str, str]) -> str:
    surface = {'standard': 'standard', 'versebal': 'versebal', 'finest': 'FinEst-init', 'finestscratch': 'FinEst-scratch'}.get(row.get('surface', ''), row.get('surface', ''))
    objective = OBJECTIVE_LABELS.get(row.get('sem_config', ''), row.get('sem_config', ''))
    return f'{surface} {objective}'

def plot_loss_vs_balance(rows: List[Dict[str, str]], output_dir: Path, prefix: str, formats: Iterable[str], dpi: int) -> None:
    surfaces = present_surfaces(rows)
    objectives = present_objectives(rows)
    selected = [row for row in rows if row.get('surface') in surfaces and row.get('sem_config') in objectives]
    if not selected:
        raise SystemExit('No rows found for intrinsic MLM scatter plot.')
    (fig, ax) = plt.subplots(figsize=(6.8, 5.1))
    for row in selected:
        surface = row.get('surface', '')
        objective = row.get('sem_config', '')
        style = SURFACE_STYLES.get(surface, SURFACE_STYLES['standard'])
        ax.scatter(fv(row, 'mlm_loss'), fv(row, 'local_translation_balance'), marker=OBJECTIVE_MARKERS.get(objective, 'o'), s=58, facecolor=style['facecolor'], edgecolor='black', linewidth=0.8, zorder=3)
    valid = [row for row in selected if not math.isnan(fv(row, 'mlm_loss')) and (not math.isnan(fv(row, 'local_translation_balance')))]
    if valid:
        best_balance = max(valid, key=lambda row: fv(row, 'local_translation_balance'))
        lowest_loss = min(valid, key=lambda row: fv(row, 'mlm_loss'))
        ax.annotate('best balance\n' + compact_point_label(best_balance), xy=(fv(best_balance, 'mlm_loss'), fv(best_balance, 'local_translation_balance')), xytext=(-88, -28), textcoords='offset points', fontsize=8, ha='right', arrowprops={'arrowstyle': '-', 'linewidth': 0.7, 'color': '#333333'})
        ax.annotate('lowest MLM loss\n' + compact_point_label(lowest_loss), xy=(fv(lowest_loss, 'mlm_loss'), fv(lowest_loss, 'local_translation_balance')), xytext=(10, -28), textcoords='offset points', fontsize=8, arrowprops={'arrowstyle': '-', 'linewidth': 0.7, 'color': '#333333'})
    ax.set_xlabel('MLM loss')
    ax.set_ylabel('Local--translation balance')
    ax.grid(True, linewidth=0.4, alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-0.018, 0.425)
    ax.legend(handles=objective_handles(objectives), loc='lower right', fontsize=8, frameon=True, title='Objective (v/w/m)', title_fontsize=8)
    fig.legend(handles=surface_marker_handles(surfaces), loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_all(fig, output_dir, f'{prefix}_loss_vs_balance', formats, dpi)

def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary_csv)
    for path in args.additional_summary_csv:
        rows.extend(read_rows(path))
    formats = [item.strip() for item in str(args.formats).split(',') if item.strip()]
    hf_baseline = read_hf_baseline(args.hf_baseline_json)
    plot_objective_loss(rows, args.output_dir, args.prefix, formats, args.dpi, hf_baseline)
    if not args.only_objective:
        plot_loss_vs_balance(rows, args.output_dir, args.prefix, formats, args.dpi)
if __name__ == '__main__':
    main()
