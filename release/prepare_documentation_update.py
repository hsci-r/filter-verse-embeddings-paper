#!/usr/bin/env python3
"""Prepare corrected release archives while preserving all scientific artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
SOURCE = 'https://github.com/hsci-r/filter-data'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def copy_file(source, target):
    if Path(source).stat().st_size > 128 * 1024:
        os.link(source, target)
        return target
    return shutil.copy2(source, target)


def prepare(args):
    old_plan = json.loads(args.original_plan.read_text())
    state_path = args.original_state or args.original_plan.with_name('zenodo_state.json')
    old_state = json.loads(state_path.read_text())
    published_records = old_state.get('deposits') or old_state.get('records')
    if not published_records:
        raise ValueError('The previous publication state has no record identifiers')
    revision = subprocess.check_output(['git', 'rev-parse', args.code_revision + '^{commit}'],
                                       cwd=REPO, text=True).strip()
    completed = args.output / 'archives/upload_plan.json'
    if completed.exists():
        previous = json.loads(completed.read_text())
        if previous['code_revision'] != revision or not all(
                g.get('record_id') == published_records[n]['id'] for n, g in previous['deposits'].items()):
            raise ValueError('Existing prepared update belongs to a different revision or record set')
        print('Using prepared documentation update: ' + str(completed), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    staged = args.output / 'assets'
    staged.mkdir()  # Preserve an interrupted preparation rather than overwrite it.
    for group in ['model', 'data', 'embeddings']:
        shutil.copytree(args.assets / group, staged / group, copy_function=copy_file,
                        ignore=shutil.ignore_patterns('.cache', '__pycache__', '*.pyc'))
        readme = REPO / {'model': 'release/model_card.md', 'data': 'docs/data.md',
                         'embeddings': 'docs/embeddings.md'}[group]
        shutil.copy2(readme, staged / group / 'README.md')
        shutil.copy2(REPO / 'CITATION.bib', staged / group / 'CITATION.bib')
        if group == 'model':
            shutil.copy2(REPO / 'release/model_card_validation.json', staged / group / 'model_card_validation.json')
        else:
            (staged / group / 'LICENSE_GENERATED.md').write_text(
                '# Licence\n\nGenerated translations, word analyses and embeddings are licensed under '
                '[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). '
                'Please cite the paper using the BibTeX entry in [the documentation]('
                'https://github.com/hsci-r/filter-verse-embeddings-paper#citation).\n\n'
                'Original corpus text, identifiers and metadata retain the terms described in '
                '[FILTER data](' + SOURCE + '#license).\n')
        for path in (staged / group).rglob('*.receipt.json'):
            original = json.loads(path.read_text())
            write_json(path, dict(path=original['path'], bytes=original['bytes'], sha256=original['sha256'],
                upstream_repository=SOURCE,
                distribution='https://zenodo.org/records/' + str(published_records[group]['id'])))
        if group == 'data':
            readiness_path = staged / group / 'readiness.json'
            readiness = json.loads(readiness_path.read_text())
            readiness['word_order_equivalence'] = 'Original training-order mapping is unavailable'
            readiness.pop('translation_model_generation', None)
            readiness['translation_model'] = 'unsloth/DeepSeek-R1-BF16'
            write_json(readiness_path, readiness)
        manifest_path = staged / group / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for path in (staged / group).rglob('*'):
            if not path.is_file() or path.name in {'manifest.json', 'SHA256SUMS'}:
                continue
            name = str(path.relative_to(staged / group))
            if path.stat().st_size <= 128 * 1024 or name not in manifest['files']:
                manifest['files'][name] = dict(bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        write_json(manifest_path, manifest)
        if group == 'model':
            (staged / group / 'SHA256SUMS').write_text(''.join(
                record['sha256'] + '  ' + name + '\n' for name, record in sorted(manifest['files'].items())))
    # Validate and archive the staged files using the existing release preparer.
    subprocess.run([sys.executable, str(REPO / 'release/prepare_zenodo.py'),
                    '--assets', str(staged), '--output', str(args.output / 'archives'),
                    '--code-revision', revision, '--hf-revision', old_plan['hf_revision']], check=True)
    plan = json.loads(completed.read_text())
    for name, group in plan['deposits'].items():
        saved = published_records[name]
        if not saved['published']:
            raise ValueError('Expected an already published record: ' + name)
        group['record_id'] = saved['id']
        group['doi'] = saved['doi']
        group['previous_files'] = [{k: f[k] for k in ['filename', 'bytes', 'md5']}
                                   for f in old_plan['deposits'][name]['files']]
    # Each previously published direct Parquet must remain identical.
    for name, old_group in old_plan['deposits'].items():
        current = {f['filename']: f for f in plan['deposits'][name]['files']}
        for original in old_group['files']:
            if original['filename'].endswith('.parquet'):
                actual = current[original['filename']]
                if any(actual[k] != original[k] for k in ['bytes', 'sha256', 'md5']):
                    raise ValueError('Scientific artifact changed: ' + original['filename'])
    write_json(completed, plan)
    print('Prepared documentation update: ' + str(completed), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--original-plan', type=Path, required=True)
    parser.add_argument('--original-state', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-revision', default='HEAD')
    args = parser.parse_args()
    args.assets, args.original_plan, args.output = args.assets.resolve(), args.original_plan.resolve(), args.output.resolve()
    if args.original_state is not None:
        args.original_state = args.original_state.resolve()
    prepare(args)


if __name__ == '__main__':
    main()
