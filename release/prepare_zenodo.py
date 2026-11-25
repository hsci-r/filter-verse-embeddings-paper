#!/usr/bin/env python3
"""Build versioned archives and a checksum-verified Zenodo upload plan in Slurm."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def fingerprints(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            md5.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=sha.hexdigest(), md5=md5.hexdigest())


def archive(target, root, files):
    if target.exists():
        raise FileExistsError(target)
    with target.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as tar:
            for path in sorted(files):
                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with path.open('rb') as stream:
                    tar.addfile(info, stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--hf-revision', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    assets, output = args.assets.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'upload_plan.json').exists():
        raise FileExistsError('A completed upload plan exists; use that plan or a new output directory')
    cli = json.loads((repo / 'release/cli_validation.json').read_text())
    assert cli and all(item['exit_code'] == 0 for item in cli.values())
    assert json.loads((assets / 'model/validation.json').read_text())['status'] == 'passed'
    assert json.loads((assets / 'embeddings/coverage.json').read_text())['same_identity_multiset']
    # Validate every supplied artifact against its frozen manifest before archiving.
    for group in ['model', 'data', 'embeddings']:
        manifest = json.loads((assets / group / 'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            actual = fingerprints(assets / group / name)
            if actual['bytes'] != expected['bytes'] or actual['sha256'] != expected['sha256']:
                raise ValueError(f'Artifact changed: {group}/{name}')
    code = output / 'filter-verse-embeddings-code-v1.0.0.tar.gz'
    subprocess.run([sys.executable, str(repo / 'release/archive_code.py'),
                    '--output', str(code), '--revision', args.code_revision], check=True)
    model = output / 'filter-verse-embeddings-model-v1.0.0.tar.gz'
    archive(model, assets / 'model', [p for p in (assets / 'model').rglob('*') if p.is_file()])
    direct_data = ['verses_translated.parquet', 'word_analysis.parquet', 'paper_verses.parquet']
    support = output / 'filter-verse-embeddings-data-support-v1.0.0.tar.gz'
    archive(support, assets / 'data', [p for p in (assets / 'data').rglob('*')
        if p.is_file() and str(p.relative_to(assets / 'data')) not in direct_data])
    groups = {
        'code': [code, code.with_name(code.name + '.manifest.json'), repo / 'README.md', repo / 'CITATION.bib'],
        'model': [model, assets / 'model/README.md', assets / 'model/CITATION.bib'],
        'data': [assets / 'data' / name for name in direct_data] + [support, assets / 'data/README.md', assets / 'data/CITATION.bib'],
        'embeddings': [p for p in (assets / 'embeddings').iterdir() if p.is_file()],
    }
    plan = dict(version='1.0.0', code_revision=args.code_revision, hf_revision=args.hf_revision, deposits={})
    for group, paths in groups.items():
        metadata = json.loads((repo / 'release/zenodo' / (group + '.json')).read_text())
        rows = [dict(filename=p.name, path=str(p), **fingerprints(p)) for p in sorted(paths)]
        if len({r['filename'] for r in rows}) != len(rows):
            raise ValueError('Duplicate Zenodo filename')
        checksums = output / (group + '-SHA256SUMS')
        checksums.write_text(''.join(r['sha256'] + '  ' + r['filename'] + '\n' for r in rows))
        rows.append(dict(filename=checksums.name, path=str(checksums), **fingerprints(checksums)))
        assert len(rows) <= 100 and sum(r['bytes'] for r in rows) < 50_000_000_000
        plan['deposits'][group] = dict(metadata=metadata, files=rows)
    (output / 'upload_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({k: dict(files=len(v['files']), bytes=sum(x['bytes'] for x in v['files']))
                      for k, v in plan['deposits'].items()}, indent=2))


if __name__ == '__main__':
    main()
