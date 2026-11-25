#!/usr/bin/env python3
"""Create a permanent source archive from a reviewed Git commit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--version', default='1.0.0')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    revision = subprocess.check_output(['git', 'rev-parse', args.revision + '^{commit}'], cwd=repo, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True):
        raise RuntimeError('Commit the reviewed release files before creating the permanent archive')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + '.part')
    with partial.open('xb') as stream:
        subprocess.run(['git', 'archive', '--format=tar.gz',
                        '--prefix=filter-verse-embeddings-paper-' + args.version + '/', revision],
                       cwd=repo, stdout=stream, check=True)
    h = hashlib.sha256()
    with partial.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    os.replace(partial, output)
    record = dict(version=args.version, git_commit=revision,
                  repository='https://github.com/hsci-r/filter-verse-embeddings-paper',
                  filename=output.name, bytes=output.stat().st_size, sha256=h.hexdigest())
    output.with_name(output.name + '.manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
