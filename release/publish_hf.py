#!/usr/bin/env python3
"""Upload a verified model directory using the locally configured HF login."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--repo', default='University-of-Helsinki/filter-verse-embeddings')
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    validation = json.loads((root / 'validation.json').read_text())
    if validation.get('status') != 'passed':
        raise ValueError('Model validation has not passed')
    cli_record = Path(__file__).resolve().parent / 'cli_validation.json'
    cli = json.loads(cli_record.read_text())
    if not cli or any(item['exit_code'] for item in cli.values()):
        raise ValueError('Release CLI validation has not passed')
    dependencies = json.loads((cli_record.parent / 'dependency_validation.json').read_text())
    if dependencies.get('status') != 'passed':
        raise ValueError('Reproduction dependency validation has not passed')
    for name, record in manifest['files'].items():
        path = root / name
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'Package checksum mismatch: {name}')
    from huggingface_hub import HfApi
    api = HfApi()
    api.whoami()  # Fail before upload if no authenticated session is available.
    result = api.upload_folder(repo_id=args.repo, repo_type='model', folder_path=str(root),
                               commit_message='Release the paper model and matching tokenizer')
    print(result.commit_url)


if __name__ == '__main__':
    main()
