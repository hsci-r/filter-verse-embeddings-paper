#!/usr/bin/env python3
"""Publish the finalized model card and citation without uploading weights."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--repo', default='University-of-Helsinki/filter-verse-embeddings')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import CommitOperationAdd, HfApi
    root = args.directory
    record = json.loads((root / 'model_card_validation.json').read_text())
    if record['status'] != 'passed' or record['card_sha256'] != hashlib.sha256((root / 'README.md').read_bytes()).hexdigest():
        raise ValueError('The updated model card has not passed Hub validation')
    api = HfApi()
    api.whoami()
    current = api.model_info(args.repo)
    if args.receipt.exists() and json.loads(args.receipt.read_text())['revision'] == current.sha:
        print('Hugging Face documentation is already published: ' + str(current.sha), flush=True)
        return
    files = ['README.md', 'CITATION.bib', 'manifest.json', 'SHA256SUMS', 'model_card_validation.json']
    result = api.create_commit(repo_id=args.repo, repo_type='model', parent_commit=current.sha,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name)) for name in files],
        commit_message='Finalize documentation, resource links and paper citation')
    args.receipt.write_text(json.dumps(dict(repository=args.repo, revision=result.oid,
        commit_url=result.commit_url, updated_files=files), indent=2) + '\n')
    print(result.commit_url, flush=True)


if __name__ == '__main__':
    main()
