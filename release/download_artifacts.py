#!/usr/bin/env python3
"""Mirror selected public Allas objects; preserve bytes and record SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def transfer(root, item):
    deposit = 'embeddings' if item['deposit'] == 'Zenodo-embeddings' else 'data'
    target = root / deposit / item['path']
    receipt = target.with_name(target.name + '.receipt.json')
    if target.exists() or receipt.exists():
        if not (target.exists() and receipt.exists()):
            raise FileExistsError(f'Unverified existing target: {target}')
        record = json.loads(receipt.read_text())
        h = hashlib.sha256()
        with target.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        if (record['source'] != item or target.stat().st_size != item['bytes']
                or h.hexdigest() != record['sha256']):
            raise ValueError(f'Existing file does not match its transfer receipt: {target}')
        return record
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    if partial.exists():
        raise FileExistsError(f'Incomplete prior download: {partial}; inspect before retrying')
    with urlopen(Request(item['url'], headers={'If-Match': '"' + item['etag'] + '"'}), timeout=180) as response:
        if int(response.headers['Content-Length']) != item['bytes']:
            raise ValueError(f'Changed source size: {item["url"]}')
        if response.headers.get('ETag', '').strip('"') != item['etag']:
            raise ValueError(f'Changed source ETag: {item["url"]}')
        h = hashlib.sha256()
        size = 0
        with partial.open('xb') as stream:
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
                stream.write(chunk)
                h.update(chunk)
                size += len(chunk)
    if size != item['bytes']:
        raise ValueError(f'Incomplete download: {target}')
    os.replace(partial, target)
    record = dict(path=str(target.relative_to(root)), bytes=size, sha256=h.hexdigest(), source=item)
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Verified {record["path"]}: {size} bytes', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=Path('data/allas_sources.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    items = json.loads(args.sources.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda item: transfer(args.output, item), items))
    (args.output / 'allas_transfer_manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
