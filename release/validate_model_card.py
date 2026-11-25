#!/usr/bin/env python3
"""Check a model card against the Hub's metadata validator without uploading."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('card', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.card.read_bytes()
    payload = json.dumps({'content': raw.decode('utf-8'), 'repoType': 'model'}).encode('utf-8')
    request = Request('https://huggingface.co/api/validate-yaml', data=payload,
        headers={'Content-Type': 'application/json', 'User-Agent': 'FILTER-paper-release/1.0'})
    try:
        with urlopen(request, timeout=60) as response:
            status = response.status
            answer = json.loads(response.read().decode('utf-8'))
    except HTTPError as error:
        print(error.read().decode('utf-8'))
        raise
    if status != 200 or answer.get('errors'):
        raise ValueError(f'Model card validation failed: {answer}')
    record = dict(status='passed', validator='https://huggingface.co/api/validate-yaml',
                  card_sha256=hashlib.sha256(raw).hexdigest(), response=answer)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
