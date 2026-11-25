#!/usr/bin/env python3
"""Upload, verify and publish a prepared four-record Zenodo release; resume by ID."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

API = 'https://zenodo.org/api'
USER_AGENT = 'FILTER-paper-release/1.0 (+https://github.com/hsci-r/filter-verse-embeddings-paper)'


def request(session, method, url, **kwargs):
    if urlparse(url).scheme != 'https' or urlparse(url).netloc != 'zenodo.org':
        raise ValueError('Unexpected Zenodo API link')
    response = session.request(method, url, timeout=(30, 600), **kwargs)
    if not response.ok:
        raise RuntimeError(f'Zenodo {response.status_code}: {response.text[:2500]}')
    return response.json()


def verify_local(record):
    path = Path(record['path'])
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            md5.update(chunk)
    if (path.stat().st_size != record['bytes'] or sha.hexdigest() != record['sha256']
            or md5.hexdigest() != record['md5']):
        raise ValueError('Local archive checksum mismatch: ' + record['filename'])


def verify_remote(remote, expected):
    size = remote.get('filesize', remote.get('size'))
    checksum = remote['checksum'].removeprefix('md5:')
    if size != expected['bytes'] or checksum != expected['md5']:
        raise ValueError('Uploaded file checksum mismatch: ' + expected['filename'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    parser.add_argument('--draft-only', action='store_true', help='Upload records without publishing them.')
    args = parser.parse_args()
    token = os.environ.get('ZENODO_TOKEN')
    if not token:
        raise SystemExit('Set ZENODO_TOKEN in your terminal before submitting this job')
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    state_path = args.plan.parent / 'zenodo_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(plan_sha256=plan_hash, deposits={})
    if state['plan_sha256'] != plan_hash:
        raise ValueError('Upload plan changed; preserve the original plan and its existing Zenodo records')

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(temporary, state_path)

    session = requests.Session()
    session.headers.update(Authorization='Bearer ' + token, **{'User-Agent': USER_AGENT})
    # Check all local bytes before creating or changing any remote record.
    for group in plan['deposits'].values():
        for file in group['files']:
            verify_local(file)
    drafts = {}
    for name, group in plan['deposits'].items():
        previous = state['deposits'].get(name)
        if previous:
            draft = request(session, 'GET', API + '/deposit/depositions/' + str(previous['id']))
        else:
            metadata = dict(group['metadata'], prereserve_doi=True)
            draft = request(session, 'POST', API + '/deposit/depositions', json={'metadata': metadata})
            state['deposits'][name] = dict(id=draft['id'], files={})
            save()
        drafts[name] = draft
        print(f'{name}: Zenodo record {draft["id"]}', flush=True)
    # Link the reserved artifact DOIs; the accepted paper has no assigned DOI.
    dois = {name: draft['metadata'].get('prereserve_doi', {}).get('doi') or draft.get('doi')
            for name, draft in drafts.items()}
    if not all(dois.values()):
        raise ValueError('Zenodo did not return reserved identifiers for all records')
    for name, group in plan['deposits'].items():
        draft = drafts[name]
        if draft.get('submitted'):
            continue
        metadata = dict(group['metadata'])
        metadata['related_identifiers'] = list(metadata.get('related_identifiers', [])) + [
            dict(identifier=doi, relation='references') for other, doi in dois.items() if other != name]
        drafts[name] = request(session, 'PUT', API + '/deposit/depositions/' + str(draft['id']),
                               json={'metadata': metadata})
    # All four metadata records are accepted before the large file transfers.
    for name, group in plan['deposits'].items():
        draft = drafts[name]
        remote_files = request(session, 'GET', API + '/deposit/depositions/' + str(draft['id']) + '/files')
        existing = {r.get('filename', r.get('name')): r for r in remote_files}
        expected_names = {r['filename'] for r in group['files']}
        if set(existing) - expected_names:
            raise ValueError('Unexpected files in Zenodo record: ' + name)
        for file in group['files']:
            if file['filename'] in existing:
                verify_remote(existing[file['filename']], file)
            else:
                if draft.get('submitted'):
                    raise ValueError('Published record is missing an expected file')
                with Path(file['path']).open('rb') as stream:
                    response = request(session, 'PUT', draft['links']['bucket'].rstrip('/') + '/' +
                                       quote(file['filename'], safe=''), data=stream,
                                       headers={'Content-Type': 'application/octet-stream'})
                verify_remote(response, file)
            state['deposits'][name]['files'][file['filename']] = file['sha256']
            save()
            print(f'{name}: verified {file["filename"]}', flush=True)
    # Publish only after every planned file in all four records passed verification.
    for name, draft in drafts.items():
        if not args.draft_only and not draft.get('submitted'):
            draft = request(session, 'POST', draft['links']['publish'])
        record = state['deposits'][name]
        record['doi'] = dois[name]
        record['published'] = bool(draft.get('submitted'))
        record['url'] = draft.get('record_url') or draft['links'].get('html')
        save()
        print(json.dumps(dict(group=name, **record), indent=2), flush=True)


if __name__ == '__main__':
    main()
