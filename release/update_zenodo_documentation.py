#!/usr/bin/env python3
"""Correct documentation on the four published records during the grace period."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

import requests
from upload_zenodo import API, USER_AGENT, request, verify_local, verify_remote


def same_file(remote, expected):
    return (remote.get('filesize', remote.get('size')) == expected['bytes'] and
            remote['checksum'].removeprefix('md5:') == expected['md5'])


def remote_files(session, record_id):
    files = request(session, 'GET', API + '/deposit/depositions/' + str(record_id) + '/files')
    return {f.get('filename', f.get('name')): f for f in files}


def validate_inventory(files, group):
    current = {f['filename']: f for f in group['files']}
    previous = {f['filename']: f for f in group['previous_files']}
    if set(files) - set(current):
        raise ValueError('Unexpected remote files; no files were removed')
    for name, file in files.items():
        if not same_file(file, current[name]) and (name not in previous or not same_file(file, previous[name])):
            raise ValueError('Remote file differs from both reviewed versions: ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    token = os.environ.get('ZENODO_TOKEN')
    if not token:
        raise SystemExit('Set ZENODO_TOKEN before submitting the update')
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    plan_hash = hashlib.sha256(raw).hexdigest()
    state_path = args.plan.with_name('documentation_update_state.json')
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(plan_sha256=plan_hash, records={})
    if state['plan_sha256'] != plan_hash:
        raise ValueError('The prepared update changed; keep its original plan and state')

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(temporary, state_path)

    session = requests.Session()
    session.headers.update(Authorization='Bearer ' + token, **{'User-Agent': USER_AGENT})
    request(session, 'GET', API + '/vocabularies/licenses/cc-by-4.0')
    # Check all local and remote versions before opening any edit session.
    for name, group in plan['deposits'].items():
        for file in group['files']:
            verify_local(file)
        validate_inventory(remote_files(session, group['record_id']), group)
        state['records'].setdefault(name, dict(id=group['record_id'], doi=group['doi']))
    save()
    drafts = {}
    for name, group in plan['deposits'].items():
        saved = state['records'][name]
        if saved.get('published'):
            continue
        base = API + '/deposit/depositions/' + str(group['record_id'])
        draft = request(session, 'GET', base)
        if not saved.get('editing'):
            request(session, 'POST', draft['links']['edit'])
            saved['editing'] = True
            save()
        if not saved.get('files_unlocked'):
            result = request(session, 'POST', API + '/records/' + str(group['record_id']) + '/file-modification',
                json={'comment': 'Correct release READMEs, resource links, source attribution and paper citation. '
                                 'Model weights, tokenizer and scientific data remain unchanged.'})
            if result.get('status') != 'accepted':
                raise RuntimeError('Zenodo did not immediately enable file corrections; inspect the record before resubmitting')
            saved['files_unlocked'] = True
            save()
        draft = request(session, 'PUT', base, json={'metadata': group['metadata']})
        files = remote_files(session, group['record_id'])
        validate_inventory(files, group)
        previous = {f['filename']: f for f in group['previous_files']}
        for file in group['files']:
            name_in_record = file['filename']
            existing = files.get(name_in_record)
            if existing is not None and same_file(existing, file):
                continue
            if existing is not None:
                verify_remote(existing, previous[name_in_record])
                request(session, 'DELETE', base + '/files/' + quote(str(existing['id']), safe=''))
            with Path(file['path']).open('rb') as stream:
                uploaded = request(session, 'PUT', draft['links']['bucket'].rstrip('/') + '/' +
                    quote(name_in_record, safe=''), data=stream, headers={'Content-Type': 'application/octet-stream'})
            verify_remote(uploaded, file)
            print(name + ': updated ' + name_in_record, flush=True)
        final_files = remote_files(session, group['record_id'])
        if set(final_files) != {f['filename'] for f in group['files']}:
            raise ValueError('Incomplete corrected record: ' + name)
        for file in group['files']:
            verify_remote(final_files[file['filename']], file)
        drafts[name] = draft
    # Finish all four corrected file sets before publishing any edited record.
    for name, draft in drafts.items():
        published = request(session, 'POST', draft['links']['publish'])
        if not published.get('submitted'):
            raise ValueError('Zenodo did not confirm publication: ' + name)
        state['records'][name]['published'] = True
        state['records'][name]['url'] = 'https://zenodo.org/records/' + str(plan['deposits'][name]['record_id'])
        save()
        print(name + ': published ' + state['records'][name]['url'], flush=True)


if __name__ == '__main__':
    main()
