"""Read-only verification of migrated identities, protected hashes, and local links."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'docs/architecture/migration_20261012'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((AUDIT / 'plan.json').read_text(encoding='utf-8'))
    finalized = json.loads((AUDIT / 'finalized.json').read_text(encoding='utf-8'))
    files = [json.loads(x) for x in (AUDIT / 'files.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    docs = {a['new'] for a in plan['actions'] if a['kind'] == 'document'}
    for path, digest in plan['protected_sha256'].items():
        assert sha(ROOT / path) == digest, f'Protected source changed: {path}'
    for record in files:
        old, new = ROOT / record['old'], ROOT / record['new']
        if Path(record['new']).as_posix() in docs:
            assert sha(new) == finalized['documents'][Path(record['new']).as_posix()]
            assert old.is_file()
        else:
            assert os.path.samefile(old, new), f'Compatibility identity lost: {old}'
            assert new.stat().st_size == record['bytes'], f'Size changed: {new}'
    # Full before/after hashes were checked by apply_structure_migration.ps1;
    # no second 11GB hash pass is needed to test link identity.
    for action in plan['actions']:
        if action['kind'] == 'junction':
            assert (ROOT / action['old']).resolve() == (ROOT / action['new']).resolve()

    checked_docs = [ROOT / path for path in docs]
    checked_docs += [ROOT / name for name in ('README.md', 'docs/README.md', 'docs/architecture/README.md', 'experiments/README.md', 'runs/README.md', 'data/README.md', 'data/manifests/README.md', 'configs/README.md', 'tests/README.md')]
    checked_docs += list((ROOT / 'docs/research').glob('*/README.md'))
    broken, links = [], 0
    for doc in checked_docs:
        text = doc.read_text(encoding='utf-8-sig')
        # Fenced examples can contain sample links, so ignore those blocks.
        text = re.sub(r'```.*?```', '', text, flags=re.S)
        for match in re.finditer(r'\]\(([^)]+)\)', text):
            value = match.group(1)
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', value) or value.startswith('#'):
                continue
            target = unquote(value.partition('#')[0])
            links += 1
            if not (doc.parent / target).exists():
                broken.append({'document': str(doc.relative_to(ROOT)), 'target': target})

    entrypoints = 0
    for config_path in (ROOT / 'experiments').rglob('config.json'):
        config = json.loads(config_path.read_text(encoding='utf-8'))
        for spec in config['stages'].values():
            assert (ROOT / spec['python']).is_file(), spec
            if 'script' in spec:
                assert (ROOT / spec['script']).is_file(), spec
            else:
                assert (ROOT / spec['cwd'] / (spec['module'].replace('.', '/') + '.py')).is_file(), spec
        ast.parse(config_path.with_name('run.py').read_text(encoding='utf-8'))
        entrypoints += 1
    summary = {'status': 'passed' if not broken else 'links_need_review', 'actions': len(plan['actions']), 'action_types': dict(Counter(a['kind'] for a in plan['actions'])), 'moved_files': len(files), 'moved_bytes': sum(f['bytes'] for f in files), 'protected_hashes_unchanged': len(plan['protected_sha256']), 'local_links_checked': links, 'broken_links': broken, 'experiment_entrypoints': entrypoints}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if broken:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
