"""Record an explicit, workspace-local migration plan; never execute experiments."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'docs/architecture/migration_20261012'
OLD = Path('netheal-topology-generalization')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    target = AUDIT / 'plan.json'
    if target.exists():
        raise FileExistsError(target)
    actions = []

    def add(old, new, kind):
        old, new = Path(old), Path(new)
        assert (ROOT / old).exists(), old
        assert not (ROOT / new).exists(), new
        assert '..' not in new.parts and not new.is_absolute()
        actions.append(dict(old=old.as_posix(), new=new.as_posix(), kind=kind))

    for path in sorted((ROOT / OLD / 'results').iterdir()):
        if not path.is_dir():
            if path.suffix in ('.json', '.csv', '.log', '.txt') and not path.name.startswith('directory_inventory_'):
                add(path.relative_to(ROOT), Path('runs/legacy_poc/artifacts') / path.name, 'hardlink')
            continue
        name = path.name
        # Active LAYA inputs and official frozen upstream remain in place.
        if name in ('laya_official_20261012_v1', 'netheal_official_jev_20261011_v1'):
            continue
        if name.startswith('netheal_official_jev_'):
            dest = Path('runs/official_benchmark/jev') / name
        elif name.startswith('laya_official_resource'):
            dest = Path('runs/official_benchmark/laya/resource_probes') / name
        elif name.startswith('laya_data_'):
            dest = Path('data/generated/laya') / name
        elif name.startswith(('laya_learning_', 'laya_short_', 'laya_symbol_', 'laya_optimization_', 'laya_observation_')):
            dest = Path('runs/laya_controls') / name
        elif name.startswith('laya_'):
            dest = Path('runs/laya_pilot') / name
        elif name.startswith('api_'):
            dest = Path('runs/api_pilot') / name
        elif name.startswith(('factorial_', 'reliability_')):
            dest = Path('runs/generalization') / name
        elif name == 'archive':
            dest = Path('runs/archive')
        else:
            dest = Path('runs/legacy_poc') / name
        add(path.relative_to(ROOT), dest, 'junction')

    # These modules have no sibling-relative imports or __file__-based roots.
    for name, group in [('metrics_utils', 'evaluation'), ('research_metrics', 'evaluation'), ('controlled_faults', 'environments')]:
        add(OLD / 'src' / f'{name}.py', Path('src/netheal_research') / group / f'{name}.py', 'hardlink')

    for path in sorted((ROOT / 'docs/results').glob('*.md')):
        name = path.name
        group = ('official_benchmark' if name.startswith('netheal_official_') else
                 'laya_controls' if name.startswith('laya_') else
                 'api_pilot' if name.startswith('gpt_jev_') else 'generalization')
        add(path.relative_to(ROOT), Path('docs/research') / group / 'reports' / name, 'document')
    add('docs/specs/laya_handoff_20261012.md', 'docs/handoffs/2026-10-12_laya_handoff.md', 'document')

    protected = [OLD / 'src/run_nonlink_single_fault_audit.py']
    protected += list((ROOT / OLD / 'scripts').glob('*.py'))
    protected += list((ROOT / 'docs/specs').glob('*.md'))
    moved_docs = {a['old'] for a in actions if a['kind'] == 'document'}
    hashes = {}
    for p in protected:
        p = p if p.is_absolute() else ROOT / p
        relative = p.relative_to(ROOT).as_posix()
        if relative not in moved_docs:
            hashes[relative] = digest(p)
    target.write_text(json.dumps({'date': '2026-10-12', 'actions': actions, 'protected_sha256': hashes}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'actions': len(actions), 'protected_files': len(hashes), 'plan': str(target)}))


if __name__ == '__main__':
    main()
