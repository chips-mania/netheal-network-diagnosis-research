"""Finish documentation and experiment entry points after the recorded physical moves."""
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'docs/architecture/migration_20261012'
PLAN = json.loads((AUDIT / 'plan.json').read_text(encoding='utf-8'))
DOC_MAP = {ROOT / a['old']: ROOT / a['new'] for a in PLAN['actions'] if a['kind'] == 'document'}


def write(relative, content):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + '\n', encoding='utf-8')


def rel(target, parent):
    return Path(os.path.relpath(target, parent)).as_posix()


def rewrite_links(text, before, after):
    def replace(match):
        value = match.group(1)
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', value) or value.startswith('#'):
            return match.group(0)
        path, sep, fragment = value.partition('#')
        absolute = Path(os.path.abspath(before.parent / unquote(path)))
        target = DOC_MAP.get(absolute, absolute)
        return '](' + rel(target, after.parent) + (sep + fragment if sep else '') + ')'
    return re.sub(r'\]\(([^)]+)\)', replace, text)


def stage(script=None, module=None, python='venv', args=None):
    out = {'python': f'.runtime/{python}/Scripts/python.exe'}
    if script:
        out['script'] = f'netheal-topology-generalization/scripts/{script}.py'
    else:
        out.update(module=f'src.{module}', cwd='netheal-topology-generalization')
    if args:
        out['args'] = args
    return out


def main():
    done = [json.loads(x) for x in (AUDIT / 'moves.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    assert len(done) == len(PLAN['actions']), 'Moves incomplete'
    marker = AUDIT / 'finalized.json'
    if marker.exists():
        raise FileExistsError('Already finalized; inspect instead of replaying')
    doc_hashes = {}
    for before, after in DOC_MAP.items():
        assert after.is_file() and not before.exists(), before
        original = after.read_text(encoding='utf-8-sig')
        revised = rewrite_links(original, before, after)
        after.write_text(revised, encoding='utf-8')
        before.write_text('# 문서 위치 변경\n\n본문은 [' + after.name + '](' + rel(after, before.parent) + ')로 이동했습니다.\n\n2026-10-12 디렉터리 재구성에 따른 호환 안내입니다.\n', encoding='utf-8')
        doc_hashes[after.relative_to(ROOT).as_posix()] = hashlib.sha256(after.read_bytes()).hexdigest()

    experiments = {
        'generalization/reliability_v2': ('GNN·MLP 일반화 v2. 기존 결과 완료.', {
            'run': stage(module='run_research_suite'), 'verify': stage(module='verify_research_run'), 'report': stage(module='report_research_suite')}),
        'generalization/factorial_v3': ('ID·순서와 토폴로지 학습 구조 3×2 실험. 기존 결과 완료.', {
            'run': stage(module='run_factorial_v3'), 'report': stage(module='report_factorial_v3')}),
        'api_pilot': ('Sol·Jev 예비 비교. 실행 시 유료 API를 사용할 수 있음.', {
            'run': stage(module='run_api_comparison'), 'report': stage(module='report_api_comparison')}),
        'laya_controls/learning': ('LAYA 학습 가능 여부 대조. 과거 실행 경로와 조건 보존.', {
            'run': stage(script='laya_learning_controls', python='laya-venv'), 'report': stage(script='report_laya_learning_controls', python='laya-venv')}),
        'laya_controls/symbols': ('LAYA 짧은 합성 신호 대조. 실제 네트워크 진단 점수가 아님.', {
            'run': stage(script='laya_symbol_controls', python='laya-venv'), 'report': stage(script='report_laya_symbol_controls', python='laya-venv')}),
        'laya_controls/optimization': ('LAYA 학습률·update 예산·선택지 순서 대조.', {
            'run': stage(script='laya_optimization_controls', python='laya-venv'), 'verify': stage(script='verify_laya_optimization_restore', python='laya-venv'), 'report': stage(script='report_laya_optimization_controls', python='laya-venv')}),
        'laya_controls/observations': ('LAYA 실제 공개 관측 표·문장 대조. 기존 실험 완료.', {
            'prepare': stage(script='laya_observation_data', python='laya-venv'), 'run': stage(script='laya_observation_controls', python='laya-venv'), 'report': stage(script='report_laya_observation_controls', python='laya-venv')}),
        'official_benchmark/jev': ('Jev v4 100개 완료: 성공 42 / 오진 58 / 문맥 오류 0. 재실행 시 유료 API 가능.', {
            'evaluate': stage(script='jev_official_eval_v4', python='jev-official-venv', args=['evaluate']), 'report': stage(script='report_jev_official_eval_v4', python='jev-official-venv')}),
        'official_benchmark/laya': ('데이터 40/20 생성 완료. 입력 준비 중단; capped probe·본 학습·공식 평가 미실행. 인계 문서 먼저 확인.', {
            'prepare': stage(script='laya_official_prepare', python='laya-official-venv'), 'capped_probe': stage(script='laya_official_capped_probe', python='laya-official-venv')}),
    }
    for name, (description, stages) in experiments.items():
        path = Path('experiments') / name
        write(path / 'config.json', json.dumps({'description': description, 'stages': stages}, ensure_ascii=False, indent=2))
        write(path / 'run.py', '''"""Preview or explicitly execute preserved experiment stages."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / '.gitmodules').is_file())
sys.path.insert(0, str(ROOT / 'src'))
from netheal_research.experiment_runner import main

if __name__ == '__main__':
    main(HERE.with_name('config.json'))
''')
        write(path / 'README.md', f'''# {name}

{description}

`config.json`은 실행 진입점 목록이다. 기존 실험의 학습 조건을 새로 정의하거나 덮어쓰지 않는다. 실제 조건은 기존 명세와 동결 산출물을 따른다.

프로젝트 루트에서 `.runtime/venv/Scripts/python.exe -B -X utf8 {path.as_posix()}/run.py --list`로 확인한다. `run.py 단계`는 명령을 출력만 하고, `run.py --execute 단계 -- 기존인자`일 때만 실행한다.

기존 스크립트는 고정 출력 경로를 사용할 수 있다. 새로운 실험은 기존 결과를 덮어쓰지 않는 출력 설정을 먼저 마련해야 한다. LAYA GPU 작업은 한 번에 하나만 실행한다.
''')

    for group in ('official_benchmark', 'laya_controls', 'api_pilot', 'generalization'):
        folder = ROOT / 'docs/research' / group
        links = ['- [' + p.name + '](reports/' + p.name + ')' for p in sorted((folder / 'reports').glob('*.md'))]
        write(folder.relative_to(ROOT) / 'README.md', '# ' + group + '\n\n' + '\n'.join(links) + '\n\n실행 전 명세는 [기존 specs](../../specs/)에 원문과 해시를 보존한다.\n')

    for relative in ('README.md', 'docs/README.md', 'netheal-topology-generalization/README.md', 'netheal-topology-generalization/scripts/README.md', 'docs/project_directory_20261012.md'):
        path = ROOT / relative
        path.write_text(rewrite_links(path.read_text(encoding='utf-8-sig'), path, path), encoding='utf-8')

    marker.write_text(json.dumps({'status': 'completed', 'documents': doc_hashes, 'experiment_entrypoints': list(experiments)}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'documents_moved': len(DOC_MAP), 'experiment_entrypoints': len(experiments)}))


if __name__ == '__main__':
    main()
