"""Resume frozen context-safe run; preserve interrupted traces, never retry completed cases."""
import ast
import asyncio
import contextlib
import json
import time
import context_safe_baseline as run


async def main():
    protocol = run.read(run.OUT/'protocol.json')
    for name, expected in protocol['source_hashes'].items():
        assert run.env.sha(run.ROOT/name) == expected, name
    run.env.OUT = run.OUT
    run.env.Jev = run.SafeJev
    ref = run.read(run.env.UP/'manikyabard-20260201-100940.json')['results'][0]
    reference = {r['snapshot_id']:r for a in ref['artifacts'] if a['label']=='aaa_metrics' for r in a['data']['results']}
    key = run.env.dotenv_values(run.ROOT/'.env').get('TYPESAFE_API_KEY')
    assert key
    archive = run.OUT / ('interrupted_' + str(int(time.time())))
    restored = []
    for s in run.env.snapshots():
        folder = run.OUT/'episodes'/s['snapshot_id']
        if (folder/'result.json').exists():
            continue
        if folder.exists():
            assert folder.resolve().is_relative_to(run.OUT.resolve())
            archive.mkdir(exist_ok=True)
            destination = archive/folder.name
            assert destination.resolve().is_relative_to(run.OUT.resolve())
            folder.rename(destination)
            restored.append(folder.name)
            run.env.save(archive/'resume.json', {'reason':'Session ended; user requested continuation',
                'restarted_incomplete_cases':restored,'completed_cases_reused':True,'frozen_sources_verified':True})
        with (run.OUT/'execution.log').open('a',encoding='utf-8') as log, contextlib.redirect_stdout(log):
            await run.env.episode(s,key,folder)
        report = run.summarize(reference)
        print(json.dumps({k:report[k] for k in ('evaluated','success','statuses')}),flush=True)
    report = run.summarize(reference)
    # Reuse the frozen report finalization statements verbatim without rerunning setup/evaluation.
    source = ast.parse(run.ROOT.joinpath(protocol['source_hashes'] and 'experiments/official_benchmark/jev/context_safe_baseline.py').read_text(encoding='utf-8'))
    main_node = next(n for n in source.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='main')
    index = next(i for i,n in enumerate(main_node.body) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='req' for t in n.targets))
    tail = ast.Module(body=main_node.body[index:],type_ignores=[])
    scope = dict(vars(run),report=report)
    exec(compile(ast.fix_missing_locations(tail),'<frozen_report_finalization>','exec'),scope)
    with run.REPORT.open('a',encoding='utf-8') as f:
        f.write('\n세션 종료 후 완료된 22개를 재사용하고 미완료 1개 로그를 별도 보존한 뒤 재개했다. 중단된 시도와 사전 검사의 비용은 위 최종 100개 사용량에 포함하지 않았다.\n')


if __name__=='__main__':
    asyncio.run(main())
