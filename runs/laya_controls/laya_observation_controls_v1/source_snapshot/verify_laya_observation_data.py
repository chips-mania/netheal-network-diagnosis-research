"""CPU-only integrity, history overlap and rendering tests; no model scores."""
import json, re
from collections import Counter
from pathlib import Path
from laya_observation_data import OUT, PROJECT, dump, sha, digest, core, parse, render, old_facts

def main():
    cases={s:[json.loads(l) for l in (OUT/f'{s}_cases.jsonl').read_text(encoding='utf8').splitlines()] for s in ['train','test']}
    all_cases=cases['train']+cases['test'];ids={r['episode_seed'] for r in all_cases}
    assert len(ids)==96 and len({r['core_sha256'] for r in all_cases})==96
    checks={}
    for s,rr in cases.items():
        counts=Counter((r['topology'],r['ground_truth']['type']) for r in rr)
        assert len(counts)==8 and set(counts.values())==({4} if s=='train' else {8})
        for arm in ['table','prose']:
            rendered=[json.loads(l) for l in (OUT/f'{arm}_{s}.jsonl').read_text(encoding='utf8').splitlines()]
            for r,row in zip(rr,rendered):
                assert r['episode_seed']==row['episode_seed']
                assert parse(row['state'],arm)==r['facts']
                # Labels and expert metadata cannot affect the rendering.
                mutant=dict(r,expected={},ground_truth={},expert_conclusion=('wrong','wrong'),episode_seed=-1)
                assert render(mutant['facts'],arm)==row['state']
                assert str(r['episode_seed']) not in row['state']
                assert len(r['trace'])==1+2*r['n']+len([f for f in r['facts'] if f[0]=='Link'])
                assert all(x['action'].get('category')!='diagnostic_submission' for x in r['trace'])
                for kind,name,keys,values in r['facts']:
                    f=dict(zip(keys,values))
                    if kind=='Device':assert f['status_checked']==f['interfaces_checked']=='1'
                    elif kind=='Link':assert f['ping_done']=='1' and f['reserved_zero']=='0'
                    else:assert f['scanned']=='1'
        checks[s]={'episodes':len(rr),'cells':len(counts),'expert_successes':sum(r['expert_success'] for r in rr),
                   'all_status_interfaces_and_nominal_pings_observed':True,'reserved_hidden_link_status_zero':True}
    # Scan every existing result JSON/JSONL for the reserved new episode IDs.
    # Full state strings are checked where archived. Tensor-only snapshots are
    # covered by their accompanying seed/audit files, not silently deserialized.
    files=[];overlap=[];observation_overlap=[];states=0;core_hashes={r['core_sha256'] for r in all_cases}
    def visit(obj,path):
        nonlocal states
        if isinstance(obj,dict):
            if isinstance(obj.get('state'),str):
                fs=old_facts(obj['state'])
                if fs:
                    states+=1
                    if digest(core(fs)) in core_hashes:observation_overlap.append(str(path))
            for v in obj.values():
                if isinstance(v,(dict,list)):visit(v,path)
        elif isinstance(obj,list):
            for v in obj:
                if isinstance(v,(dict,list)):visit(v,path)
    for path in sorted((PROJECT/'results').rglob('*')):
        if OUT in path.parents or path.suffix not in ['.json','.jsonl']:continue
        text=path.read_text(encoding='utf-8-sig')
        found={int(x) for x in re.findall(r'(?<!\d)9[12]\d{6}(?!\d)',text)} & ids
        if found:overlap.append({'path':str(path),'ids':sorted(found)})
        files.append({'path':str(path.relative_to(PROJECT)),'sha256':sha(path)})
        try:
            if path.suffix=='.jsonl':
                for line in text.splitlines():
                    if line.strip():visit(json.loads(line),path)
            else:visit(json.loads(text),path)
        except json.JSONDecodeError as e:
            raise RuntimeError(f'Unparsed historical artifact: {path}') from e
    assert not overlap and not observation_overlap
    dump(OUT/'extended_data_audit.json',dict(status='completed_verified',checks=checks,
        historical_files=files,historical_compact_state_records=states,episode_overlap=overlap,
        observation_overlap=observation_overlap,
        limitation='Exact equality at 5 significant digits; near duplicates and task templates may recur. Historical artifacts without state text are seed-checked only.'))
    prior=PROJECT/'results/laya_optimization_controls_v1'
    summary=json.loads((prior/'summary.json').read_text(encoding='utf8'))
    restored=json.loads((prior/'restoration_audit.json').read_text(encoding='utf8'))
    assert len(summary['runs'])==len(restored['runs'])==12
    previous=[]
    for run in summary['runs']:
        folder=prior/f'{run["arm"]}_{run["seed"]}'
        assert sha(folder/'trainable.pt')==run['checkpoint_sha256']
        final=run['final']['validation']
        correct=sum(max(range(len(z)),key=z.__getitem__)==max(range(len(y)),key=y.__getitem__) for z,y in zip(final['logits'],final['targets']))
        assert abs(correct/len(final['logits'])-final['accuracy'])<.0001
        previous.append({'arm':run['arm'],'seed':run['seed'],'accuracy_recomputed':correct/len(final['logits'])})
    dump(OUT/'previous_results_audit.json',{'status':'verified_read_only','runs':previous,
        'prior_restoration_status':restored['status'],
        'note':'Validated prior stored logits, saved-weight hashes and recorded restoration audit; did not rerun prior GPU experiments.'})
    print(json.dumps({'status':'completed_verified','historical_files':len(files),'states_checked':states,'splits':checks}))

if __name__=='__main__':main()
