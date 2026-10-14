import json
import jev_official_eval_v4 as run
from jev_context_codec_v4 import encode_checked,decode
from jev_tree_partition_v4 import restore

checked=0;maximum=0;tree_count=0
old=run.OUT.parent/'netheal_official_jev_20261011_v2'
for f in old.glob('episodes/*/attempt_00/requests.jsonl'):
    for line in f.read_text(encoding='utf-8').splitlines():
        r=json.loads(line); state=r['request']['state']; q=r['request']['questions']['decision']
        purpose=q['instructions'][len(run.INSTRUCTIONS)+1:]; options=list(q['criteria'].values())
        chunks=run.partition_state(state,purpose,options)
        for c in chunks:
            encoded=encode_checked(c)
            assert len(encoded.encode())+len(json.dumps(run.question(purpose,options),ensure_ascii=False).encode())+512<=24000
        if 'evidence_fragments' in chunks[0]:
            recovered=restore([x for c in chunks for x in c['evidence_fragments']]);tree_count+=1
            assert json.dumps(recovered,sort_keys=True)==json.dumps(state,sort_keys=True)
        elif len(chunks)>1:
            assert json.dumps([e for c in chunks for e in c['history']],sort_keys=True)==json.dumps(state['history'],sort_keys=True)
        checked+=1;maximum=max(maximum,len(chunks))
assert checked==1571
report={'status':'passed','requests':checked,'max_partitions':maximum,'tree_partition_decisions':tree_count}
run.save(run.OUT/'partition_audit.json',report);print(json.dumps(report))
