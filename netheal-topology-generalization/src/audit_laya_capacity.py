"""Audit all initial pilot graphs before expensive inference."""
from pathlib import Path
from transformers import AutoTokenizer
from laya.common import build_sequence
from laya.train import to_internal
from .run_factorial_v3 import CONDITIONS
from .run_research_suite import make_env,FAULTS,dump
from .public_observation import PublicView
from .graph_obs import build_graph_obs
from .laya_schema_compact import state_text,questions

def main():
    tok=AutoTokenizer.from_pretrained(str(next(Path('../.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots').iterdir())/'tokenizer'))
    rows=[]
    for ci,c in enumerate(CONDITIONS):
        for i in range(8):
            e=make_env(c[1],c[2],FAULTS[i%4]);e.reset(seed=9100000+ci*10000+i)
            g=build_graph_obs(PublicView(e));s=state_text(g)
            for name,q in questions(g).items():
                ids,m,st,tr=build_sequence(tok,s,to_internal(name,q),4096,1024,return_stats=True,return_truncation_stats=True)
                rows.append(dict(case=f'{c[0]}:{i}',question=name,tokens=len(ids),truncated=tr['truncated'],
                    dropped=tr['state_tokens_dropped'],options=len(q['criteria']),markers=len(m)))
            e.close()
    dump('results/laya_data_compact_v2/capacity_audit.json',rows)
    print(dict(questions=len(rows),truncated=sum(r['truncated'] for r in rows),max_tokens=max(r['tokens'] for r in rows)))
if __name__=='__main__':main()
