import asyncio,json,time
from dotenv import dotenv_values
import jev_official_eval_v4 as run
from jev_context_codec_v4 import decode,encode_checked

async def main():
    v3=run.OUT.parent/'netheal_official_jev_20261011_v3'
    folders=[p.parent for p in v3.glob('episodes/*/attempt_*/result.json') if json.loads(p.read_text())['status']=='error_ContextBudgetError']
    assert folders
    output=run.OUT/'oversized_arguments_check'; output.mkdir(exist_ok=False)
    client=run.Jev(dotenv_values(run.ROOT/'.env')['TYPESAFE_API_KEY'],output,time.monotonic()+1200)
    report=[]
    try:
        for folder in folders:
            requests=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
            state=decode(json.loads(requests[-1]['request']['state']))
            state['history']=[json.loads(s) for s in (folder/'tools.jsonl').read_text().splitlines()]
            state['solver_turn']=len(state['history'])
            helpers=[json.loads(s) for s in (folder/'helpers.jsonl').read_text().splitlines()]
            options=[a['parameters'] for a in helpers[-1]['result']['valid_actions'] if run.action_tool(a)=='ping']
            purpose='Select arguments for ping.'
            assert len(options)<=255
            chunks=run.partition_state(state,purpose,options)
            run.save(output/(folder.parent.name+'_state.json'),state)
            await client.choose(state,purpose,options)
            report.append({'snapshot_id':folder.parent.name,'partitions':len(chunks),'status':'accepted','mode':'tree' if 'evidence_fragments' in chunks[0] else 'history'})
        run.save(output/'check.json',report); print(json.dumps(report))
    finally: await client.client.aclose()

if __name__=='__main__': asyncio.run(main())
