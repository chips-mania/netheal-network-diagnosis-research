"""API acceptance/usage only. Predictions are not used to tune the adapter."""
import asyncio,json,os,time
from dotenv import dotenv_values
import jev_official_eval_v3 as run

async def main():
    key=os.environ.get('TYPESAFE_API_KEY') or dotenv_values(run.ROOT/'.env')['TYPESAFE_API_KEY']
    folder=run.OUT/'context_api_check'
    folder.mkdir(exist_ok=False)
    client=run.Jev(key,folder,time.monotonic()+1200)
    old=run.OUT.parent/'netheal_official_jev_20261011_v2'
    states=[]
    for f in old.glob('episodes/*/attempt_00/requests.jsonl'):
        for line in f.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['status']==400: states.append((f.parents[1].name,r['request']))
    assert len(states)==39
    states.append(('development_100steps',{'state':json.loads((run.OUT/'development_100step_state.json').read_text()),
                        'questions':{'decision':{'criteria':{'a':'scan_network','b':'submit_diagnosis'}}}}))
    report=[]
    for sid,body in states:
        options=list(body['questions']['decision']['criteria'].values())
        try:
            await client.choose(body['state'],'Choose from the supplied options.',options)
            status='accepted'
        except Exception as e:
            status=str(e) if isinstance(e,RuntimeError) else type(e).__name__
        report.append({'snapshot_id':sid,'status':status})
        print(sid,status,flush=True)
    await client.client.aclose()
    run.save(run.OUT/'context_api_check.json',report)

if __name__=='__main__': asyncio.run(main())
