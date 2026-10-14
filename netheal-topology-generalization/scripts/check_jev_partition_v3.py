"""Development-only acceptance check; no diagnosis accuracy used for configuration."""
import asyncio, json, time
from dotenv import dotenv_values
import jev_official_eval_v3 as run
from jev_context_codec import encode_checked

async def main():
    state=json.loads((run.OUT/'development_100step_state.json').read_text(encoding='utf-8'))
    purpose='Select the next tool.'
    chunks=run.partition_state(state,purpose,run.TOOLS)
    assert json.dumps([e for c in chunks for e in c['history']],sort_keys=True)==json.dumps(state['history'],sort_keys=True)
    sizes=[len(encode_checked(c).encode())+len(json.dumps(run.question(purpose,run.TOOLS),ensure_ascii=False).encode())+512 for c in chunks]
    assert max(sizes)<=run.MAX_REQUEST_TEXT_BYTES
    folder=run.OUT/'stress_partition_check'
    folder.mkdir(exist_ok=False)
    j=run.Jev(dotenv_values(run.ROOT/'.env')['TYPESAFE_API_KEY'],folder,time.monotonic()+1200)
    try:
        await j.choose(state,purpose,run.TOOLS)
        report={'status':'accepted','events':100,'partitions':len(chunks),'guard_bytes':sizes,'requests':j.calls,'roundtrip':'exact'}
        run.save(folder/'check.json',report)
        print(json.dumps(report))
    finally: await j.client.aclose()

if __name__=='__main__': asyncio.run(main())
