"""First-case live Jev smoke: readable public observations, original Claude policy."""
import ast
import asyncio
import json
import shutil
import time
from pathlib import Path
from compare_first_case_input import ROOT, readable
import jev_official_eval_v4 as env

OUT = ROOT / 'runs/official_benchmark/jev/2026-10-13_readable_baseline_smoke_r01'
SID = '00bd290a-df64-46a6-8f06-f087bde022a9'
CLAUDE_SOURCE = ROOT / 'runs/official_benchmark/claude/2026-10-13_purple_image_audit/app/netheal/aaa/gpt_agent.py'
tree = ast.parse(CLAUDE_SOURCE.read_text(encoding='utf-8'))
SYSTEM = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in n.targets))
ADAPTER = ('Follow the original agent task policy supplied in state. This interface returns a choice only; '
           'the original requirement to emit explanation text and native tool calls is replaced by the '
           'typed decision requested here. Select from all supplied options using the public observations. ')


class ReadableJev(env.Jev):
    async def choose(self, state, purpose, options):
        start = time.perf_counter()
        clean = readable(state)
        clean['original_agent_system_prompt'] = SYSTEM
        clean['initial_user_message'] = (
            f"**Hint/Context**: {state['hint']}\n\n" if state.get('hint') else '') + (
            f"**Available Tools**: {', '.join(t['name'] for t in clean['tool_schemas'])}\n\n"
            'Begin by exploring the environment and gathering information to complete the task.')
        criteria = {f'c{i:03d}': v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    for i, v in enumerate(options)}
        assert 1 <= len(criteria) <= 255
        body = {'model': 'jev-1.13.0', 'state': json.dumps(clean, ensure_ascii=False, separators=(',', ':')),
                'questions': {'decision': {'type': 'choice', 'instructions': ADAPTER + purpose, 'criteria': criteria}}}
        assert json.loads(body['state'])['history'] == state['history']
        if time.monotonic() >= self.deadline:
            raise TimeoutError('episode deadline')
        prep_seconds = time.perf_counter() - start
        self.calls += 1
        env.append(self.folder / 'request_started.jsonl', {'call': self.calls, 'unix': time.time()})
        start = time.perf_counter()
        response = await self.client.post('https://api.typesafe.ai/v1/systemone', json=body,
                                         timeout=min(120, max(1, self.deadline-time.monotonic())))
        record = {'request': body, 'status': response.status_code, 'seconds': time.perf_counter()-start,
                  'input_preparation_seconds': prep_seconds, 'history_equal': True}
        data = response.json()
        if response.status_code != 200:
            record['error_detail'] = data.get('detail', {})
            env.append(self.folder / 'requests.jsonl', record)
            raise RuntimeError('Jev HTTP ' + str(response.status_code))
        record['response'] = {k: data[k] for k in ('model', 'answers', 'usage', 'id') if k in data}
        env.append(self.folder / 'requests.jsonl', record)
        answer = data['answers']['decision']
        assert answer['choice'] in criteria
        selected = options[list(criteria).index(answer['choice'])]
        print(json.dumps({'call': self.calls, 'purpose': purpose, 'selected': selected,
                          'input_tokens': data.get('usage', {}).get('input_tokens')}, ensure_ascii=True), flush=True)
        return selected


async def main():
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(env.OUT / 'tool_schemas.json', OUT / 'tool_schemas.json')
    env.OUT = OUT
    env.Jev = ReadableJev
    env.save(OUT / 'protocol.json', {'case': SID, 'model': 'jev-1.13.0', 'compression': False,
        'partition': False, 'local_byte_guard': None, 'fixed_sleep': 0, 'concurrency': 1,
        'max_turns': 100, 'max_environment_steps': 100, 'deadline_seconds': 1200, 'retries': 0,
        'claude_source_sha256': env.sha(CLAUDE_SOURCE), 'script_sha256': env.sha(Path(__file__)),
        'original_system_prompt': SYSTEM, 'adapter_instruction': ADAPTER,
        'difference_from_original_claude': ['choice-based tool and argument selection',
            'no generated assistant reasoning text', 'JSON public history instead of native chat tool messages',
            'schema title annotations and outputSchema omitted', 'Windows direct handler versus original container'],
        'reference': 'Published Claude result reused; no Claude API invocation.'})
    snapshot = next(s for s in env.snapshots() if s['snapshot_id'] == SID)
    key = env.dotenv_values(ROOT / '.env').get('TYPESAFE_API_KEY')
    assert key
    status = await env.episode(snapshot, key, OUT / 'episode')
    result = json.loads((OUT / 'episode/result.json').read_text(encoding='utf-8'))
    requests = [json.loads(s) for s in (OUT / 'episode/requests.jsonl').read_text(encoding='utf-8').splitlines()]
    ref = json.loads((env.UP / 'manikyabard-20260201-100940.json').read_text(encoding='utf-8'))['results'][0]
    claude = next(r for a in ref['artifacts'] if a['label'] == 'aaa_metrics' for r in a['data']['results'] if r['snapshot_id'] == SID)
    assert result['metrics']['scenario_fingerprint'] == claude['scenario_fingerprint']
    tokens = [r.get('response', {}).get('usage', {}).get('input_tokens', 0) for r in requests]
    report = {'status': status, 'jev': result, 'claude_published': claude,
              'api_calls': len(requests), 'api_seconds': sum(r['seconds'] for r in requests),
              'input_tokens': sum(tokens), 'max_input_tokens': max(tokens, default=0),
              'estimated_usd': sum(tokens)*.042/1e6,
              'input_preparation_seconds': sum(r['input_preparation_seconds'] for r in requests)}
    env.save(OUT / 'comparison.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
