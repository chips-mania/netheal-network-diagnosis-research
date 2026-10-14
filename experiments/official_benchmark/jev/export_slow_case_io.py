"""Offline, complete I/O export for the slowest context-safe evaluation case."""
import collections
import json
import shutil
from pathlib import Path
from compare_first_case_input import ROOT

SID = '6b3c82e6-1657-4dfe-aa44-70e28c2587be'
SOURCE = ROOT / 'runs/official_benchmark/jev/2026-10-13_context_safe_baseline_r01/episodes' / SID
OUT = ROOT / 'docs/research/official_benchmark/case_studies/2026-10-13_slowest_case'


def lines(name):
    return [json.loads(x) for x in (SOURCE/name).read_text(encoding='utf-8').splitlines()]


def block(x):
    return '```json\n'+json.dumps(x,ensure_ascii=False,indent=2)+'\n```\n\n'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=OUT/'raw'; raw.mkdir(exist_ok=True)
    for path in SOURCE.iterdir():
        if path.is_file(): shutil.copy2(path,raw/path.name)
    requests=lines('requests.jsonl'); decisions=lines('decisions.jsonl'); tools=lines('tools.jsonl')
    result=json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    assert len(requests)==sum(d['parts'] for d in decisions)==248
    assert len(tools)==55 and len(decisions)==109
    first=json.loads(requests[0]['request']['state'])
    total=result['seconds']; api=sum(r['seconds'] for r in requests)
    summary=['# 177초 사례의 실제 입출력\n\n',f'사례: `{SID}`. 모든 자료는 저장된 실제 실행 로그에서 추출했으며 모델을 새로 호출하지 않았다.\n\n',
        f'- 전체 시간: **{total:.6f}초**\n- API 응답 시간 합계: **{api:.6f}초**\n- 차이: **{total-api:.6f}초** (로컬 처리 등, 세부 시간 미계측)\n- 환경 도구: **55회**\n- 판단: **109회**\n- 실제 API 요청: **248회**\n- 고정 요청 대기: **없음**\n\n',
        '정답은 `performance_degradation / device_4->device_1`, 최종 제출은 `performance_degradation / device_0->device_1`으로 위치 오답이다. 아래 정답은 사후 평가 메타데이터이며 모델 입력에 삽입한 정보가 아니다.\n\n',
        '## 파일 안내\n\n- [모든 API 입력·응답 전문](api_io_full.md): 248회. 전송 request body와 반환 response를 생략 없이 수록.\n- [입력 state를 펼쳐 읽는 파일](input_states_full.md): JSON 문자열을 파싱해 표시. 압축 문법은 실제 전송 그대로 유지하며 원문 관측으로 복원한 파일은 아님.\n- [모든 도구 입력·결과 전문](tool_io_full.md): 55회.\n- [원본 요청 로그](raw/requests.jsonl), [판단 로그](raw/decisions.jsonl), [후보 조회 로그](raw/helpers.jsonl), [최종 채점](raw/result.json).\n\n',
        '## 첫 입력의 과제와 지시\n\n',block({k:v for k,v in first.items() if k!='history'}),
        '## 시간과 반복 행동\n\nAPI seconds는 HTTP 요청 직전부터 응답 수신까지다. 전체 시간에서 이를 뺀 값이 전부 압축 시간이라는 뜻은 아니다. 모델 서버의 순수 연산 시간도 아니다. 원문/압축/분할 전환 및 API 호출 번호는 아래에서 확인할 수 있다. 분할 판단은 각 조각 확률의 원자료 값 바이트 가중 평균으로 선택했으며, 단일 조각은 제공사 choice를 사용했다.\n\n',
        '| 도구 | 실행 횟수 |\n|---|---:|\n']
    for name,count in collections.Counter(t['tool'] for t in tools).items():summary.append(f'| `{name}` | {count} |\n')
    summary.append('\n## 판단 순서와 실제 API 호출 대응\n\n| 판단 | solver turn (0부터) | 질문 | 모드 | API 호출 번호 | 합계 HTTP 초 | 최종 선택 |\n|---:|---:|---|---|---|---:|---|\n')
    cursor=0
    with (OUT/'api_io_full.md').open('w',encoding='utf-8') as f, (OUT/'input_states_full.md').open('w',encoding='utf-8') as states:
        f.write('# 실제 API 입력·출력 전문\n\n[요약으로](README.md). request/response를 생략 없이 JSON으로 재출력했다. JSON 공백 표기는 달라질 수 있으나 값과 state 문자열은 보존했다. 인증 헤더는 로그에 포함되지 않는다.\n\n')
        states.write('# 실제 전송 state 펼쳐 보기\n\nJSON 문자열 바깥의 이스케이프만 해제했다. `$ref` 등의 압축 문법은 실제 입력이며 이 파일에서는 복원하지 않았다. 전체 질문·선택지는 [API 전문](api_io_full.md)을 참조한다.\n\n')
        for di,d in enumerate(decisions,1):
            group=requests[cursor:cursor+d['parts']]
            selected=json.dumps(d['selected'],ensure_ascii=False).replace('|','\\|')
            summary.append(f"| {di} | {d['solver_turn']} | {d['purpose']} | {d['mode']} | {cursor+1}–{cursor+len(group)} | {sum(r['seconds'] for r in group):.4f} | `{selected}` |\n")
            for j,r in enumerate(group,1):
                n=cursor+j
                f.write(f"## API {n} / 판단 {di} / 조각 {j}·{len(group)}\n\n모드 `{r['mode']}`, 가중치 `{r['weight']}`, HTTP `{r['status']}`, 시간 `{r['seconds']:.6f}초`.\n\n### 입력\n\n")
                f.write(block(r['request']))
                f.write('### 출력\n\n'+block(r.get('response',r.get('error_detail'))))
                states.write(f'## API {n} / 판단 {di} / 조각 {j}·{len(group)}\n\n'+block(json.loads(r['request']['state'])))
            cursor+=len(group)
    with (OUT/'tool_io_full.md').open('w',encoding='utf-8') as f:
        f.write('# 실제 도구 입력·출력 전문\n\n[요약으로](README.md). 자동 후보 조회 list_actions는 별도 [helpers.jsonl](raw/helpers.jsonl)에 보존했다.\n\n')
        for i,t in enumerate(tools,1):
            f.write(f"## 도구 실행 {i}: {t['tool']}\n\n### 입력\n\n"+block({'tool':t['tool'],'arguments':t['arguments']}))
            f.write('### 출력\n\n'+block(t['result']))
    summary.append('\n## 최종 채점 원문\n\n'+block(result))
    (OUT/'README.md').write_text(''.join(summary),encoding='utf-8')
    print(json.dumps({'folder':str(OUT),'api_requests':len(requests),'decisions':len(decisions),'tools':len(tools),
                     'files':{p.name:p.stat().st_size for p in OUT.glob('*.md')}},ensure_ascii=False))


if __name__=='__main__':main()
