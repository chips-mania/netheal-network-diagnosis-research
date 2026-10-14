# NetHeal 원본 Claude 조건의 Jev 평가 착수 기록

작성일: **2026-10-11 (KST)**

후속 갱신: 사용자가 2026-10-11 유료 Jev API 호출을 명시적으로 허용했다. 이후 smoke는 정상 응답 5회로 완료했고, v1 입력 길이 결함을 보존·수정한 뒤 별도 v2 평가를 시작했다. 아래 'API 0/미실행' 설명은 최초 사전 검증 시점의 기록이다. 최종 상태는 후속 결과 보고서를 따른다.

## 상태

**원본 확보와 API 없는 사전 검증 완료. Jev API smoke 및 본 평가는 미실행.**

유료 Jev smoke 호출을 위한 실행 요청은 자동 승인 검토에서 거절됐다. 검토 이유는 이전 사용자의 '유료 GPT/Jev API 제외' 조건이 이후 Jev 테스트 요청만으로 명시적으로 철회됐다고 보기 어렵다는 것이었다. 거절 후 우회 호출하지 않았다. 이 작업에서 Jev 요청과 API 비용은 0이다. 유료 호출의 명시적 범위 확인이 필요하다.

## 확보·검증한 원본

- 대상 Claude Run ID: `685bad60-d554-4300-9c7e-e849301d6df7`.
- 결과·TOML·provenance: leaderboard commit `496a07b13bdc1174325afefd8a7d63416e246c41`의 `manikyabard-20260201-100940` 파일들.
- 평가 스냅샷: 위 commit의 `snapshots/v1/` 100개.
- 원본 green 이미지: `manikya/netheal-green-agent@sha256:ae1e090e36cd5acb419be39efca2b247f711bf12ea523ca5e898ea519fae7d2f`.
- Linux amd64 manifest: `sha256:8218f48801960c98e1443bcd49dbfe3b3edda6c19c28bdfc3179d3638d58ce40`.
- 환경 소스 레이어: `sha256:6e2436e6bfb7337f3566ecda5d5674af6d691f1101315b6a5f9b85cb292779ea`.
- Docker 설치가 발견되지 않아 공개 registry에서 해당 소스 레이어를 내려받아 새 결과 폴더에 추출했다. 기존 submodule 소스를 수정하지 않았다.
- **추출한 환경 코드로 100개 스냅샷을 복원했고 scenario fingerprint가 원본 결과와 100/100 일치했다.** 모두 환경 예산 100단계다.

공개 원본: [대시보드](https://agentbeats.dev/manikyabard/netheal-ai-agent-benchmark), [결과 커밋](https://github.com/manikyabard/netheal-agentbeats-leaderboard/commit/496a07b13bdc1174325afefd8a7d63416e246c41).

## 이전 비교 설명 정정

원본 대시보드의 diagnosis_success_rate는 **장비 수로 가중한 평균**이다. 원본 결과 100개를 다시 집계하면:

- 단순 성공률: **66/100 = 66.0%**.
- 대시보드 방식: **65.0375939849624%**, 화면에는 65.0%.

따라서 우리 과거 Sol 87.5%와 화면의 65.0%를 단순 차감한 값은 평가 조건뿐 아니라 집계 방식도 다르다. 새 Jev 결과에서는 원본 가중 점수와 단순 사례 성공률을 모두 보고해야 한다.

## 실행 조건과 남은 차이

| 항목 | 원본 | 현재 준비 상태 |
|---|---|---|
| 사례 | 고정 스냅샷 100개 | 동일 파일·fingerprint 검증 |
| 환경·도구·채점 소스 | 고정 green 이미지 | 해당 이미지에서 추출한 원본 사용 |
| 행동 예산 | 100 | 스냅샷에서 100 확인 |
| 힌트 | 제공, heuristic 설정 | 스냅샷의 기존 공개 힌트 사용 |
| 제한 시간 | 에피소드 1,200초 | 연결 코드에 반영 |
| 동시성 | 5 | 본 평가 코드에 반영 |
| 에피소드 재시도 | 최대 10 | **아직 미구현, 본 평가 전 확정 필요** |
| solver | Claude 생성형 도구 호출 | Jev 도구 선택 → 인자 선택으로 변환 |
| 설명 문장 | 원본 solver에 추론 설명 지시 | Jev에는 생성형 설명 요구하지 않음 |
| 전송 | MCP/A2A, Docker Linux Python 3.11.14 | 현재 원본 FastMCP handler 직접 호출, Windows Python 3.13.2 |
| 의존성 | 컨테이너 설치 버전 | 새 로컬 환경, 동일 버전 보장하지 않음 |

이 단계에서 컨테이너·전송까지 완전히 동일한 실행이라고 주장하지 않는다. 직접 handler 호출은 원본 도구 실행/채점 코드를 유지하지만 통신 시간은 비교 불가다. 본 평가 전 차이표와 프로토콜을 고정해야 한다.

Jev 연결 초안은 공개 task/hint/tool schema/도구 결과만 입력하고 전체 관측 히스토리를 유지한다. 인자 후보 확보를 위해 무료 `list_actions`를 자동 호출하는 차이가 있다. 비공개 환경 정보나 정답을 후보 생성에 쓰지 않는다. 한 번에 도구 하나를 고르고, 진단 위치는 Jev가 선택한 유형에 따라 질문한다. 기존 compact graph pilot과 별도 구성이다.

## API 없는 검증

- 공식 평가와 다른 seed `93001101`의 4노드 Star 개발 사례 생성.
- list_actions/scan_network, discover_neighbors/check_status/check_interfaces/ping/traceroute 정상 실행 확인.
- 최종 submit_diagnosis와 성공 채점까지 원본 handler로 검증. 이 연결 검사는 개발 사례의 정답을 사용하는 결정적 테스트이며 모델 성능 평가가 아니다. Jev에 정답을 전달한 적은 없다.
- 모델에 전달할 도구 결과에서 ground_truth_fault 제외 확인.
- 원본 결과 가중 성공률 재집계 및 100개 fingerprint 검사 통과.
- 별도 `.runtime/jev-official-venv` 생성. 기존 환경은 변경하지 않았으며 일부 패키지는 기존 환경 경로를 읽기 전용 참조한다. 새 환경에 MCP 1.30.0/A2A 0.3.26 등의 추가 의존성을 설치했다.
- 이후 유료 smoke는 **개발 사례 1개, 최대 3개 solver turn / 최대 9개 Jev 요청**으로 제한했다. 본 평가 사례 성적을 보고 연결 설정을 선택하지 않는다.

## 산출물·재현

- `netheal-topology-generalization/scripts/jev_official_eval.py`: 연결 초안, preflight/smoke/evaluate 명령.
- `netheal-topology-generalization/results/netheal_official_jev_20261011_v1/upstream/`: 원본 결과·설정·provenance·소스·스냅샷.
- 같은 결과 폴더의 `preflight.json`, `tool_schemas.json`, `development_snapshot.json`.

저장소 루트에서 API 없는 검증:

```powershell
.runtime/jev-official-venv/Scripts/python.exe netheal-topology-generalization/scripts/jev_official_eval.py preflight
```

유료 호출 범위를 확인한 뒤 개발 smoke를 실행한다. 본 평가는 `frozen_protocol.json`과 소스 해시가 있어야 시작하도록 막아 두었다. smoke 미실행 상태에서 본 평가가 준비 완료됐다고 쓰지 않는다.
