# 프로젝트 디렉터리 안내

> 이 문서는 초기 정리 이력입니다. 이후 승인된 실제 재구성으로 결과·데이터·보고서를 이동했습니다. 현재 구조와 경로 호환성은 [architecture/README.md](architecture/README.md)를 우선하세요.

정리 기준일: 2026-10-12 KST. 첫 정리에서는 안내 문서를 정비했고, 이후 사용자 요청으로 임시 실험·디버그 산출물을 실제 보관 폴더로 옮기고 재생성 가능 캐시를 삭제했다. 정식 실행 경로·결과·가중치·가상환경은 유지했다.

## 어디서 시작할까

| 목적 | 파일 |
|---|---|
| 최신 Jev 결과 확인 | [최종 보고서](research/official_benchmark/reports/netheal_official_jev_context_fix_20261012.md) |
| LAYA 작업 재개 | [인계 문서](handoffs/2026-10-12_laya_handoff.md) |
| 과거 연구 및 설계 탐색 | [전체 문서 인덱스](README.md) |
| 원시 로그·가중치 위치 확인 | [결과 폴더 목록](../netheal-topology-generalization/results/README.md) |
| 실행 코드 찾기 | [scripts 안내](../netheal-topology-generalization/scripts/README.md) |

## 디렉터리 역할

| 경로 | 역할과 관리 기준 |
|---|---|
| `docs/results/` | 사람이 읽는 결과·해석·한계 보고서. 과거 기록은 당시 조건으로 해석 |
| `docs/specs/` | 실행 전 조건·재개 기록·인계 문서. 과거 명세를 최신 상태로 덮어쓰지 않음 |
| `docs/plans/`, `docs/archive/` | 연구 계획 및 이전 설계 보관 |
| `netheal-topology-generalization/src/` | 공통 정책·환경 래퍼·모듈 실행기 |
| `netheal-topology-generalization/scripts/` | 후속 LAYA·Jev 실험 실행·감사·보고 도구 |
| `netheal-topology-generalization/results/` | 원시 결과·요청 로그·소스 사본·해시·학습 산출물. 경로 보존 |
| `netheal-topology-generalization/reports/` | 이전 PoC 분석 보고서 |
| `netheal-topology-generalization/config/`, `tests/` | 설정 및 검증 코드 |
| `netheal-topology-generalization/models/` | 이전 모델 체크포인트. 후속 모델은 각 results 폴더에도 존재 |
| `netheal-ai-agent-benchmark/` | 기존 NetHeal 환경 서브모듈 |
| `OutlierDenStream-BigDama18/`, `telemetry/` | 별도 이상탐지·telemetry 연구 자료 |
| `.runtime/` | 로컬 가상환경 및 Hugging Face 모델 캐시. Git 제외 |
| `.env` | 비밀 설정. 열람·복사·Git 추가 대상 아님 |

공식 Jev 평가에 사용한 원본 환경 소스와 100개 스냅샷은 `results/netheal_official_jev_20261011_v1/upstream/`에 있다. 이름이 비슷한 루트 NetHeal 서브모듈로 임의 대체하면 안 된다.

## 현재 로컬 Python 환경

프로젝트 루트 기준 경로다. 새 환경으로 기존 환경을 덮어쓰지 않는다.

| Python | 사용 목적 |
|---|---|
| `.runtime/venv/Scripts/python.exe` | 기존 시뮬레이터·연구 실험 |
| `.runtime/jev-official-venv/Scripts/python.exe` | 공식 NetHeal 환경의 Jev 실행 및 데이터 생성 |
| `.runtime/laya-venv/Scripts/python.exe` | 기존 LAYA GPU 실험 |
| `.runtime/laya-official-venv/Scripts/python.exe` | 새 공식 환경 LAYA 작업, bitsandbytes 설치 환경 |

새 LAYA 환경은 기존 환경의 패키지 경로를 참조한다. 이전 가상환경은 중복으로 간주해 지우지 않는다. GPU 작업은 하나씩 실행한다. Windows에서 한글 출력은 `-X utf8`을 사용한다.

## 보존과 용량

- [결과 목록](../netheal-topology-generalization/results/README.md)의 용량은 파일 크기 합계이며 디스크 실제 점유·캐시·가상환경 전체 용량은 아니다.
- 짧은/긴 LAYA 자원 시험 폴더는 각각 약 2.54GB다. 저장·복원 검증에 쓰인 model/optimizer 상태이므로 임의 삭제하지 않았다.
- `_tmp_*`, smoke, 실패·중단 이름의 폴더도 감사 기록일 수 있다. 이름만으로 삭제하지 않는다.
- `__pycache__`, `.pytest_cache`, 가상환경, `.env`는 기존 `.gitignore`에서 제외되어 있다. 이번 정리는 Git 추적 정책을 바꾸지 않았다.
- 작업 트리에는 이전부터 많은 변경이 있다. 특히 `src/run_nonlink_single_fault_audit.py`는 이번 정리 대상에서 제외했다. 자동 commit·reset·checkout은 하지 않았다.

## 이번 정리 내용

루트 README와 문서 인덱스를 최신 상태로 보완하고, 구성 안내·스크립트 안내·전체 결과 목록 및 파일 용량 목록을 추가했다.

이후 임시 smoke 폴더 2개와 디버그 로그 2개, 총 64개 파일을 `results/archive/cleanup_20261012/` 아래 smoke/debug로 재배치했다. 이동 전후 모든 파일의 SHA-256을 검증했다. 재생성 가능한 캐시 5곳을 삭제해 836,111 bytes를 정리했다. [실제 이동·삭제 기록](../netheal-topology-generalization/results/archive/README.md)에 복원 안내도 남겼다. 정식 결과와 실행 코드는 이동하지 않았으며 학습이나 유료 API 호출은 하지 않았다.
