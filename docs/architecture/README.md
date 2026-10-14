# 디렉터리 구조와 호환성

2026-10-12 승인된 재구성을 적용했다. 연구 산출물은 주제별 위치로 실제 이동했으며, 기존 실행을 보존하는 전환 구조다.

| 경로 | 역할 |
|---|---|
| `src/netheal_research/` | 새 공통 Python 패키지. 경로 처리·실행 진입점과 이전한 환경/평가 모듈 |
| `experiments/` | 연구 주제별 실행 진입점 9개. 각 폴더의 config.json은 기존 실행기와 환경을 연결 |
| `runs/` | 일반화·API pilot·LAYA 대조·공식 평가·기존 PoC·보관 기록 |
| `data/generated/laya/` | 기존 LAYA 학습 데이터의 실제 위치 |
| `data/manifests/` | 데이터 위치 및 분할·출처 기록을 찾는 안내 |
| `configs/` | 기존 동결 설정 위치와 새 설정 작성 규칙 |
| `tests/` | 새 구조의 검증. 기존 tests도 루트 pytest 설정에 포함 |
| `docs/research/` | 연구 주제별 결과 보고서 |
| `docs/handoffs/` | 새 세션 인계 문서 |
| `docs/specs/` | 기존 실행 전 명세. 동결 해시를 위해 원문 보존 |
| `scripts/maintenance/` | 이동 계획·실행·검증 및 호환 경로 복구 도구 |
| `netheal-topology-generalization/` | 동결 실행 코드와 이전 경로 호환 영역 |

## 물리적 이동과 호환 경로

결과 디렉터리는 `runs/` 또는 `data/`로 옮기고 이전 위치에 NTFS junction을 남겼다. 개별 기존 PoC 산출물과 의존성이 독립적인 공통 Python 모듈 3개는 새 위치로 옮긴 뒤 이전 위치에 hard link를 남겼다. 두 경로는 같은 파일을 가리키므로 대용량 사본을 추가하지 않는다.

**연결 경로의 파일을 수정하면 실제 파일도 바뀐다.** 두 경로를 별도 백업이나 별도 실행 결과로 취급하지 않는다. 파일을 통째로 교체하는 편집기는 hard link를 분리할 수 있으므로 수정 후 호환성 검증이 필요하다. 용량 집계·백업은 `runs/`, `data/`를 기준으로 수행하고 기존 연결 폴더를 중복 순회하지 않는다.

보고서 14개와 인계 문서 1개는 실제 이동하고 상대 링크를 갱신했다. 이전 문서 위치에는 새 위치로 연결하는 짧은 안내를 남겼다. 기존 보고서 생성기는 옛 출력 경로에 쓰므로 재생성 시 안내 파일을 덮어쓸 수 있다. 향후 보고서 생성기를 수정하는 실험에서는 새 출력 경로를 명시해야 한다. 이번 정리에서는 생성기를 실행하지 않았다.

## 보존한 영역

- 원본 공식 NetHeal 소스·스냅샷인 `netheal_official_jev_20261011_v1/upstream/`은 기존 위치 유지.
- 진행 중인 `laya_official_20261012_v1/` 데이터·코드는 기존 위치 유지. LAYA 본 학습은 미실행.
- Jev·LAYA scripts와 실행 전 명세, `run_nonlink_single_fault_audit.py`를 포함한 보호 파일 66개는 해시로 변경 없음을 확인한다.
- 상대 import와 경로 계산이 연결된 나머지 정책 코드는 아직 기존 src에 있다. 새 패키지로 전부 이식했다고 주장하지 않는다.
- 가상환경·모델 캐시·외부 submodule 위치 유지. 기존 가상환경에 패키지 설치를 실행하지 않았다.

## 실행 방법

프로젝트 루트에서 목록 확인:

```powershell
.runtime/venv/Scripts/python.exe -B -X utf8 experiments/official_benchmark/jev/run.py --list
.runtime/venv/Scripts/python.exe -B -X utf8 experiments/official_benchmark/laya/run.py prepare
```

두 명령은 실험을 실행하지 않는다. 실제 실행에는 `--execute`가 필요하다. 기존 학습 조건과 출력 경로 확인 후 사용한다.

루트에서 회귀 검증:

```powershell
.runtime/venv/Scripts/python.exe -B -m pytest -p no:cacheprovider
```

모델별 테스트 의존성이 환경에 없으면 해당 환경을 사용한다. 기존 Python·NetHeal 버전을 자동 설치하거나 바꾸지 않는다.

## 이름과 새 실험 규칙

- Python은 소문자 snake_case. 공통 모듈명에는 날짜를 넣지 않는다.
- 새 실행은 `runs/<연구>/<모델>/YYYY-MM-DD_<조건>_rNN/`에 생성한다.
- 실행 폴더에는 `manifest.json`, `config.json`, `provenance/`, `logs/`, `checkpoints/`, `metrics/`를 필요한 만큼 둔다.
- 실행 상태는 manifest에 기록하고, 상태 변경 때문에 경로를 바꾸지 않는다.
- 기존 실행 이름은 당시 명세와 참조를 보존하기 위해 소급 변경하지 않았다.

## 재현·복구 기록

- [전체 이전·현재 경로 및 보호 해시](migration_20261012/plan.json)
- [파일별 이동 전 해시](migration_20261012/files.jsonl)
- [완료된 이동 기록](migration_20261012/moves.jsonl)
- [문서 링크 변경 후 해시](migration_20261012/finalized.json)

NTFS 연결은 다른 운영체제나 ZIP·Git 복제에 자동 보존된다고 보장할 수 없다. 새 위치의 실제 파일을 먼저 복원한 뒤 `scripts/maintenance/restore_compatibility.ps1`로 이전 경로를 다시 만든다. 파일이 존재하는 경우 덮어쓰지 않는다. 이동 스크립트는 재실행용 실행기가 아니며 이미 기록이 있으면 중단한다.

실행 시작 시 C: 여유 공간이 0으로 계획 파일 저장이 한 번 실패했다. 빈 실패 파일을 보존하고 `.runtime`의 .pyc 전용 캐시를 정리해 약 142MB의 여유를 확보했다. 가중치·패키지 소스·연구 데이터는 삭제하지 않았다. 이후 최종 확인에서는 여유 공간이 9,404,395,520 bytes로 관측됐다. 그 사이 증가한 공간 전체를 이번 캐시 정리 효과라고 주장하지 않는다. 본 학습 직전에 공간을 다시 확인한다.

최종 검증: 이동 시 모든 파일의 전후 해시 일치, 보호 파일 66개 불변, 검사한 문서 링크 모두 정상, 회귀 테스트 19개 통과. [검증 결과](migration_20261012/verification.json)와 [테스트 기록](migration_20261012/tests.md)을 참고한다.
