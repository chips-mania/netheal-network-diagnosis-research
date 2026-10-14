# 재구성 검증 기록

2026-10-12 KST.

## 회귀 테스트

프로젝트 루트에서 실행:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.runtime/venv/Scripts/python.exe -B -m pytest tests/unit netheal-topology-generalization/tests/test_research_suite.py netheal-topology-generalization/tests/test_factorial_v3.py netheal-topology-generalization/tests/test_api_comparison.py -q -p no:cacheprovider
```

결과: **19 passed in 9.71s**. 경로 경계·호환 접근·공개 관측·일반화·모의 API 동작을 확인했다. 전체 LAYA 모델 테스트나 GPU 학습은 실행하지 않았다.

## 실행·데이터 보존

- 새 패키지의 metrics_utils, research_metrics, controlled_faults 3개 import 성공.
- Jev evaluate와 LAYA prepare 진입점의 명령 미리보기에서 `execute: false` 확인. 실제 API 호출·데이터 재생성·학습 없음.
- 이동 계획 115개 작업 완료: junction 31개, hard link 69개, 문서 15개.
- 파일 2,362개, 11,403,449,451 bytes 이동. 모든 파일은 이동 직후 전후 SHA-256 검증을 통과했다.
- 문서는 이동 후 상대 링크를 변경했으며 변경 후 해시는 finalized.json에 별도로 기록했다.
- 보호 소스·명세 66개의 SHA-256 불변. 사용자 기존 수정 파일도 보존.
- 로컬 링크 검증 결과 및 파일 동일성은 verification.json 참조.

새 패키지 editable 설치, 가상환경 교체, Git commit·index 변경은 실행하지 않았다. Windows 연결 경로는 다른 컴퓨터에서 복구가 필요할 수 있다. 최종 구조 안내의 한계를 함께 확인한다.
