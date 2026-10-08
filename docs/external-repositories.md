# 외부 저장소와 고정 버전

이 프로젝트는 아래 저장소를 Git submodule로 참조합니다. 기존 작업 사본은 이동하거나 수정하지 않았으며, 자체 실험 코드는 `netheal-topology-generalization/`에 있습니다. 커밋별 원본 라이선스와 사용 조건을 확인하세요.

## NetHeal 환경

- 경로: `netheal-ai-agent-benchmark/`
- 원본: https://github.com/cisco-ai-platform/netheal-ai-agent-benchmark
- 고정 커밋: `815dba180bc1f700a978013ce62837f2021285ca`
- 용도: 네트워크 토폴로지·장애·도구·Gym 인터페이스를 제공하는 실험 환경
- 라이선스: [원본 LICENSE](../netheal-ai-agent-benchmark/LICENSE)

## OutlierDenStream 참고 구현

- 경로: `OutlierDenStream-BigDama18/`
- 원본: https://github.com/anrputina/OutlierDenStream-BigDama18
- 고정 커밋: `3fc2db8bee696dc20a902d1ecdf9ecd5030dd8f3`
- 용도: telemetry 기반 BGP 이상탐지 논문의 재현용 참고 코드·데이터
- 독립 LICENSE 파일이 확인되지 않아 자체 연구 코드의 라이선스로 재분류하지 않습니다. 원본 저작자의 이용 조건을 확인하세요.

## Cisco telemetry 데이터

- 경로: `telemetry/`
- 원본: https://github.com/cisco-ie/telemetry
- 고정 커밋: `e2bd39fbafd0a6d716d5a34f1a2fe41e40b3bb86`
- 용도: 정상 패턴 전이·traffic regime shift 연구 계획의 원본 데이터
- 라이선스: [LICENSE.CDLA](../telemetry/LICENSE.CDLA), 원본 README 표기 CDLA-Permissive 1.0
- 크기가 크므로 NetHeal 실험만 실행할 때는 내려받을 필요가 없습니다.

## 복원

```powershell
# 저장소 루트
git submodule update --init netheal-ai-agent-benchmark
# 선택 사항: 참고 코드와 전체 원본 데이터
git submodule update --init OutlierDenStream-BigDama18 telemetry
# 현재 고정 버전 확인
git submodule status
```

외부 저장소 안의 무시된 로컬 파일과 캐시는 새 상위 저장소에 포함되지 않습니다. 기존 로컬 파일과 각 저장소의 Git 이력은 그대로 보존합니다.
