# 원본 Claude purple 이미지 프롬프트 검증

2026-10-13 KST. **앞서 제시한 SYSTEM_PROMPT가 원본 Claude 평가의 고정 purple 이미지에도 들어 있음을 검증했다.**

검증 연결:

1. `manikyabard-20260201-100940.provenance.json`의 Claude 이미지: `manikya/netheal-purple-agent@sha256:afc3a3d86cd668defb721f758cef59f247ef65da3405c08b690e17de66dc00f7`.
2. Docker Registry에서 위 digest를 직접 조회. 내려받은 OCI index 바이트의 SHA-256 일치.
3. linux/amd64 플랫폼 manifest: `sha256:09c3ac185f753a2d49767d17bba8cc936b443a9305822dfae816e6b16ce643b9`. manifest 및 config digest 일치.
4. 마지막 파일시스템 레이어가 `COPY netheal/ ./netheal/`. 압축 크기 303,949 bytes, digest `sha256:6e2436e6bfb7337f3566ecda5d5674af6d691f1101315b6a5f9b85cb292779ea`를 다운로드·검증했다. 이 레이어 위의 기록은 ENV/ENTRYPOINT/CMD 등 파일 변경 없는 설정이다.
5. 두 파일만 추출해 보관된 green 소스와 비교. `gpt_agent.py` SHA-256: `59ea19d9711aaede9a0df83b67600d01823cff15b15647cf544d8c3073f6d4ec`; `purple_server.py`: `79605ded83ffb1d47964e798e1323bfff01ab2b33622c735719b820871ff7138`. 둘 다 byte-for-byte 일치.
6. 이미지 entrypoint는 `python -m netheal.aaa.purple_server`. 기본 CMD는 dummy지만 제출 TOML의 `solver="gpt"`를 compose 생성기가 `--solver gpt`로 덮어쓴다. 제출 provider는 `bedrock`, temperature는 0이다. purple_server의 gpt 경로가 GPTAgent/SYSTEM_PROMPT를 사용한다.

따라서 앞서 보인 프롬프트와 입력 구성 코드는 단순한 유사 구현이 아니라, provenance에 기록된 Claude 실행 이미지에 포함된 동일 코드다. 별도 실제 HTTP 요청·응답 로그를 복구했다는 뜻은 아니다. 이미지와 provenance가 입증하는 실행 구성을 확인한 것이다.

실제 입력 구성: 기본 SYSTEM_PROMPT에 과제·목표·단계 제한을 덧붙이고, user 메시지에 힌트·도구 이름을 전달한다. 도구 정의는 Bedrock toolConfig로 전달한다. 이후 assistant 텍스트/toolUse와 toolResult JSON을 누적한다. 기본 프롬프트에는 도구 호출 전 관측·가설·선택 이유·기대 정보를 설명하라는 지시가 있다.

산출물: `runs/official_benchmark/claude/2026-10-13_purple_image_audit/`의 digest별 manifest, config.json, source.tar.gz, app/netheal/aaa의 두 추출 파일, verification.json. 외부 이미지는 실행하지 않았고 API 추론도 호출하지 않았다.
