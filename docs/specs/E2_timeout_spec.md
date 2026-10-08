# E2 — Timeout-as-Incorrect-Diagnosis

E1 seed 1은 No-Go였다. Policy가 `scan → diagnose`라 Φ가 오를 행동이 없었다.  
E1 스펙의 E2는 “graded diagnosis + dynamic timeout”이었으나, **graded/symmetric partial은 9.13-A에서 이미 실패**했다. E2는 아직 안 건드린 축만 분리한다.

## 한 가지 항

원본 exact / wrong / partial / step penalty는 **그대로**다.  
`truncated`이고 이번 episode에 diagnosis가 없으면, 원본 incorrect-diagnosis와 같은 스케일 페널티를 한 번 더한다.

```
6 devices, scaling_factor=10 → multiplier 1.6 → timeout extra ≈ −16.0
timeout return ≈ −1.5 + −16.0 = −17.5
wrong diagnose at step 2 ≈ −16.2
exact 그대로 ≈ +15.9 − 0.1×steps
```

Timeout이 오진단보다 싸지 않게 만든다. Gate / Factorized / Symmetric / PBRS와 섞지 않는다.

## 하지 않는 것

- diagnosis action의 type/location partial을 다시 바꾸지 않는다 (9.13-A 재실행 금지)
- 72-way action 유지
- seed 1 Go/No-Go: non-link DSR Δ < +3%p **그리고** shortcut Δ > −10%p 이면 No-Go (도구를 안 쓰면 shortcut이 그대로다)
- seed 2/3는 Go일 때만

Control: Masked PPO, original reward.  
E2: Masked PPO + timeout penalty only.
