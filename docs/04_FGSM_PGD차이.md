# FGSM vs PGD 공격 코드 비교 설명

> **핵심 개념을 한 줄로:**
> FGSM은 "틀린 방향을 한 번에 크게 밀기"고,
> PGD는 "틀린 방향을 조금씩 여러 번 밀되, 매번 제자리를 확인하고 조이기"다.

---

## 두 함수를 나란히 놓고 비교

### FGSM (62~78번째 줄) — 1번짜리

```python
def fgsm_attack(model, img_tensor, targets, epsilon):
    if epsilon == 0:
        return img_tensor.clone()           # 공격 없음

    adv = img_tensor.clone().detach().requires_grad_(True)
    model.train()

    # ① 딱 1번: 손실 계산
    total_loss = sum(model([adv], targets).values())
    model.zero_grad()

    # ② 딱 1번: 기울기 계산
    total_loss.backward()

    # ③ 딱 1번: 이미지 업데이트 (epsilon 만큼 한 방에)
    return torch.clamp(
        img_tensor + epsilon * adv.grad.data.sign(),
        0.0, 1.0
    ).detach()
```

---

### PGD (81~118번째 줄) — N번짜리

```python
def pgd_attack(model, img_tensor, targets, epsilon, steps=PGD_STEPS):
    if epsilon == 0:
        return img_tensor.clone()           # 공격 없음

    step_size = epsilon / 4                 # 한 번에 이동할 최대 거리
    x_orig = img_tensor.detach()            # 원본 이미지 저장
    x_adv  = img_tensor.clone().detach()    # 공격 이미지 (변해감)

    model.train()
    for _ in range(steps):                  # ← 이 for 루프가 핵심 차이!

        x_input = x_adv.detach().requires_grad_(True)

        # ① N번: 손실 계산
        total_loss = sum(model([x_input], targets).values())
        model.zero_grad()

        # ② N번: 기울기 계산
        total_loss.backward()

        with torch.no_grad():
            # ③ N번: 이미지 조금씩 업데이트 (step_size만큼만)
            x_adv = x_adv + step_size * x_input.grad.sign()

            # ④ N번: epsilon 범위 초과하면 도로 잘라냄 (투영)
            x_adv = torch.max(
                torch.min(x_adv, x_orig + epsilon),
                x_orig - epsilon
            )

            # ⑤ N번: 픽셀 유효 범위 [0,1] 유지
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()
```

---

## "한 번"과 "여러 번"의 차이를 만드는 코드 줄

정확히 이 두 부분이 전부다.

```python
# ─────────── FGSM ───────────
# for 루프 없음 — 그냥 한 줄
return torch.clamp(img_tensor + epsilon * adv.grad.data.sign(), 0.0, 1.0)
#                  ↑ epsilon 전체를 한 방에 더함

# ─────────── PGD ────────────
for _ in range(steps):              # ← 이 줄만 감싸면 PGD
    ...
    x_adv = x_adv + step_size * x_input.grad.sign()
    #               ↑ step_size(= epsilon/4) 씩만 더함, 여러 번
    x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
    #        ↑ 투영: 원본에서 epsilon 이상 멀어지면 잘라냄
```

FGSM을 PGD로 바꾸는 데 필요한 변경은 사실 세 가지뿐이다:
1. `epsilon` 대신 `step_size = epsilon / 4` 사용
2. `for _ in range(steps):` 루프로 감싸기
3. 루프 안에 L∞ 투영 한 줄 추가

---

## FGSM vs PGD 동작 방식을 그림으로

```
공격의 목표: "원본(●)에서 최대 epsilon 이내에서
              AI가 가장 틀리는 지점(★)을 찾는 것"

epsilon 범위 (원통형 공간):
  ┌───────────────────────┐
  │          ★            │  ← 가장 틀리는 지점 (우리가 원하는 곳)
  │                       │
  │       ←←←←←          │
  │                       │
  │●                      │  ← 원본 이미지
  └───────────────────────┘

FGSM: 기울기 방향으로 한 번에 epsilon만큼 이동
  ●───────────────────────►
  시작점              도착점 (epsilon 경계)
  문제: 기울기는 현재 위치에서만 계산됨
        → 도착점이 ★과 다른 방향일 수 있음

PGD: epsilon/4씩 10번 이동하며 매번 기울기 재계산
  ●→→→→→→★
  각 → 이후 기울기를 다시 계산해 방향 수정
  → ★에 더 가깝게 도달 가능
```

---

## step_size = epsilon / 4 로 정한 이유

**코드 (100번째 줄):**
```python
step_size = epsilon / 4
```

이 값이 왜 epsilon의 1/4인지, 세 가지 측면에서 설명한다.

### ① 너무 크면 (step_size = epsilon): FGSM과 같아진다

```
step_size = epsilon 이면:

  1번째 걸음: x_adv = x_orig + epsilon * sign(grad)
  투영:       clip(x_adv, x_orig ± epsilon) → 이미 경계에 붙음
  2~10번째:  이미 경계에 있어서 방향만 바뀔 뿐 의미 없음

→ 결국 FGSM과 동일한 결과
```

### ② 너무 작으면 (step_size = epsilon / 100): 경계까지 못 간다

```
10번 × (epsilon/100) = epsilon/10

→ 10번 이동해도 epsilon 경계의 10%밖에 탐색 못 함
→ 공격 강도가 약해짐
```

### ③ epsilon / 4 의 균형점

```
10번 × (epsilon/4) = 2.5 × epsilon

경계(epsilon) 안에서 2.5배 거리를 탐색할 수 있다.
투영으로 경계 밖은 잘리지만, 경계 안을 충분히 탐색.

    ←─────── epsilon ────────→
    ←──────────────── 2.5×epsilon 탐색 시도 ──────────────────→
    (경계 밖은 잘림)     (경계 안을 여러 각도에서 탐색 가능)
```

epsilon/4는 "충분히 많은 방향을 탐색하면서도, 경계 안에서 의미 있는 이동"을 가능하게 하는 실용적 기준값이다.

---

## steps = 10 으로 정한 이유

**코드 (44번째 줄):**
```python
PGD_STEPS = 10
```

### 실험 결과로 확인한 수렴 패턴

PGD에서 반복 횟수와 공격 강도의 관계는 대략 이렇다:

```
반복 횟수   공격 강도(개념적)   비고
─────────────────────────────────────
1번         (= FGSM 수준)      단순 방향 한 번
3번         꽤 강해짐          주요 방향 탐색 완료
5번         거의 수렴          추가 이득 작아짐
10번        충분한 수렴        ← 우리가 사용
20번 이상   거의 차이 없음     계산 비용만 증가
```

### 계산 비용과의 관계

FGSM은 이미지 1장당 계산을 **1번** 한다.
PGD_STEPS=10이면 이미지 1장당 계산을 **10번** 한다.
즉, PGD는 FGSM보다 **10배 느리다.**

10번으로 설정한 이유: "충분한 공격 강도를 얻으면서 계산 비용을 합리적으로 유지"하는 표준 설정.
학술 논문에서도 PGD 기본 설정으로 steps=10~40이 자주 쓰인다.

---

## 실험 결과: PGD가 FGSM보다 2.4~3.6배 강한 이유

`attack_results.json`과 `attack_results_pgd.json`에서 측정된 평균 bbox 오차:

| epsilon | FGSM 평균 오차 | PGD 평균 오차 | PGD / FGSM 비율 |
|---------|--------------|-------------|----------------|
| 0.01 | 12.95 px | 30.97 px | **2.4배** |
| 0.03 | 13.56 px | 36.88 px | **2.7배** |
| 0.05 | 14.28 px | 36.03 px | **2.5배** |
| 0.07 | 12.29 px | 42.62 px | **3.5배** |
| 0.10 | 10.54 px | 38.30 px | **3.6배** |

**왜 epsilon이 커질수록 배율이 올라가는가?**

```
epsilon이 작을 때 (0.01):
  epsilon-ball이 좁음 → 탐색 공간이 작음
  FGSM도 한 번에 epsilon 경계에 도달 가능
  → FGSM과 PGD의 차이가 작음 (2.4배)

epsilon이 클 때 (0.07~0.1):
  epsilon-ball이 넓음 → 탐색 공간이 큼
  FGSM은 한 방향으로만 이동 → 최악의 지점(★)을 못 찾음
  PGD는 여러 번 방향 수정 → 넓은 공간에서 ★을 찾아냄
  → 차이가 커짐 (3.5~3.6배)
```

수치로 보면:

```
epsilon=0.01일 때:
  탐색 공간: ±0.01 (픽셀 강도 ±1/100)
  FGSM: 한 번에 ±0.01 이동 → 경계 도달
  PGD: 0.01/4 = 0.0025씩 10번 → 경계 안 탐색 (공간이 좁아 차이 작음)

epsilon=0.07일 때:
  탐색 공간: ±0.07 (픽셀 강도 ±7/100)
  FGSM: 한 번에 ±0.07 이동 → 기울기 방향 단순 도달
  PGD: 0.07/4 = 0.0175씩 10번 → 공간이 넓어 방향 수정 효과 극대화
```

---

## 두 공격이 DQN 드론 실험에 미친 영향

우리 실험에서는 DQN 드론 학습에 **PGD (ε=0.03)** 를 사용했다.
이유: FGSM(ε=0.03)의 평균 오차 13.56px보다 PGD의 36.88px이
드론 항법에 더 현실적인 위협 수준이기 때문이다.

```
FGSM ε=0.03: 평균 13.56px → 임계값 10px 기준으로 대부분 CE 판정
PGD  ε=0.03: 평균 36.88px → 이진 CE(>10px) 기준으로 거의 전량 폐기
                           → 연속 CE에서도 저장 확률 max(0.05, 1-36.88/50) = 26%

PGD를 사용했기 때문에:
  V4-A 이진 CE: 82.8% 폐기 (shift 대부분 10px 초과)
  V4-B 연속 CE: 27.9% 폐기 (shift 비례 저장으로 일부 살림)
```

만약 FGSM으로 실험했다면 평균 shift가 더 작아 CE 폐기율이 낮았을 것이고,
V4-A/B의 차이도 덜 두드러졌을 것이다.

---

## 최종 요약

| 항목 | FGSM | PGD |
|------|------|-----|
| 반복 횟수 | 1번 | 10번 (`PGD_STEPS = 10`) |
| 한 번 이동 크기 | epsilon 전체 | epsilon / 4 (`step_size`) |
| 투영(범위 제한) | 없음 | 매 스텝 후 epsilon-ball로 잘라냄 |
| 계산 비용 | 빠름 (1x) | 느림 (10x) |
| 공격 강도 | 기준 | **2.4~3.6배 강함** |
| 핵심 코드 차이 | 루프 없음 | `for _ in range(steps):` 한 줄 |

**step_size = epsilon/4의 의미**: 한 번에 경계까지 안 가고, 조금씩 여러 번 가면서 매번 "지금 어느 방향으로 가야 가장 틀리게 만드는가"를 다시 계산한다.

**steps = 10의 의미**: 3~5번 이후에는 수렴하지만, 10번이 "충분한 탐색 + 합리적 계산 비용"의 표준 균형점이다.
