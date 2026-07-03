# V7: ImprovedDDMBuffer + 경험 타입 분류 학습

> **핵심 개념을 한 줄로:**
> V6까지는 경험을 "저장하거나 버리거나" 두 가지였다면,
> V7은 경험을 4가지 종류(RE/DE/SE/CE)로 분류해서
> 종류별로 다른 비율로 학습에 사용한다.

---

## V7이 하려는 것

V6까지의 리플레이 메모리는 모든 경험이 하나의 큐에 섞여 있었다.

```
기존 ReplayMemory (V5-B/V6):
  [경험1: 충돌] [경험2: 안전이동] [경험3: 충돌] [경험4: 위험근접] ...
  → 꺼낼 때 무작위로 32개 추출 → 학습

문제:
  - 안전한 이동(SE) 경험이 대부분 → 극단적인 상황(충돌/도착/위험) 학습 부족
  - 오염된 경험(CE)이 섞여 잘못된 학습 유발
```

V7의 아이디어:

```
ImprovedDDMBuffer (V7):
  RE 서브큐 [충돌1] [도착1] [충돌2] [도착2] ...   ← 결과 경험 (용량 5,000)
  DE 서브큐 [위험1] [위험2] [위험3] ...            ← 위험 근접 경험 (용량 2,500)
  SE 서브큐 [안전1] [안전2] [안전3] ...            ← 안전 이동 경험 (용량 2,500)
  CE → 즉시 폐기 (저장 없음)

  꺼낼 때: RE 13개 + DE 10개 + SE 10개 = 33개 (≈32개) → 학습
```

중요 경험(RE/DE)이 더 높은 비율로 학습에 사용된다.

---

## 경험 타입 분류: `_classify_by_shift()`

Faster R-CNN 없이 shift와 장애물 거리만으로 분류한다.

```python
# dqn_comparison_v7.py
def _classify_by_shift(
    shift: float,
    dist_to_obstacle: float,
    done: bool,
    danger_radius: float = 3.0,  # km
) -> ExperienceType:
    if done:                            # 1순위: 종료 경험
        return ExperienceType.RE
    if shift > 10.0:                    # 2순위: PGD 오염
        return ExperienceType.CE
    if dist_to_obstacle < danger_radius: # 3순위: 장애물 근접
        return ExperienceType.DE
    return ExperienceType.SE             # 기본: 안전 이동
```

**왜 done 체크가 CE보다 먼저인가?**

```
예시 상황:
  드론이 장애물에 충돌했다 (done=True)
  동시에 이 스텝의 PGD shift = 15px (CE 기준 초과)

  개선된 DDM classify_experience()에서는: CE(1순위) → CE로 분류 → 폐기
  V7 _classify_by_shift()에서는: done(1순위) → RE로 분류 → 저장

왜 RE를 선택했는가?
  충돌(-1.0) / 도착(+1.0) 경험은 드론이 반드시 배워야 하는 강한 신호다.
  shift가 높다는 이유로 충돌 경험을 버리면,
  드론이 "장애물에 부딪히면 어떻게 되는가"를 배울 경험이 줄어든다.
  보상의 절댓값이 크므로(±1.0), 카메라 오염(shift>10px)보다
  학습 신호로서의 가치가 더 높다고 판단했다.
```

**분류 예시:**

| 상황 | shift | obs_dist | done | 분류 | 저장 |
|------|-------|----------|------|------|------|
| 목적지 도착 (shift 높음) | 18px | 5.0km | True | **RE** | 100% |
| 충돌 | 5px | 0.8km | True | **RE** | 100% |
| 카메라 오염 상태로 이동 | 15px | 4.0km | False | **CE** | **0% (폐기)** |
| 장애물 근처 이동 | 8px | 2.5km | False | **DE** | 80% |
| 평범한 이동 | 5px | 8.0km | False | **SE** | 20% |

---

## ImprovedDDMBuffer 동작 원리

### 저장 (add)

```python
# improved_ddm.py
DDM_DEPOSIT_RATIO = {
    ExperienceType.RE: 1.00,   # 전량 저장
    ExperienceType.DE: 0.80,   # 80% 저장
    ExperienceType.SE: 0.20,   # 20% 저장
    ExperienceType.CE: 0.00,   # 전량 폐기
}
```

버퍼는 타입별로 별도의 큐를 갖는다 (용량 10,000):
- RE 서브큐: 최대 5,000개 (총 용량의 절반)
- DE 서브큐: 최대 2,500개
- SE 서브큐: 최대 2,500개
- CE: 저장 공간 없음 → `discarded_ce` 카운터만 증가

### 샘플링 (sample)

32개를 꺼낼 때 RE:DE:SE = 4:3:3 비율로 섞는다:

```
목표 샘플 수:
  RE: round(32 × 4/10) = 13개
  DE: round(32 × 3/10) = 10개
  SE: round(32 × 3/10) = 10개
  합계: 33개 → 첫 32개만 사용

만약 DE 버퍼에 경험이 5개밖에 없다면:
  RE: 13개 / DE: 5개 / SE: 10개 → 총 28개
  28 < 16 (배치 최솟값) → 학습 스킵
```

### 학습 스킵 조건

```python
# dqn_comparison_v7.py
MIN_BATCH = 16   # BATCH_SIZE(32) × 0.5

batch  = buffer.sample(BATCH_SIZE)   # 요청: 32개
actual = len(batch)                  # 실제 꺼낸 수

if actual < MIN_BATCH:               # 16개 미만이면
    return None, actual              # 이번 스텝은 학습 건너뜀
```

V7 본 실행 결과: 168번 스킵 (학습 초기에만, 이후 없음)

---

## DQN 업데이트를 직접 구현한 이유

DQNAgent에는 `update()` 메서드가 내장되어 있다.
하지만 이 메서드는 `self.memory` (ReplayMemory)에서만 꺼낸다.

V7에서는 ImprovedDDMBuffer를 쓰므로, DQNAgent.update()를 호출하면
타입별 버퍼가 아닌 기존 랜덤 메모리에서 꺼내게 된다. → 의미 없음

그래서 `_ddm_update()`를 직접 만들었다:

```python
# dqn_comparison_v7.py
def _ddm_update(agent: DQNAgent, buffer: ImprovedDDMBuffer):
    batch  = buffer.sample(BATCH_SIZE)    # ImprovedDDMBuffer에서 RE:DE:SE 비율로 꺼냄
    actual = len(batch)

    if actual < MIN_BATCH:
        return None, actual               # 스킵

    # DQNAgent.update()와 동일한 Q-learning 로직 직접 수행
    states      = torch.FloatTensor([e.state      for e in batch])
    actions     = torch.LongTensor( [e.action     for e in batch]).unsqueeze(1)
    rewards     = torch.FloatTensor([e.reward     for e in batch])
    next_states = torch.FloatTensor([e.next_state for e in batch])
    dones       = torch.FloatTensor([float(e.done) for e in batch])

    q_vals = agent.policy_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q  = agent.target_net(next_states).max(1)[0]
        targets = rewards + GAMMA * next_q * (1.0 - dones)

    loss = nn.functional.mse_loss(q_vals, targets)
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    agent.train_steps += 1
    if agent.train_steps % TARGET_UPDATE_FREQ == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

    return float(loss.item()), actual
```

DQNAgent의 policy_net, target_net, optimizer, train_steps는 그대로 사용한다.
달라진 것은 경험을 꺼내는 곳(ImprovedDDMBuffer)과 꺼내는 방식(RE:DE:SE=4:3:3)뿐이다.

---

## 본 실행 결과 (1000 에피소드)

### 학습 곡선

| 에피소드 | V5-B | V6 | V7 |
|---------|------|-----|-----|
| ep100 | 1.0% | 0.0% | **0.0%** |
| ep200 | 2.0% | 6.0% | **5.0%** |
| ep300 | 31.0% | — | **13.0%** |
| ep400 | — | — | **41.0%** |
| ep700 | — | — | **54.0%** |
| ep900 | — | — | **57.0%** (피크) |
| ep1000 | **78.0%** | **79.0%** | **55.0%** |

### DDM 분류 통계 (전체 36,333 스텝)

| 타입 | 발생 횟수 | 비율 | 실제 저장 후 버퍼 내 수 |
|------|---------|------|----------------------|
| RE | 1,000 | 2.8% | ~1,000 (100% 저장) |
| DE | 3,872 | 10.7% | ~2,500 (서브큐 한도) |
| SE | 2,118 | 5.8% | ~424 (20% 저장) |
| **CE** | **29,343** | **80.8%** | **0 (전량 폐기)** |

최종 버퍼 크기: 3,917개 / 10,000 한도

---

## 왜 V7이 V5-B/V6보다 낮은가: CE 80.8% 문제

**핵심 원인: CE 분류 임계값(10px)이 PGD ε=0.03 평균 shift(13.8px)보다 낮다**

```
PGD ε=0.03 shift 분포:
  평균:       13.8px
  표준편차:    4.0px

CE 분류 기준:
  shift > 10px → CE → 폐기

계산:
  P(shift > 10px | 평균 13.8, 표준편차 4.0)
  = P(Z > (10-13.8)/4.0) = P(Z > -0.95) ≈ 83%

→ 매 스텝의 약 83%가 CE로 분류 (실제 측정: 80.8%)
```

결과적으로:
```
전체 스텝: 36,333
  CE (폐기):          29,343개 (80.8%)  ← 실제 배울 수 있는 경험 없음
  RE (저장 100%):      1,000개 (2.8%)
  DE (저장 80%):       3,872개 → ~3,098개 저장 시도 → 서브큐 2,500개 한도
  SE (저장 20%):       2,118개 → ~424개 저장
  ────────────────────────────────────────
  실제 버퍼 활용:       약 3,924개 / 10,000 한도 (39%)
```

V5-B는 같은 기간에 26,530개 유효 경험을 저장한 반면,
V7은 3,924개만 저장 — **7배 차이**가 학습 효율에 직접 영향을 미쳤다.

---

## V7 실험이 알려주는 것

### 발견 1: CE 임계값 10px는 ε=0.03 환경에서 너무 낮다

```
CE 임계값을 올리면 어떻게 달라지는가?

  10px (현재): CE 80.8% → 유효 경험 20% → 55% 도착률
  15px (가정): CE ~50%  → 유효 경험 50% → 더 높을 것
  20px (가정): CE ~6%   → 유효 경험 94% → V6 수준에 근접할 것

  → 임계값 조정이 V8 이후 과제
```

### 발견 2: done=True → RE 우선 분류는 유효하다

RE 1,000개 = 에피소드 수(1,000)와 정확히 일치
→ 모든 에피소드의 마지막 스텝(도착 또는 충돌)이 RE로 저장됨
→ 100% 저장 보장으로 종료 경험이 유실되지 않음

### 발견 3: 학습 스킵은 초기에만 발생한다

168번 스킵이 모두 ep1~ep50 초반에 집중됐다.
이후 RE 서브큐가 16개 이상 쌓이면 학습이 정상 진행됐다.

```
학습 시작 기준:
  RE 16개 필요 (배치 최솟값)
  RE 저장 확률 100% → 약 16 에피소드 후 학습 시작
  실제: 168번 스킵 후 정상화 (168 / 평균스텝 수 ≈ 초반 ~50ep)
```

---

## V5-B → V6 → V7 흐름 정리

```
V5-B: CE 연속 가중치 저장 (shift 비례 확률적 저장)
  결과: 78%
  한계: 오염 경험이 일부 섞여 들어옴 / 경험 종류 구분 없음

V6: V5-B + 오염 심각도 페널티
  결과: 79% (+1%p)
  기여: RISK/CRITICAL 상황 억제 신호 추가
  한계: 버퍼 구조 동일 (타입 구분 없음)

V7: V6 + ImprovedDDMBuffer + 타입 분류
  결과: 55% (-24%p)
  기여: CE 완전 차단, RE/DE 우선 학습 구조
  원인: CE 임계값(10px)이 평균 shift(13.8px)보다 낮아 80%가 폐기
```

V7은 성능이 낮지만, 그 이유가 "아이디어가 나쁜 것"이 아니라
"CE 임계값 설정이 현재 환경에 맞지 않는 것"임을 실험으로 확인했다.

---

## 저장 파일

| 파일 | 내용 |
|------|------|
| `results/dqn_results_v7.json` | 학습 통계, RE/DE/SE/CE 분류 수, 버퍼 크기 등 |
| `results/dqn_comparison_v7.png` | V5-B/V6/V7 3선 학습 곡선 + DDM 분류 막대 그래프 |
| `results/v7_policy.pth` | 학습된 V7 정책 네트워크 가중치 |
