"""이 파일은 v8까지 모든 버전이 매 스텝 예외 없이 공격받는 조건에서만
학습했다는 한계를 확인하고, 공격을 확률적으로 적용해 정상 상태와
오염 상태를 모두 경험하도록 바꾼 버전이다. 또한 v8에서 ExperienceType.CE가
실제로는 어떤 경험에도 부여되지 않고 폐기 카운터로만 쓰였다는 점을
확인하고 이름을 discard_count로 정리했다. 이 결과는 단일 시드
기준이며 반복 검증은 수행하지 않았다.

변경점 (v8 대비):
  1. 확률적 PGD 공격 (ATTACK_PROB = 0.5)
       is_attacked = random.random() < ATTACK_PROB
       공격 적용:   c_state, shift = pgd_attack(state, D_obs, theta_obs)
       공격 미적용: c_state = state, shift = 0.0
       next_state도 같은 스텝의 is_attacked 값 그대로 사용 (혼재 없음)
  2. ExperienceType.CE 명칭 정리
       type_cnt[CE] → discard_count 변수로 대체
       CE는 어떤 Experience 객체에도 exp_type으로 부여되지 않는다.
       (buffer.discarded_ce는 imporved_ddm.py 내부 카운터로 유지)
  3. DE/SE 4분류 추적 신설
       de_from_clean    / se_from_clean    : shift<=10px 구간
       de_from_survived / se_from_survived : shift>10px 구간 survive_prob 통과
  4. 학습 후 두 조건으로 정책 평가
       조건 A: ATTACK_PROB=1.0 (항상 공격), 200ep, epsilon=0.05 고정
       조건 B: ATTACK_PROB=0.5 (학습 조건과 동일), 200ep, epsilon=0.05 고정

실행 방법:
  시험 (200ep):  python src/dqn_comparison_v9.py --episodes 200
  본 실행:       python src/dqn_comparison_v9.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
import torch.nn as nn

# ── 한글 폰트 ───────────────────────────────────────────────────────────────
_prefer = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Gulim', 'Dotum']
_available = {f.name for f in fm.fontManager.ttflist}
for _font in _prefer:
    if _font in _available:
        plt.rcParams['font.family'] = _font
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 재현성 ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 공유 컴포넌트 import ─────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from dqn_comparison import (
    DQNAgent, pgd_attack, MAX_STEPS,
    BATCH_SIZE, GAMMA, TARGET_UPDATE_FREQ, MEMORY_CAPACITY,
)
from env_test import DroneEnvN, OBSTACLES_8
from improved_ddm import (
    CorruptionLevel, ExperienceType, Experience, ImprovedDDMBuffer,
)

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

EPISODES_V9   = 1000    # v5~v8과 동일
EVAL_EPISODES = 200     # 정책 평가용 에피소드 수

# V9 핵심: 확률적 공격
ATTACK_PROB = 0.5   # 매 스텝 이 확률로 pgd_attack 적용

# v8과 동일한 보상 상수
TIME_PENALTY  = -0.01
DIST_APPROACH = +0.01
DIST_RECEDE   = -0.01

# v8과 동일한 오염 페널티
PENALTY_RISK         = -0.01
PENALTY_CRITICAL     = -0.03
OBS_DIST_RISK_THRESH =  3.0   # km

# 오염 판정 임계값
SHIFT_THRESH_RISK     = 10.0  # px
SHIFT_THRESH_CRITICAL = 20.0  # px

# DE 분류 위험 반경
DANGER_RADIUS = 3.0   # km

# V5-B / V8 연속 가중치 파라미터
SURVIVE_SHIFT_MAX = 50.0

# 배치 최소 비율
MIN_BATCH_RATIO = 0.5
MIN_BATCH       = int(BATCH_SIZE * MIN_BATCH_RATIO)   # = 16


# ══════════════════════════════════════════════════════════════════════════════
# 오염 판정 (v8과 동일)
# ══════════════════════════════════════════════════════════════════════════════

def _assess_shift(shift: float) -> CorruptionLevel:
    """shift=0.0 (공격 미적용)이면 항상 STABLE 반환."""
    if shift <= SHIFT_THRESH_RISK:
        return CorruptionLevel.STABLE
    elif shift <= SHIFT_THRESH_CRITICAL:
        return CorruptionLevel.RISK
    else:
        return CorruptionLevel.CRITICAL


# ══════════════════════════════════════════════════════════════════════════════
# 커스텀 DQN 업데이트 (v8과 동일)
# ══════════════════════════════════════════════════════════════════════════════

def _ddm_update(
    agent: DQNAgent,
    buffer: ImprovedDDMBuffer,
) -> tuple[float | None, int]:
    """ImprovedDDMBuffer에서 배치를 샘플링해 DQN gradient 업데이트."""
    batch  = buffer.sample(BATCH_SIZE)
    actual = len(batch)

    if actual < MIN_BATCH:
        return None, actual

    states      = torch.FloatTensor([e.state           for e in batch])
    actions     = torch.LongTensor( [e.action          for e in batch]).unsqueeze(1)
    rewards     = torch.FloatTensor([e.reward          for e in batch])
    next_states = torch.FloatTensor([e.next_state      for e in batch])
    dones       = torch.FloatTensor([float(e.done)     for e in batch])

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


# ══════════════════════════════════════════════════════════════════════════════
# 학습 루틴
# ══════════════════════════════════════════════════════════════════════════════

def train_v9(n_episodes: int = EPISODES_V9) -> tuple[list[float], list[float], dict]:
    """확률적 PGD 공격 + V8의 연속 가중치 CE 처리 + ImprovedDDMBuffer.

    [확인 포인트 a] attacked_steps 비율:
      매 스텝 random.random() < ATTACK_PROB(0.5)로 결정하므로
      큰 수의 법칙에 의해 attacked_steps ≈ total_steps * 0.5.

    [확인 포인트 b] done=True 우선 RE 처리:
      공격 여부(is_attacked)와 무관하게 done=True이면 RE.
      RE 저장 수 = 에피소드 수 (환경이 MAX_STEPS 내 종료를 보장한다고 가정).

    [확인 포인트 c] discard_count vs type_ce:
      ExperienceType.CE는 어떤 Experience.exp_type에도 설정되지 않는다.
      폐기 횟수는 discard_count 변수와 buffer.discarded_ce로 동일하게 추적.
      이름만 바뀐 것이며 로직은 v8의 type_cnt[CE] 증가와 동일하다.
    """
    env    = DroneEnvN(list(OBSTACLES_8), fallback_pos=[1.0, 1.0])
    agent  = DQNAgent()
    buffer = ImprovedDDMBuffer(capacity=MEMORY_CAPACITY)

    arr_hist:  list[float] = []
    col_hist:  list[float] = []
    window_arr = deque(maxlen=100)
    window_col = deque(maxlen=100)

    total_steps     = 0
    arrival_count   = 0
    risk_count      = 0
    critical_count  = 0
    skipped_updates = 0
    batch_size_sum  = 0
    update_count    = 0

    # V9 추가: 공격 적용/미적용 카운터
    # [확인 포인트 a] 이 두 값의 합 = total_steps, 비율이 ~50%인지 확인
    attacked_steps = 0
    clean_steps    = 0

    # 타입별 분류 카운터
    # [확인 포인트 c] CE는 이 dict에 없음 — ExperienceType.CE는 실제로 부여되지 않음
    type_cnt = {
        ExperienceType.RE: 0,
        ExperienceType.DE: 0,
        ExperienceType.SE: 0,
    }
    # 폐기 카운터 (v8의 type_cnt[CE] 대체, 값은 동일)
    discard_count = 0

    # V8에서 이어받은 고-shift 구간 통계
    high_shift_total    = 0
    high_shift_survived = 0

    # V9 추가: DE/SE 4분류 추적
    de_from_clean    = 0   # shift<=10px → DE (buffer.add() 경유)
    de_from_survived = 0   # shift>10px, survive_prob 통과 → DE (직접 추가)
    se_from_clean    = 0   # shift<=10px → SE (buffer.add() 경유)
    se_from_survived = 0   # shift>10px, survive_prob 통과 → SE (직접 추가)

    # 거리 보상 실측 카운터
    nonterminal_steps = 0
    goal_approach_cnt = 0
    goal_recede_cnt   = 0
    goal_neutral_cnt  = 0

    valid_exp_hist: list[int] = []

    print(f"\n[V9] 학습 시작 ({n_episodes}ep  |  ATTACK_PROB={ATTACK_PROB}  "
          f"ImprovedDDMBuffer + 연속 가중치 생존)")
    print(f"     RISK={PENALTY_RISK}(접근중+{OBS_DIST_RISK_THRESH}km이내)  "
          f"CRITICAL={PENALTY_CRITICAL}(무조건)")
    print(f"     공격: 매 스텝 {ATTACK_PROB:.0%} 확률로 pgd_attack 적용, "
          f"미적용 시 c_state=state, shift=0.0")
    print(f"     분류: done=True->RE  shift>10->survive_prob->DE/SE or 폐기  "
          f"shift<=10->DE/SE(DDM비율)")
    print(f"     CE 명칭 정리: ExperienceType.CE는 실제로 어떤 경험에도 부여되지 않음 "
          f"→ discard_count로 관리")

    for ep in range(n_episodes):
        state   = env.reset()
        arrived = False
        hit     = False
        prev_obs_dist: float | None = None

        for _ in range(MAX_STEPS):
            prev_dist = math.dist(env.pos, env.goal)
            D_obs_km, theta_obs_deg = env.get_raw_obstacle()

            # ── V9 핵심: 확률적 공격 결정 ────────────────────────────────────
            # [확인 포인트 a] is_attacked 는 c_state와 c_next 양쪽에 동일하게 적용
            is_attacked = random.random() < ATTACK_PROB
            if is_attacked:
                c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)
                attacked_steps += 1
            else:
                # 공격 미적용: 원본 상태 그대로, shift=0.0
                # shift=0.0 → _assess_shift → STABLE → 페널티 없음
                # shift=0.0 → shift<=10px 경로 → DE/SE + DDM_DEPOSIT_RATIO
                c_state, shift = state, 0.0
                clean_steps += 1

            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)
            total_steps += 1

            obs_dist   = info["obs_dist"]
            corruption = _assess_shift(shift)

            # ── 비종료 스텝 보상 블록 (v8과 동일) ──────────────────────────
            if not (info["goal_reached"] or info["collision"] or info["out_of_bounds"]):
                nonterminal_steps += 1

                reward += TIME_PENALTY
                curr_dist = math.dist(env.pos, env.goal)
                if curr_dist < prev_dist:
                    reward += DIST_APPROACH
                    goal_approach_cnt += 1
                elif curr_dist > prev_dist:
                    reward += DIST_RECEDE
                    goal_recede_cnt += 1
                else:
                    goal_neutral_cnt += 1

                if corruption is CorruptionLevel.RISK:
                    if (prev_obs_dist is not None
                            and obs_dist < prev_obs_dist
                            and obs_dist < OBS_DIST_RISK_THRESH):
                        reward += PENALTY_RISK
                        risk_count += 1
                elif corruption is CorruptionLevel.CRITICAL:
                    reward += PENALTY_CRITICAL
                    critical_count += 1

            # ── 경험 분류 및 저장 ─────────────────────────────────────────────
            # [확인 포인트 b] done=True → RE (공격 여부 무관, 최우선)
            # [확인 포인트 c] CE exp_type 없음 — discard_count로만 추적
            if done:
                # RE: done=True → 무조건 저장 (DDM_DEPOSIT_RATIO[RE]=1.0)
                type_cnt[ExperienceType.RE] += 1
                D_obs_next, theta_obs_next = env.get_raw_obstacle()
                # next_state도 같은 is_attacked 적용 (혼재 없음)
                if is_attacked:
                    c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
                else:
                    c_next = next_state
                exp = Experience(
                    state=c_state, action=action, reward=reward,
                    next_state=c_next, done=done,
                    exp_type=ExperienceType.RE, corruption=corruption,
                )
                buffer.add(exp)

            elif shift > SHIFT_THRESH_RISK:
                # 고-shift 구간: 연속 가중치 생존 (v8과 동일)
                # shift>10px는 is_attacked=True인 스텝에서만 발생
                # (is_attacked=False → shift=0.0 → 이 경로 진입 불가)
                high_shift_total += 1
                survive_prob = max(0.05, 1.0 - shift / SURVIVE_SHIFT_MAX)

                if random.random() < survive_prob:
                    exp_type = (ExperienceType.DE if obs_dist < DANGER_RADIUS
                                else ExperienceType.SE)
                    high_shift_survived += 1
                    type_cnt[exp_type] += 1

                    if exp_type is ExperienceType.DE:
                        de_from_survived += 1
                    else:
                        se_from_survived += 1

                    D_obs_next, theta_obs_next = env.get_raw_obstacle()
                    # 고-shift는 항상 is_attacked=True이므로 c_next도 항상 공격
                    c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
                    exp = Experience(
                        state=c_state, action=action, reward=reward,
                        next_state=c_next, done=done,
                        exp_type=exp_type, corruption=corruption,
                    )
                    # DDM_DEPOSIT_RATIO 우회: deque에 직접 추가 (v8과 동일)
                    buffer._buffers[exp_type].append(exp)
                    buffer._stats[exp_type] += 1
                else:
                    # [확인 포인트 c] CE는 부여하지 않음 — discard_count만 증가
                    discard_count += 1
                    buffer.discarded_ce += 1   # improved_ddm.py 내부 카운터 동기화

            else:
                # 정상 구간: shift<=10px (공격 미적용 스텝 전부, 저-shift 공격 스텝)
                # buffer.add()가 DDM_DEPOSIT_RATIO(DE=0.8, SE=0.2) 그대로 적용 (v8과 동일)
                exp_type = (ExperienceType.DE if obs_dist < DANGER_RADIUS
                            else ExperienceType.SE)
                type_cnt[exp_type] += 1

                if exp_type is ExperienceType.DE:
                    de_from_clean += 1
                else:
                    se_from_clean += 1

                D_obs_next, theta_obs_next = env.get_raw_obstacle()
                if is_attacked:
                    c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
                else:
                    c_next = next_state
                exp = Experience(
                    state=c_state, action=action, reward=reward,
                    next_state=c_next, done=done,
                    exp_type=exp_type, corruption=corruption,
                )
                buffer.add(exp)

            # ── DDM 업데이트 ─────────────────────────────────────────────────
            loss, actual_batch = _ddm_update(agent, buffer)
            if loss is None:
                skipped_updates += 1
            else:
                batch_size_sum += actual_batch
                update_count   += 1

            prev_obs_dist = obs_dist
            state = next_state
            if info["goal_reached"]:
                arrived = True
                break
            if done:
                hit = info["collision"] or info["out_of_bounds"]
                break

        agent.decay_epsilon()
        if arrived:
            arrival_count += 1
        window_arr.append(1 if arrived else 0)
        window_col.append(1 if hit else 0)
        arr_hist.append(sum(window_arr) / len(window_arr))
        col_hist.append(sum(window_col) / len(window_col))
        valid_exp_hist.append(buffer.size)

        if (ep + 1) % 100 == 0:
            attack_rate  = attacked_steps / total_steps   if total_steps     > 0 else 0.0
            survive_rate = high_shift_survived / high_shift_total if high_shift_total > 0 else 0.0
            disc_rate    = discard_count / total_steps    if total_steps     > 0 else 0.0
            avg_batch    = batch_size_sum / update_count  if update_count    > 0 else 0.0
            print(f"  Ep {ep+1:4d} | 도착률={arr_hist[-1]:.3f}"
                  f" | 충돌률={col_hist[-1]:.3f}"
                  f" | e={agent.epsilon:.3f}"
                  f" | RE={type_cnt[ExperienceType.RE]:,}"
                  f" DE={type_cnt[ExperienceType.DE]:,}(c{de_from_clean},s{de_from_survived})"
                  f" SE={type_cnt[ExperienceType.SE]:,}(c{se_from_clean},s{se_from_survived})"
                  f" 폐기={discard_count:,}({disc_rate:.1%})"
                  f" | 공격={attacked_steps:,}({attack_rate:.1%})"
                  f" HS생존={survive_rate:.1%}({high_shift_survived}/{high_shift_total})"
                  f" | skip={skipped_updates:,} avg_batch={avg_batch:.1f}"
                  f" | 버퍼={buffer.size:,}")

    nt = nonterminal_steps or 1
    avg_dist_reward   = (goal_approach_cnt * DIST_APPROACH
                         + goal_recede_cnt  * DIST_RECEDE) / nt
    avg_step_penalty  = TIME_PENALTY + avg_dist_reward
    avg_batch_final   = batch_size_sum / update_count   if update_count    > 0 else 0.0
    survive_rate_fin  = high_shift_survived / high_shift_total if high_shift_total > 0 else 0.0
    attack_rate_fin   = attacked_steps / total_steps    if total_steps     > 0 else 0.0

    stats = {
        "label":   f"V9 (확률적공격 p={ATTACK_PROB} + 연속CE생존)",
        "episodes": n_episodes,
        "attack_prob":        ATTACK_PROB,
        "total_steps":        total_steps,
        "attacked_steps":     attacked_steps,
        "clean_steps":        clean_steps,
        "attack_rate_actual": float(attack_rate_fin),
        "arrival_count":      arrival_count,
        "final_arr":          float(arr_hist[-1]),
        "final_col":          float(col_hist[-1]),
        "mean_arr_last100":   float(np.mean(arr_hist[-100:])),
        "peak_arr":           float(max(arr_hist)),
        "peak_ep":            int(np.argmax(arr_hist)) + 1,
        # DDM 분류 지표
        "type_re": type_cnt[ExperienceType.RE],
        "type_de": type_cnt[ExperienceType.DE],
        "type_se": type_cnt[ExperienceType.SE],
        # [확인 포인트 c] discard_count = v8의 type_ce와 동일한 의미
        "discard_count": discard_count,
        "discarded_ce":  buffer.discarded_ce,   # = discard_count (동기화 확인용)
        "buffer_size_final": buffer.size,
        # V9 추가: DE/SE 4분류
        "de_from_clean":    de_from_clean,
        "de_from_survived": de_from_survived,
        "se_from_clean":    se_from_clean,
        "se_from_survived": se_from_survived,
        # V8 호환: 고-shift 구간 통계
        "high_shift_total":    high_shift_total,
        "high_shift_survived": high_shift_survived,
        "high_shift_discarded": high_shift_total - high_shift_survived,
        "high_shift_survive_rate": float(survive_rate_fin),
        "high_shift_de": de_from_survived,
        "high_shift_se": se_from_survived,
        # 업데이트 지표
        "skipped_updates": skipped_updates,
        "update_count":    update_count,
        "avg_batch_size":  float(avg_batch_final),
        "train_steps":     agent.train_steps,
        # 오염 페널티
        "risk_penalty_count":     risk_count,
        "critical_penalty_count": critical_count,
        "penalty_risk":           PENALTY_RISK,
        "penalty_critical":       PENALTY_CRITICAL,
        "obs_dist_risk_thresh":   OBS_DIST_RISK_THRESH,
        # 거리 보상 실측
        "nonterminal_steps":  nonterminal_steps,
        "goal_approach_cnt":  goal_approach_cnt,
        "goal_recede_cnt":    goal_recede_cnt,
        "goal_neutral_cnt":   goal_neutral_cnt,
        "goal_approach_rate": goal_approach_cnt / nt,
        "goal_recede_rate":   goal_recede_cnt   / nt,
        "avg_dist_reward":    float(avg_dist_reward),
        "avg_step_penalty":   float(avg_step_penalty),
        # 이력
        "arrival_history":   arr_hist,
        "collision_history": col_hist,
        "valid_exp_hist":    valid_exp_hist,
    }

    torch.save(agent.policy_net.state_dict(), RESULTS_DIR / "v9_policy.pth")
    print(f"\n  [V9] 가중치 저장 -> results/v9_policy.pth")
    return arr_hist, col_hist, stats, agent


# ══════════════════════════════════════════════════════════════════════════════
# 정책 평가 (추론 전용)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(
    agent: DQNAgent,
    attack_prob: float,
    n_eval: int = EVAL_EPISODES,
    label: str = "",
) -> tuple[float, list[float]]:
    """학습 완료된 정책을 epsilon=0.05 고정으로 추론 전용 평가.

    gradient 업데이트 없이 도착률만 측정한다.

    Args:
        agent:       학습 완료된 DQNAgent (policy_net 가중치 로드 완료 상태)
        attack_prob: 평가 시 PGD 공격 적용 확률 (1.0 = 항상, 0.5 = 50%)
        n_eval:      평가 에피소드 수
        label:       출력용 레이블

    Returns:
        (final_arr, arr_hist_eval)
    """
    env = DroneEnvN(list(OBSTACLES_8), fallback_pos=[1.0, 1.0])

    # 평가 모드: epsilon 고정, gradient 없음
    agent.policy_net.eval()
    original_epsilon = agent.epsilon
    agent.epsilon    = 0.05

    arr_hist_eval = []
    window_arr    = deque(maxlen=100)
    arrival_count = 0

    print(f"\n  [{label}] 정책 평가 시작 (attack_prob={attack_prob}, {n_eval}ep, "
          f"epsilon=0.05 고정, 추론 전용)")

    with torch.no_grad():
        for ep in range(n_eval):
            state   = env.reset()
            arrived = False

            for _ in range(MAX_STEPS):
                D_obs_km, theta_obs_deg = env.get_raw_obstacle()

                is_attacked = random.random() < attack_prob
                if is_attacked:
                    c_state, _ = pgd_attack(state, D_obs_km, theta_obs_deg)
                else:
                    c_state = state

                action = agent.act(c_state)
                next_state, reward, done, info = env.step(action)

                state = next_state
                if info["goal_reached"]:
                    arrived = True
                    break
                if done:
                    break

            if arrived:
                arrival_count += 1
            window_arr.append(1 if arrived else 0)
            arr_hist_eval.append(sum(window_arr) / len(window_arr))

    final_arr = arr_hist_eval[-1] if arr_hist_eval else 0.0
    print(f"  [{label}] 평가 완료 | 도착률={final_arr:.3f} ({arrival_count}/{n_eval})")

    # 평가 후 상태 복원
    agent.policy_net.train()
    agent.epsilon = original_epsilon
    return final_arr, arr_hist_eval


# ══════════════════════════════════════════════════════════════════════════════
# 결과 저장 및 시각화
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    arr_hist:         list[float],
    col_hist:         list[float],
    stats:            dict,
    eval_a_hist:      list[float],
    eval_b_hist:      list[float],
    eval_a_final:     float,
    eval_b_final:     float,
) -> None:
    """V9 결과를 JSON과 PNG 두 장으로 저장한다.

    dqn_results_v9.json          : 학습 + 평가 통계
    dqn_comparison_v9.png        : V5-B/V6/V7/V8/V9 5선 학습 곡선
    dqn_v9_eval_conditions.png   : 조건 A(항상공격) vs 조건 B(50%공격) 평가 비교
    """
    # ── JSON 저장 ──────────────────────────────────────────────────────────
    stats["eval_a_final"]     = float(eval_a_final)
    stats["eval_b_final"]     = float(eval_b_final)
    stats["eval_a_hist"]      = eval_a_hist
    stats["eval_b_hist"]      = eval_b_hist
    stats["eval_episodes"]    = EVAL_EPISODES
    stats["eval_a_attack_prob"] = 1.0
    stats["eval_b_attack_prob"] = ATTACK_PROB

    out_json = RESULTS_DIR / "dqn_results_v9.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"v9": stats}, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장 -> {out_json}")

    # ── 기준선 로드 헬퍼 ─────────────────────────────────────────────────
    def _load_baseline(path: Path, key: str) -> tuple[list[float], float | None]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            d    = data[key]
            hist = d.get("arrival_history", [])
            fin  = d.get("final_arr")
            return hist, fin
        except Exception:
            return [], None

    v5b_hist, v5b_final = _load_baseline(RESULTS_DIR / "dqn_results_v5.json", "v5b")
    v6_hist,  v6_final  = _load_baseline(RESULTS_DIR / "dqn_results_v6.json", "v6")
    v7_hist,  v7_final  = _load_baseline(RESULTS_DIR / "dqn_results_v7.json", "v7")
    v8_hist,  v8_final  = _load_baseline(RESULTS_DIR / "dqn_results_v8.json", "v8")

    n = len(arr_hist)

    def _ep_val(hist: list[float], ep: int) -> float | None:
        return hist[ep - 1] if len(hist) >= ep else None

    # ── 비교 표 출력 ──────────────────────────────────────────────────────
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  V9 vs V8 vs V7 vs V6 vs V5-B 비교 ({'본 실행' if n >= 1000 else f'시험 {n}ep'})")
    print(sep)
    fmt5 = "{:<36} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print(fmt5.format("항목", "V5-B", "V6", "V7", "V8", "V9"))
    print("-" * 100)

    def _fv(v: float | None, fmt_s: str = "{:.3f}") -> str:
        return fmt_s.format(v) if v is not None else "N/A"

    rows = [
        ("ep100 도착률",
         _fv(_ep_val(v5b_hist,100)), _fv(_ep_val(v6_hist,100)),
         _fv(_ep_val(v7_hist,100)), _fv(_ep_val(v8_hist,100)), _fv(_ep_val(arr_hist,100))),
        ("ep200 도착률",
         _fv(_ep_val(v5b_hist,200)), _fv(_ep_val(v6_hist,200)),
         _fv(_ep_val(v7_hist,200)), _fv(_ep_val(v8_hist,200)), _fv(_ep_val(arr_hist,200))),
        ("최종 도착률 (1000ep)",
         _fv(v5b_final), _fv(v6_final), _fv(v7_final), _fv(v8_final),
         _fv(stats["final_arr"])),
        ("─" * 34,) + ("─" * 8,) * 5,
        ("총 스텝 수", "─","─","─","─", f"{stats['total_steps']:,}"),
        ("공격 적용률 (실제)", "─","─","─","─",
         f"{stats['attack_rate_actual']:.1%}"
         f"({stats['attacked_steps']:,}/{stats['total_steps']:,})"),
        ("DDM RE/DE/SE/폐기", "─","─","─","─",
         f"{stats['type_re']:,}/{stats['type_de']:,}"
         f"/{stats['type_se']:,}/{stats['discard_count']:,}"),
        ("DE: clean/survived", "─","─","─","─",
         f"{stats['de_from_clean']:,}/{stats['de_from_survived']:,}"),
        ("SE: clean/survived", "─","─","─","─",
         f"{stats['se_from_clean']:,}/{stats['se_from_survived']:,}"),
        ("버퍼 크기 (최종)", "─","─","─","─", f"{stats['buffer_size_final']:,}"),
        ("HighShift 생존율", "─","─","─","─",
         f"{stats['high_shift_survive_rate']:.1%}"
         f"({stats['high_shift_survived']:,}/{stats['high_shift_total']:,})"),
        ("─" * 34,) + ("─" * 8,) * 5,
        ("평가 조건A 도착률 (공격100%)", "─","─","─","─", f"{eval_a_final:.3f}"),
        ("평가 조건B 도착률 (공격50%)",  "─","─","─","─", f"{eval_b_final:.3f}"),
        ("V8 최종 도착률 (비교기준)", "─","─","─", _fv(v8_final), "─"),
    ]
    for r in rows:
        print(fmt5.format(*r))
    print(sep)

    # ══════════════════════════════════════════════════════════════════════
    # 그래프 1: 5선 학습 곡선 비교 (dqn_comparison_v9.png)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#f5f6fa")
    ep_label = f"{n}ep" if n < 1000 else "1000ep"
    fig.suptitle(
        f"DQN V5-B / V6 / V7 / V8 / V9(확률적공격 p=0.5)  [{ep_label}]\n"
        "8 고정 장애물 | PGD eps=0.03 | 100ep 이동평균 도착률",
        fontsize=11, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(range(1, n + 1), arr_hist,
            color="#c00000", lw=2.2, label=f"V9 (공격p={ATTACK_PROB}+연속CE생존)")
    if v8_hist:
        ax.plot(range(1, len(v8_hist) + 1), v8_hist,
                color="#7030a0", lw=1.8, ls="-", alpha=0.7, label="V8 (항상공격+연속CE생존)")
    elif v8_final is not None:
        ax.axhline(v8_final, color="#7030a0", ls="-", alpha=0.7,
                   label=f"V8 최종 {v8_final:.1%}")
    if v7_hist:
        ax.plot(range(1, len(v7_hist) + 1), v7_hist,
                color="#2e75b6", lw=1.3, ls="-.", label="V7 (CE이진컷오프)")
    elif v7_final is not None:
        ax.axhline(v7_final, color="#2e75b6", ls="-.",
                   label=f"V7 최종 {v7_final:.1%}")
    if v6_hist:
        ax.plot(range(1, len(v6_hist) + 1), v6_hist,
                color="#ed7d31", lw=1.1, ls="--", label="V6 (오염페널티)")
    elif v6_final is not None:
        ax.axhline(v6_final, color="#ed7d31", ls="--",
                   label=f"V6 최종 {v6_final:.1%}")
    if v5b_hist:
        ax.plot(range(1, len(v5b_hist) + 1), v5b_hist,
                color="#70ad47", lw=1.0, ls=":", label="V5-B (연속CE, 단일버퍼)")
    elif v5b_final is not None:
        ax.axhline(v5b_final, color="#70ad47", ls=":",
                   label=f"V5-B 최종 {v5b_final:.1%}")
    ax.set_xlabel("에피소드", fontsize=10)
    ax.set_ylabel("도착률 (최근 100ep 이동평균)", fontsize=10)
    ax.set_title("도착률 학습 곡선 비교 (V5-B~V9)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.set_facecolor("#f8f9fb")

    # 오른쪽: DE/SE 4분류 막대 + 폐기 수
    ax = axes[1]
    bar_labels = ["RE\n(결과)",
                  "DE\n clean",
                  "DE\nsurvived",
                  "SE\n clean",
                  "SE\nsurvived",
                  "폐기\n(discard)"]
    bar_vals   = [
        stats["type_re"],
        stats["de_from_clean"],
        stats["de_from_survived"],
        stats["se_from_clean"],
        stats["se_from_survived"],
        stats["discard_count"],
    ]
    bar_colors = ["#2e75b6", "#ffc000", "#ff9900",
                  "#70ad47", "#a9d18e", "#ff0000"]
    bars = ax.bar(bar_labels, bar_vals,
                  color=bar_colors, alpha=0.85, edgecolor="white", lw=0.8)
    max_v = max(bar_vals) if bar_vals else 1
    for bar, v in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max_v * 0.012,
                f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("분류 횟수 (전체 학습)", fontsize=10)
    ax.set_title("V9 경험 타입 4분류 분포\n(clean=shift≤10px, survived=shift>10px 생존)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#f8f9fb")

    plt.tight_layout()
    out_png1 = RESULTS_DIR / "dqn_comparison_v9.png"
    plt.savefig(out_png1, dpi=150)
    plt.close()
    print(f"  그래프 저장 -> {out_png1}")

    # ══════════════════════════════════════════════════════════════════════
    # 그래프 2: 조건 A/B 평가 비교 (dqn_v9_eval_conditions.png)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#f5f6fa")
    fig.suptitle(
        "V9 정책 평가: 조건 A(항상공격) vs 조건 B(공격50%) | epsilon=0.05 고정\n"
        f"V8 최종 도착률({v8_final:.1%}) 대비 비교 (단일 시드, 반복검증 미수행)",
        fontsize=11, fontweight="bold",
    )

    # 왼쪽: 평가 롤링 도착률 곡선 200ep
    ax = axes[0]
    n_a = len(eval_a_hist)
    n_b = len(eval_b_hist)
    ax.plot(range(1, n_a + 1), eval_a_hist,
            color="#c00000", lw=1.8, label=f"조건 A: 항상 공격 (최종={eval_a_final:.3f})")
    ax.plot(range(1, n_b + 1), eval_b_hist,
            color="#2e75b6", lw=1.8, ls="--",
            label=f"조건 B: 50% 공격 (최종={eval_b_final:.3f})")
    if v8_final is not None:
        ax.axhline(v8_final, color="#7030a0", lw=1.2, ls=":",
                   label=f"V8 학습 최종 {v8_final:.1%} (항상공격)")
    ax.set_xlabel("평가 에피소드", fontsize=10)
    ax.set_ylabel("도착률 (최근 100ep 이동평균)", fontsize=10)
    ax.set_title("평가 조건별 도착률 곡선", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.set_facecolor("#f8f9fb")

    # 오른쪽: 막대 비교
    ax = axes[1]
    bar_lbls = [f"V8 학습\n(항상공격)", f"V9 평가\n조건A(항상공격)", f"V9 평가\n조건B(50%공격)"]
    bar_vals_eval = [v8_final if v8_final is not None else 0.0, eval_a_final, eval_b_final]
    bar_clrs = ["#7030a0", "#c00000", "#2e75b6"]
    bars = ax.bar(bar_lbls, bar_vals_eval,
                  color=bar_clrs, alpha=0.85, edgecolor="white", lw=0.8)
    for bar, v in zip(bars, bar_vals_eval):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("도착률", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("V8 vs V9 평가 조건별 도착률 비교\n(단일 시드 기준)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#f8f9fb")

    plt.tight_layout()
    out_png2 = RESULTS_DIR / "dqn_v9_eval_conditions.png"
    plt.savefig(out_png2, dpi=150)
    plt.close()
    print(f"  그래프 저장 -> {out_png2}")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="DQN V9 학습")
    parser.add_argument("--episodes", type=int, default=EPISODES_V9,
                        help=f"학습 에피소드 수 (기본 {EPISODES_V9})")
    args = parser.parse_args()

    arr_hist, col_hist, stats, agent = train_v9(args.episodes)

    # ── 정책 평가: 조건 A (항상 공격) ─────────────────────────────────────
    eval_a_final, eval_a_hist = evaluate_policy(
        agent, attack_prob=1.0, n_eval=EVAL_EPISODES, label="조건A 항상공격"
    )

    # ── 정책 평가: 조건 B (50% 공격) ──────────────────────────────────────
    eval_b_final, eval_b_hist = evaluate_policy(
        agent, attack_prob=ATTACK_PROB, n_eval=EVAL_EPISODES, label="조건B 50%공격"
    )

    save_results(arr_hist, col_hist, stats,
                 eval_a_hist, eval_b_hist, eval_a_final, eval_b_final)
    print("\n완료.")


if __name__ == "__main__":
    main()
