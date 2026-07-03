"""이 파일은 V7의 ImprovedDDMBuffer 구조를 유지하면서, CE 판정을
이진 컷오프(shift>10px 무조건 폐기)에서 V5-B와 동일한 연속 가중치
방식으로 바꾼 버전이다. 목적은 타입별 버퍼링의 구조적 이점과
V5-B의 데이터 활용 효율을 동시에 확보하는 것이다.

변경점 (v7 대비):
  1. _classify_by_shift() 폐기 → 학습 루프 내 인라인 분류로 대체
  2. shift > 10px 구간: 무조건 폐기 → 연속 확률적 생존
       survive_prob = max(0.05, 1.0 - shift / 50.0)
       생존 시: obs_dist 기준 DE/SE 분류, DDM_DEPOSIT_RATIO 우회
       폐기 시: discarded_ce 카운터 증가
  3. shift <= 10px 구간: V7과 완전히 동일 (DE/SE 분류 + DDM_DEPOSIT_RATIO)
  4. done=True → RE 우선 처리: V7과 동일하게 유지
  5. 추가 지표: high_shift_total, high_shift_survived, high_shift_de, high_shift_se

실행 방법:
  시험 (200ep):  python src/dqn_comparison_v8.py --episodes 200
  본 실행:       python src/dqn_comparison_v8.py
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

EPISODES_V8 = 1000   # v5/v6/v7과 동일 (직접 비교)

# v6/v7과 동일한 보상 상수
TIME_PENALTY  = -0.01
DIST_APPROACH = +0.01
DIST_RECEDE   = -0.01

# v6/v7과 동일한 오염 페널티
PENALTY_RISK         = -0.01
PENALTY_CRITICAL     = -0.03
OBS_DIST_RISK_THRESH =  3.0   # km (RISK 페널티 발동 거리 조건)

# 오염 판정 임계값 (v7과 동일)
SHIFT_THRESH_RISK     = 10.0  # px
SHIFT_THRESH_CRITICAL = 20.0  # px

# DE 분류 위험 반경 (v7과 동일)
DANGER_RADIUS = 3.0   # km

# V5-B 연속 가중치 파라미터
SURVIVE_SHIFT_MAX = 50.0   # shift 이 값에서 survive_prob → 0.05

# 배치 최소 비율: 실제 배치 < BATCH_SIZE × 0.5 → 학습 스킵
MIN_BATCH_RATIO = 0.5
MIN_BATCH       = int(BATCH_SIZE * MIN_BATCH_RATIO)   # = 16


# ══════════════════════════════════════════════════════════════════════════════
# 오염 판정 (v7과 동일)
# ══════════════════════════════════════════════════════════════════════════════

def _assess_shift(shift: float) -> CorruptionLevel:
    if shift <= SHIFT_THRESH_RISK:
        return CorruptionLevel.STABLE
    elif shift <= SHIFT_THRESH_CRITICAL:
        return CorruptionLevel.RISK
    else:
        return CorruptionLevel.CRITICAL


# ══════════════════════════════════════════════════════════════════════════════
# 커스텀 DQN 업데이트 (v7과 동일)
# ══════════════════════════════════════════════════════════════════════════════

def _ddm_update(
    agent: DQNAgent,
    buffer: ImprovedDDMBuffer,
) -> tuple[float | None, int]:
    """ImprovedDDMBuffer에서 배치를 샘플링해 DQN gradient 업데이트를 수행한다.

    Returns:
        (loss, actual_batch_size):
          배치 크기 < MIN_BATCH 이면 학습 스킵 → (None, actual_size)
    """
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

def train_v8(n_episodes: int = EPISODES_V8) -> tuple[list[float], list[float], dict]:
    """V7 구조 유지 + CE 이진 컷오프 → 연속 가중치 확률적 생존으로 교체.

    [확인 포인트 a] shift <= 10px 구간 처리:
      - V7과 완전히 동일: obs_dist 기준 DE/SE 분류, buffer.add() 통해
        DDM_DEPOSIT_RATIO(DE=0.8, SE=0.2) 정상 적용

    [확인 포인트 b] RE 저장 경험 수:
      - done=True 스텝은 무조건 RE로 분류되므로 RE 수 = 에피소드 수
        (단, MAX_STEPS 내 done=True 없이 종료된 에피소드는 예외)

    [확인 포인트 c] survive_prob 통과 시 DDM_DEPOSIT_RATIO 미적용:
      - 생존한 고-shift 경험은 buffer._buffers[exp_type].append()로
        직접 추가해 DE/SE의 0.8/0.2 필터를 우회한다.
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

    # 타입별 분류 카운터 (전체 스텝 기준)
    type_cnt: dict[ExperienceType, int] = {t: 0 for t in ExperienceType}

    # V8 추가: 고-shift 구간 통계 (shift > 10px, done=False)
    high_shift_total    = 0   # shift > 10px 경험 총 수
    high_shift_survived = 0   # survive_prob 통과한 수
    high_shift_de       = 0   # 생존 후 DE로 분류된 수
    high_shift_se       = 0   # 생존 후 SE로 분류된 수

    # 거리 보상 실측 카운터
    nonterminal_steps = 0
    goal_approach_cnt = 0
    goal_recede_cnt   = 0
    goal_neutral_cnt  = 0

    valid_exp_hist: list[int] = []  # 에피소드별 버퍼 총 크기

    print(f"\n[V8] 학습 시작 ({n_episodes}ep  |  ImprovedDDMBuffer + 연속 가중치 생존)")
    print(f"     RISK={PENALTY_RISK}(접근중+{OBS_DIST_RISK_THRESH}km이내)  "
          f"CRITICAL={PENALTY_CRITICAL}(무조건)")
    print(f"     분류: done=True->RE(1순위)  shift>10->survive_prob->DE/SE or 폐기  "
          f"shift<=10->DE/SE(DDM비율적용)")
    print(f"     survive_prob = max(0.05, 1.0 - shift/{SURVIVE_SHIFT_MAX:.0f})")
    print(f"     학습 스킵 조건: 실제 배치 < {MIN_BATCH}")

    for ep in range(n_episodes):
        state   = env.reset()
        arrived = False
        hit     = False
        prev_obs_dist: float | None = None

        for _ in range(MAX_STEPS):
            prev_dist = math.dist(env.pos, env.goal)

            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)

            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)
            total_steps += 1

            obs_dist   = info["obs_dist"]
            corruption = _assess_shift(shift)

            # ── 비종료 스텝 보상 블록 (v6/v7과 동일) ──────────────────────
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

                # v6/v7과 동일한 오염 페널티
                if corruption is CorruptionLevel.RISK:
                    if (prev_obs_dist is not None
                            and obs_dist < prev_obs_dist
                            and obs_dist < OBS_DIST_RISK_THRESH):
                        reward += PENALTY_RISK
                        risk_count += 1
                elif corruption is CorruptionLevel.CRITICAL:
                    reward += PENALTY_CRITICAL
                    critical_count += 1

            # ── V8 핵심: 경험 분류 및 저장 ──────────────────────────────────
            # [확인 포인트 a, b, c] 참고:
            #   1순위: done=True → RE (100% 저장, V7과 동일)
            #   2순위: shift>10px → 연속 가중치 생존 판정 (V8 변경 핵심)
            #   3순위: shift<=10px → DE/SE + DDM_DEPOSIT_RATIO (V7과 동일)

            if done:
                # [확인 포인트 b] RE: done=True이면 무조건 저장
                # DDM_DEPOSIT_RATIO[RE] = 1.0이므로 buffer.add()는 항상 True 반환
                exp_type = ExperienceType.RE
                type_cnt[ExperienceType.RE] += 1
                D_obs_next, theta_obs_next = env.get_raw_obstacle()
                c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
                exp = Experience(
                    state=c_state, action=action, reward=reward,
                    next_state=c_next, done=done,
                    exp_type=exp_type, corruption=corruption,
                )
                buffer.add(exp)

            elif shift > SHIFT_THRESH_RISK:
                # [확인 포인트 c] V5-B 연속 가중치 생존
                # shift=10 → prob=0.80, shift=30 → prob=0.40, shift=50 → prob=0.05
                high_shift_total += 1
                survive_prob = max(0.05, 1.0 - shift / SURVIVE_SHIFT_MAX)

                if random.random() < survive_prob:
                    # 생존: obs_dist 기준으로 DE 또는 SE 분류
                    # [확인 포인트 c] DDM_DEPOSIT_RATIO 우회 — buffer._buffers[]에 직접 추가
                    exp_type = (ExperienceType.DE if obs_dist < DANGER_RADIUS
                                else ExperienceType.SE)
                    high_shift_survived += 1
                    if exp_type is ExperienceType.DE:
                        high_shift_de += 1
                    else:
                        high_shift_se += 1
                    type_cnt[exp_type] += 1

                    D_obs_next, theta_obs_next = env.get_raw_obstacle()
                    c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
                    exp = Experience(
                        state=c_state, action=action, reward=reward,
                        next_state=c_next, done=done,
                        exp_type=exp_type, corruption=corruption,
                    )
                    # DDM_DEPOSIT_RATIO 우회: deque에 직접 추가
                    buffer._buffers[exp_type].append(exp)
                    buffer._stats[exp_type] += 1
                else:
                    # 폐기: V7의 CE 폐기와 동일하게 discarded_ce 카운터 증가
                    type_cnt[ExperienceType.CE] += 1
                    buffer.discarded_ce += 1

            else:
                # [확인 포인트 a] shift<=10px: V7과 완전히 동일
                # buffer.add()가 DDM_DEPOSIT_RATIO(DE=0.8, SE=0.2)를 그대로 적용
                exp_type = (ExperienceType.DE if obs_dist < DANGER_RADIUS
                            else ExperienceType.SE)
                type_cnt[exp_type] += 1
                D_obs_next, theta_obs_next = env.get_raw_obstacle()
                c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
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

            # ── 다음 스텝 준비 ───────────────────────────────────────────────
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
            risk_rate    = risk_count              / total_steps     if total_steps     > 0 else 0.0
            crit_rate    = critical_count          / total_steps     if total_steps     > 0 else 0.0
            ce_rate      = type_cnt[ExperienceType.CE] / total_steps if total_steps     > 0 else 0.0
            survive_rate = high_shift_survived     / high_shift_total if high_shift_total > 0 else 0.0
            avg_batch    = batch_size_sum          / update_count    if update_count    > 0 else 0.0
            print(f"  Ep {ep+1:4d} | 도착률={arr_hist[-1]:.3f}"
                  f" | 충돌률={col_hist[-1]:.3f}"
                  f" | e={agent.epsilon:.3f}"
                  f" | RE={type_cnt[ExperienceType.RE]:,}"
                  f" DE={type_cnt[ExperienceType.DE]:,}"
                  f" SE={type_cnt[ExperienceType.SE]:,}"
                  f" CE폐기={type_cnt[ExperienceType.CE]:,}({ce_rate:.1%})"
                  f" | HighShift생존율={survive_rate:.1%}({high_shift_survived:,}/{high_shift_total:,})"
                  f" | RISK={risk_count:,}({risk_rate:.1%})"
                  f" CRIT={critical_count:,}({crit_rate:.1%})"
                  f" | skip={skipped_updates:,} avg_batch={avg_batch:.1f}"
                  f" | 버퍼={buffer.size:,}")

    nt = nonterminal_steps or 1
    avg_dist_reward  = (goal_approach_cnt * DIST_APPROACH
                        + goal_recede_cnt  * DIST_RECEDE) / nt
    avg_step_penalty = TIME_PENALTY + avg_dist_reward
    avg_batch_final  = batch_size_sum / update_count if update_count > 0 else 0.0
    survive_rate_final = (high_shift_survived / high_shift_total
                          if high_shift_total > 0 else 0.0)

    stats = {
        "label":   "V8 (ImprovedDDMBuffer + 연속CE생존)",
        "episodes": n_episodes,
        "total_steps":   total_steps,
        "arrival_count": arrival_count,
        "final_arr":     float(arr_hist[-1]),
        "final_col":     float(col_hist[-1]),
        "mean_arr_last100": float(np.mean(arr_hist[-100:])),
        "peak_arr":      float(max(arr_hist)),
        "peak_ep":       int(np.argmax(arr_hist)) + 1,
        # DDM 분류 지표 (V7과 동일한 키명 유지 → 직접 비교 가능)
        "type_re":         type_cnt[ExperienceType.RE],
        "type_de":         type_cnt[ExperienceType.DE],
        "type_se":         type_cnt[ExperienceType.SE],
        "type_ce":         type_cnt[ExperienceType.CE],   # 폐기된 수 (V7의 discarded_ce에 해당)
        "discarded_ce":    buffer.discarded_ce,            # type_ce와 동일해야 함
        "buffer_size_final": buffer.size,
        # V8 추가: 고-shift 구간 상세 통계
        "high_shift_total":    high_shift_total,
        "high_shift_survived": high_shift_survived,
        "high_shift_discarded": high_shift_total - high_shift_survived,
        "high_shift_survive_rate": float(survive_rate_final),
        "high_shift_de":       high_shift_de,
        "high_shift_se":       high_shift_se,
        # 업데이트 지표
        "skipped_updates":  skipped_updates,
        "update_count":     update_count,
        "avg_batch_size":   float(avg_batch_final),
        "train_steps":      agent.train_steps,
        # 오염 페널티 (v6/v7과 동일)
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
        "arrival_history":    arr_hist,
        "collision_history":  col_hist,
        "valid_exp_hist":     valid_exp_hist,
    }

    torch.save(agent.policy_net.state_dict(), RESULTS_DIR / "v8_policy.pth")
    print(f"\n  [V8] 가중치 저장 -> results/v8_policy.pth")
    return arr_hist, col_hist, stats


# ══════════════════════════════════════════════════════════════════════════════
# 결과 저장 및 시각화
# ══════════════════════════════════════════════════════════════════════════════

def save_results(arr_hist: list[float], col_hist: list[float], stats: dict) -> None:
    """V8 결과를 JSON과 PNG로 저장한다.

    비교 대상:
      V5-B: results/dqn_results_v5.json  → data["v5b"]["arrival_history"]
      V6:   results/dqn_results_v6.json  → data["v6"]["arrival_history"]
      V7:   results/dqn_results_v7.json  → data["v7"]["arrival_history"]
    """
    # ── JSON 저장 ──────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "dqn_results_v8.json"
    wrapper  = {"v8": stats}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)
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

    n = len(arr_hist)

    def _ep_val(hist: list[float], ep: int) -> float | None:
        return hist[ep - 1] if len(hist) >= ep else None

    v5b_ep100, v5b_ep200 = _ep_val(v5b_hist, 100), _ep_val(v5b_hist, 200)
    v6_ep100,  v6_ep200  = _ep_val(v6_hist,  100), _ep_val(v6_hist,  200)
    v7_ep100,  v7_ep200  = _ep_val(v7_hist,  100), _ep_val(v7_hist,  200)
    v8_ep100,  v8_ep200  = _ep_val(arr_hist,  100), _ep_val(arr_hist,  200)

    # ── 비교 표 출력 ──────────────────────────────────────────────────────
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  V8 vs V7 vs V6 vs V5-B 비교 ({'본 실행' if n >= 1000 else f'시험 {n}ep'})")
    print(sep)
    fmt = "{:<34} {:>12} {:>12} {:>12} {:>12}"
    print(fmt.format("항목", "V5-B", "V6", "V7", "V8"))
    print("-" * 90)

    def _fv(v: float | None, fmt_s: str = "{:.3f}") -> str:
        return fmt_s.format(v) if v is not None else "N/A"

    rows = [
        ("ep100 도착률",
         _fv(v5b_ep100), _fv(v6_ep100), _fv(v7_ep100), _fv(v8_ep100)),
        ("ep200 도착률",
         _fv(v5b_ep200), _fv(v6_ep200), _fv(v7_ep200), _fv(v8_ep200)),
        ("최종 도착률",
         _fv(v5b_final), _fv(v6_final), _fv(v7_final), _fv(stats["final_arr"])),
        ("─" * 32, "─" * 10, "─" * 10, "─" * 10, "─" * 10),
        ("총 스텝 수",
         "─", "─", "─", f"{stats['total_steps']:,}"),
        ("DDM RE / DE / SE / CE폐기",
         "─", "─", "─",
         f"{stats['type_re']:,}/{stats['type_de']:,}"
         f"/{stats['type_se']:,}/{stats['type_ce']:,}"),
        ("버퍼 크기 (최종)",
         "─", "─", "─", f"{stats['buffer_size_final']:,}"),
        ("─" * 32, "─" * 10, "─" * 10, "─" * 10, "─" * 10),
        ("HighShift 총 수 (shift>10px)",
         "─", "─", "─", f"{stats['high_shift_total']:,}"),
        ("HighShift 생존율",
         "─", "─", "─",
         f"{stats['high_shift_survive_rate']:.1%}"
         f"({stats['high_shift_survived']:,}/{stats['high_shift_total']:,})"),
        ("HighShift 생존→DE / →SE",
         "─", "─", "─",
         f"{stats['high_shift_de']:,} / {stats['high_shift_se']:,}"),
        ("HighShift 폐기 (V7 CE 비교용)",
         "─", "─", f"{stats.get('type_ce', '─')}",
         f"{stats['high_shift_discarded']:,}"),
        ("─" * 32, "─" * 10, "─" * 10, "─" * 10, "─" * 10),
        ("업데이트 스킵 횟수",
         "─", "─", "─", f"{stats['skipped_updates']:,}"),
        ("평균 실제 배치 크기",
         "─", "─", "─", f"{stats['avg_batch_size']:.1f}"),
        ("RISK 페널티 발동률",
         "─", "─", "─",
         f"{stats['risk_penalty_count']/stats['total_steps']:.1%}"),
        ("CRITICAL 페널티 발동률",
         "─", "─", "─",
         f"{stats['critical_penalty_count']/stats['total_steps']:.1%}"),
    ]

    # V7 CE폐기 수 보강 (JSON 로드)
    try:
        with open(RESULTS_DIR / "dqn_results_v7.json", encoding="utf-8") as f:
            v7_data = json.load(f)
        v7_ce = v7_data["v7"].get("type_ce", "─")
        rows[11] = (
            "HighShift 폐기 (V7 CE 비교용)",
            "─", "─",
            f"{v7_ce:,}" if isinstance(v7_ce, int) else str(v7_ce),
            f"{stats['high_shift_discarded']:,}",
        )
    except Exception:
        pass

    for r in rows:
        print(fmt.format(*r))
    print(sep)

    # ── 그래프 ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#f5f6fa")
    ep_label = f"{n}ep" if n < 1000 else "1000ep"
    fig.suptitle(
        f"DQN V5-B vs V6 vs V7 vs V8(연속CE생존)  [{ep_label}]\n"
        "8 고정 장애물 | PGD eps=0.03 | 100ep 이동평균 도착률",
        fontsize=11, fontweight="bold",
    )

    # 왼쪽: 4선 도착률 학습 곡선
    ax = axes[0]
    ax.plot(range(1, n + 1), arr_hist,
            color="#7030a0", lw=2.0, label="V8 (연속CE생존+DDMBuffer)")
    if v7_hist:
        ax.plot(range(1, len(v7_hist) + 1), v7_hist,
                color="#2e75b6", lw=1.5, ls="-.", label="V7 (CE이진컷오프)")
    elif v7_final is not None:
        ax.axhline(v7_final, color="#2e75b6", ls="-.",
                   label=f"V7 최종 {v7_final:.1%}")
    if v6_hist:
        ax.plot(range(1, len(v6_hist) + 1), v6_hist,
                color="#ed7d31", lw=1.3, ls="--", label="V6 (오염페널티)")
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
    ax.set_title("도착률 학습 곡선 비교 (V5-B / V6 / V7 / V8)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.set_facecolor("#f8f9fb")

    # 오른쪽: V8 경험 분포 + 고-shift 생존 분석
    ax = axes[1]

    # 스택 바: 전체 타입 분포
    type_labels  = ["RE\n(결과)", "DE\n(위험)", "SE\n(안전)"]
    normal_de    = stats["type_de"] - stats["high_shift_de"]
    normal_se    = stats["type_se"] - stats["high_shift_se"]
    normal_vals  = [stats["type_re"], normal_de,            normal_se]
    hs_vals      = [0,                stats["high_shift_de"], stats["high_shift_se"]]

    x = range(len(type_labels))
    bars1 = ax.bar(x, normal_vals,
                   color=["#2e75b6", "#ffc000", "#70ad47"],
                   alpha=0.85, edgecolor="white", lw=0.8,
                   label="shift≤10px (정상)")
    bars2 = ax.bar(x, hs_vals, bottom=normal_vals,
                   color=["#7030a0", "#ff9900", "#a9d18e"],
                   alpha=0.85, edgecolor="white", lw=0.8, hatch="//",
                   label="shift>10px 생존")

    # CE 폐기 수 별도 표시
    ce_bar = ax.bar(["CE\n(폐기)"], [stats["type_ce"]],
                    color="#ff0000", alpha=0.85, edgecolor="white", lw=0.8,
                    label="shift>10px 폐기")

    all_bars_x   = list(x) + [len(type_labels)]
    all_bars_val = [nv + hv for nv, hv in zip(normal_vals, hs_vals)] + [stats["type_ce"]]
    max_v = max(all_bars_val) if all_bars_val else 1
    for xi, vi in zip(all_bars_x, all_bars_val):
        ax.text(xi, vi + max_v * 0.015,
                f"{vi:,}", ha="center", va="bottom", fontsize=8)

    # 생존율 텍스트 주석
    sr = stats["high_shift_survive_rate"]
    ax.text(0.97, 0.97,
            f"shift>10px 생존율: {sr:.1%}\n"
            f"(생존 {stats['high_shift_survived']:,} / 총 {stats['high_shift_total']:,})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xticks(list(x) + [len(type_labels)])
    ax.set_xticklabels(type_labels + ["CE\n(폐기)"])
    ax.set_ylabel("분류 횟수 (전체 학습)", fontsize=10)
    ax.set_title("V8 경험 타입 분류 분포\n(빗금=shift>10px 생존분)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#f8f9fb")

    plt.tight_layout()
    out_png = RESULTS_DIR / "dqn_comparison_v8.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  그래프 저장 -> {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="DQN V8 학습")
    parser.add_argument("--episodes", type=int, default=EPISODES_V8,
                        help=f"학습 에피소드 수 (기본 {EPISODES_V8})")
    args = parser.parse_args()

    arr_hist, col_hist, stats = train_v8(args.episodes)
    save_results(arr_hist, col_hist, stats)
    print("\n완료.")


if __name__ == "__main__":
    main()
