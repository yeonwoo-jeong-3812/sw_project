"""DQN v7: ImprovedDDMBuffer + 경험 타입 분류를 실제 학습 루프에 연결.

v6(오염 페널티) 위에 타입별 리플레이 버퍼(ImprovedDDMBuffer)를 실제로 연결한다.

변경점 (v6 대비):
  1. DQNAgent.memory 대신 ImprovedDDMBuffer 사용
     RE:DE:SE=4:3:3 비율 샘플링, CE 자동 폐기
  2. _classify_by_shift()로 경험 타입 분류 (Faster R-CNN 없이 shift만 사용)
     우선순위: done=True → RE (CE보다 우선);
               shift > 10px → CE (폐기 대상);
               obs_dist < 3km → DE (위험 근접);
               else → SE
  3. buffer.add(Experience)로 저장 — CE는 discarded_ce 카운터만 증가
  4. DQNAgent.update() 미호출 — _ddm_update()로 직접 배치 학습
  5. 실제 배치 크기 < BATCH_SIZE x 0.5 이면 학습 스킵
  6. 추가 지표: RE/DE/SE/CE 분류 수, 평균 실제 배치 크기, 학습 스킵 횟수

실행 방법:
  시험 (200ep):  python src/dqn_comparison_v7.py --episodes 200
  본 실행:       python src/dqn_comparison_v7.py
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

EPISODES_V7 = 1000   # v5/v6와 동일 (직접 비교)

# v6와 동일한 보상 상수
TIME_PENALTY  = -0.01
DIST_APPROACH = +0.01
DIST_RECEDE   = -0.01

# v6와 동일한 오염 페널티
PENALTY_RISK         = -0.01
PENALTY_CRITICAL     = -0.03
OBS_DIST_RISK_THRESH =  3.0   # km (RISK 페널티 발동 거리 조건)

# 오염 판정 임계값
SHIFT_THRESH_RISK     = 10.0  # px
SHIFT_THRESH_CRITICAL = 20.0  # px

# DE 분류 위험 반경 (DANGER_RADIUS)
DANGER_RADIUS = 3.0   # km

# 배치 최소 비율: 실제 배치 < BATCH_SIZE × 0.5 → 학습 스킵
MIN_BATCH_RATIO = 0.5
MIN_BATCH       = int(BATCH_SIZE * MIN_BATCH_RATIO)   # = 16


# ══════════════════════════════════════════════════════════════════════════════
# 오염 판정 (v6와 동일)
# ══════════════════════════════════════════════════════════════════════════════

def _assess_shift(shift: float) -> CorruptionLevel:
    if shift <= SHIFT_THRESH_RISK:
        return CorruptionLevel.STABLE
    elif shift <= SHIFT_THRESH_CRITICAL:
        return CorruptionLevel.RISK
    else:
        return CorruptionLevel.CRITICAL


# ══════════════════════════════════════════════════════════════════════════════
# 경험 타입 분류 (v7 핵심 — Faster R-CNN 없이 shift+거리만 사용)
# ══════════════════════════════════════════════════════════════════════════════

def _classify_by_shift(
    shift: float,
    dist_to_obstacle: float,
    done: bool,
    danger_radius: float = DANGER_RADIUS,
) -> ExperienceType:
    """shift와 장애물 거리만으로 경험 타입을 분류한다.

    우선순위 (명시적 설계 결정):
      1. done=True      → RE  (목표 도달 / 충돌 경험; CE 체크보다 먼저)
      2. shift > 10px   → CE  (PGD 오염 구간; buffer에서 자동 폐기)
      3. obs_dist < 3km → DE  (장애물 근접 위험)
      4. 나머지         → SE

    done을 CE보다 먼저 체크하는 이유:
      목표 도달(+1.0) / 충돌(-1.0) 경험은 보상 신호가 강력하고 학습 필수적이므로
      shift가 높더라도 RE로 분류해 100% 보존한다.
    """
    # [확인 포인트 a] done=True → RE, CE보다 우선
    if done:
        return ExperienceType.RE
    # [확인 포인트 c] shift>10px → CE; buffer.add()가 discarded_ce 증가 후 폐기
    if shift > SHIFT_THRESH_RISK:
        return ExperienceType.CE
    if dist_to_obstacle < danger_radius:
        return ExperienceType.DE
    return ExperienceType.SE


# ══════════════════════════════════════════════════════════════════════════════
# 커스텀 DQN 업데이트 (ImprovedDDMBuffer 기반)
# ══════════════════════════════════════════════════════════════════════════════

def _ddm_update(
    agent: DQNAgent,
    buffer: ImprovedDDMBuffer,
) -> tuple[float | None, int]:
    """ImprovedDDMBuffer에서 배치를 샘플링해 DQN gradient 업데이트를 수행한다.

    DQNAgent.update()를 호출하지 않고 policy_net/target_net/optimizer를 직접 사용.
    (agent.memory는 완전히 무시)

    Returns:
        (loss, actual_batch_size):
          배치 크기 < MIN_BATCH 이면 학습 스킵 → (None, actual_size)
    """
    batch  = buffer.sample(BATCH_SIZE)
    actual = len(batch)

    # [확인 포인트 b] 초기 에피소드: 버퍼가 비어 실제 배치 < 16 → 스킵
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

def train_v7(n_episodes: int = EPISODES_V7) -> tuple[list[float], list[float], dict]:
    """v6 보상 구조 + ImprovedDDMBuffer + RE/DE/SE/CE 분류 학습.

    변경점 (v6 대비):
      - DQNAgent.memory 미사용; ImprovedDDMBuffer(capacity=10,000)로 대체
      - _classify_by_shift()로 경험 타입 분류
      - buffer.add(Experience); CE는 자동 폐기(discarded_ce 카운터)
      - _ddm_update()로 직접 배치 학습; DQNAgent.update() 미호출
      - 실제 배치 크기 < 16이면 학습 스킵
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

    # 거리 보상 실측 카운터
    nonterminal_steps = 0
    goal_approach_cnt = 0
    goal_recede_cnt   = 0
    goal_neutral_cnt  = 0

    valid_exp_hist: list[int] = []  # 에피소드별 버퍼 총 크기

    print(f"\n[V7] 학습 시작 ({n_episodes}ep  |  ImprovedDDMBuffer RE:DE:SE=4:3:3)")
    print(f"     RISK={PENALTY_RISK}(접근중+{OBS_DIST_RISK_THRESH}km이내)  "
          f"CRITICAL={PENALTY_CRITICAL}(무조건)")
    print(f"     분류: done=True->RE(CE보다 우선)  shift>10->CE->폐기  "
          f"obs<{DANGER_RADIUS}km->DE  else->SE")
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

            # ── 비종료 스텝 보상 블록 (v6와 동일) ─────────────────────────
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

                # v6와 동일한 오염 페널티
                if corruption is CorruptionLevel.RISK:
                    if (prev_obs_dist is not None
                            and obs_dist < prev_obs_dist
                            and obs_dist < OBS_DIST_RISK_THRESH):
                        reward += PENALTY_RISK
                        risk_count += 1
                elif corruption is CorruptionLevel.CRITICAL:
                    reward += PENALTY_CRITICAL
                    critical_count += 1

            # ── 경험 타입 분류 (v7 핵심) ────────────────────────────────────
            exp_type = _classify_by_shift(shift, obs_dist, done, DANGER_RADIUS)
            type_cnt[exp_type] += 1

            # ── c_next 계산 (모든 타입 공통; CE도 Experience 생성 후 폐기) ──
            D_obs_next, theta_obs_next = env.get_raw_obstacle()
            c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)

            exp = Experience(
                state      = c_state,
                action     = action,
                reward     = reward,
                next_state = c_next,
                done       = done,
                exp_type   = exp_type,
                corruption = corruption,
            )
            # CE: buffer.add()에서 discarded_ce += 1 후 즉시 False 반환
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
            risk_rate = risk_count     / total_steps if total_steps > 0 else 0.0
            crit_rate = critical_count / total_steps if total_steps > 0 else 0.0
            ce_rate   = type_cnt[ExperienceType.CE] / total_steps if total_steps > 0 else 0.0
            avg_batch = batch_size_sum / update_count if update_count > 0 else 0.0
            print(f"  Ep {ep+1:4d} | 도착률={arr_hist[-1]:.3f}"
                  f" | 충돌률={col_hist[-1]:.3f}"
                  f" | e={agent.epsilon:.3f}"
                  f" | RE={type_cnt[ExperienceType.RE]:,}"
                  f" DE={type_cnt[ExperienceType.DE]:,}"
                  f" SE={type_cnt[ExperienceType.SE]:,}"
                  f" CE={type_cnt[ExperienceType.CE]:,}({ce_rate:.1%})"
                  f" | RISK={risk_count:,}({risk_rate:.1%})"
                  f" CRIT={critical_count:,}({crit_rate:.1%})"
                  f" | skip={skipped_updates:,} avg_batch={avg_batch:.1f}")

    nt = nonterminal_steps or 1
    avg_dist_reward  = (goal_approach_cnt * DIST_APPROACH
                        + goal_recede_cnt  * DIST_RECEDE) / nt
    avg_step_penalty = TIME_PENALTY + avg_dist_reward
    avg_batch_final  = batch_size_sum / update_count if update_count > 0 else 0.0

    stats = {
        "label":   "V7 (ImprovedDDMBuffer + CE분류)",
        "episodes": n_episodes,
        "total_steps":   total_steps,
        "arrival_count": arrival_count,
        "final_arr":     float(arr_hist[-1]),
        "final_col":     float(col_hist[-1]),
        "mean_arr_last100": float(np.mean(arr_hist[-100:])),
        "peak_arr":      float(max(arr_hist)),
        "peak_ep":       int(np.argmax(arr_hist)) + 1,
        # DDM 분류 지표
        "type_re":         type_cnt[ExperienceType.RE],
        "type_de":         type_cnt[ExperienceType.DE],
        "type_se":         type_cnt[ExperienceType.SE],
        "type_ce":         type_cnt[ExperienceType.CE],
        "discarded_ce":    buffer.discarded_ce,
        "buffer_size_final": buffer.size,
        # 업데이트 지표
        "skipped_updates":  skipped_updates,
        "update_count":     update_count,
        "avg_batch_size":   float(avg_batch_final),
        "train_steps":      agent.train_steps,
        # 오염 페널티 (v6와 동일)
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

    torch.save(agent.policy_net.state_dict(), RESULTS_DIR / "v7_policy.pth")
    print(f"\n  [V7] 가중치 저장 -> results/v7_policy.pth")
    return arr_hist, col_hist, stats


# ══════════════════════════════════════════════════════════════════════════════
# 결과 저장 및 시각화
# ══════════════════════════════════════════════════════════════════════════════

def save_results(arr_hist: list[float], col_hist: list[float], stats: dict) -> None:
    """V7 결과를 JSON과 PNG로 저장한다.

    비교 대상:
      V5-B: results/dqn_results_v5.json → data["v5b"]["arrival_history"]
      V6:   results/dqn_results_v6.json → data["v6"]["arrival_history"]
    """
    # ── JSON 저장 ──────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "dqn_results_v7.json"
    wrapper  = {"v7": stats}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장 -> {out_json}")

    # ── V5-B 기준선 로드 ──────────────────────────────────────────────────
    v5b_hist: list[float] = []
    v5b_ep100 = v5b_ep200 = v5b_final = None
    try:
        with open(RESULTS_DIR / "dqn_results_v5.json", encoding="utf-8") as f:
            v5_data = json.load(f)
        v5b      = v5_data["v5b"]
        v5b_hist = v5b.get("arrival_history", [])
        v5b_ep100 = v5b_hist[99]  if len(v5b_hist) >= 100 else None
        v5b_ep200 = v5b_hist[199] if len(v5b_hist) >= 200 else None
        v5b_final = v5b.get("final_arr")
    except Exception:
        pass

    # ── V6 기준선 로드 ────────────────────────────────────────────────────
    v6_hist: list[float] = []
    v6_ep100 = v6_ep200 = v6_final = None
    try:
        with open(RESULTS_DIR / "dqn_results_v6.json", encoding="utf-8") as f:
            v6_data = json.load(f)
        v6       = v6_data["v6"]
        v6_hist  = v6.get("arrival_history", [])
        v6_ep100 = v6_hist[99]  if len(v6_hist) >= 100 else None
        v6_ep200 = v6_hist[199] if len(v6_hist) >= 200 else None
        v6_final = v6.get("final_arr")
    except Exception:
        pass

    n = len(arr_hist)
    v7_ep100 = arr_hist[99]  if n >= 100 else None
    v7_ep200 = arr_hist[199] if n >= 200 else None

    # ── 비교 표 출력 ──────────────────────────────────────────────────────
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  V7 vs V6 vs V5-B 비교 ({'본 실행' if n >= 1000 else f'시험 {n}ep'})")
    print(sep)
    fmt = "{:<34} {:>14} {:>14} {:>14}"
    print(fmt.format("항목", "V5-B", "V6", "V7"))
    print("-" * 80)

    def _fv(v: float | None, fmt_s: str = "{:.3f}") -> str:
        return fmt_s.format(v) if v is not None else "N/A"

    rows = [
        ("ep100 도착률",  _fv(v5b_ep100), _fv(v6_ep100), _fv(v7_ep100)),
        ("ep200 도착률",  _fv(v5b_ep200), _fv(v6_ep200), _fv(v7_ep200)),
        ("최종 도착률",   _fv(v5b_final), _fv(v6_final), _fv(stats["final_arr"])),
        ("─" * 32, "─" * 12, "─" * 12, "─" * 12),
        ("총 스텝 수", "─", "─", f"{stats['total_steps']:,}"),
        ("DDM RE / DE / SE / CE",
         "─", "─",
         f"{stats['type_re']:,} / {stats['type_de']:,}"
         f" / {stats['type_se']:,} / {stats['type_ce']:,}"),
        ("CE 폐기 (discarded_ce)", "─", "─", f"{stats['discarded_ce']:,}"),
        ("버퍼 크기 (최종)",       "─", "─", f"{stats['buffer_size_final']:,}"),
        ("업데이트 스킵 횟수",     "─", "─", f"{stats['skipped_updates']:,}"),
        ("평균 실제 배치 크기",    "─", "─", f"{stats['avg_batch_size']:.1f}"),
        ("─" * 32, "─" * 12, "─" * 12, "─" * 12),
        ("RISK 페널티 발동률",
         "─", "─",
         f"{stats['risk_penalty_count']/stats['total_steps']:.1%}"),
        ("CRITICAL 페널티 발동률",
         "─", "─",
         f"{stats['critical_penalty_count']/stats['total_steps']:.1%}"),
    ]
    for r in rows:
        print(fmt.format(*r))
    print(sep)

    # ── 그래프 (3선 도착률 + DDM 분류 막대) ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#f5f6fa")
    ep_label = f"{n}ep" if n < 1000 else "1000ep"
    fig.suptitle(
        f"DQN V5-B vs V6(오염페널티) vs V7(ImprovedDDMBuffer)  [{ep_label}]\n"
        "8 고정 장애물 | PGD eps=0.03 | 100ep 이동평균 도착률",
        fontsize=11, fontweight="bold",
    )

    # 왼쪽: 도착률 학습 곡선
    ax = axes[0]
    ax.plot(range(1, n + 1), arr_hist,
            color="#2e75b6", lw=1.8, label="V7 (DDMBuffer+CE분류)")
    if v6_hist:
        ax.plot(range(1, len(v6_hist) + 1), v6_hist,
                color="#ed7d31", lw=1.3, ls="--", label="V6 (오염페널티)")
    elif v6_final is not None:
        ax.axhline(v6_final, color="#ed7d31", ls="--",
                   label=f"V6 최종 {v6_final:.1%}")
    if v5b_hist:
        ax.plot(range(1, len(v5b_hist) + 1), v5b_hist,
                color="#70ad47", lw=1.0, ls=":", label="V5-B (연속CE)")
    elif v5b_final is not None:
        ax.axhline(v5b_final, color="#70ad47", ls=":",
                   label=f"V5-B 최종 {v5b_final:.1%}")
    ax.set_xlabel("에피소드", fontsize=10)
    ax.set_ylabel("도착률 (최근 100ep 이동평균)", fontsize=10)
    ax.set_title("도착률 학습 곡선 비교", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.set_facecolor("#f8f9fb")

    # 오른쪽: DDM 분류 분포 막대그래프
    ax = axes[1]
    type_labels = ["RE\n(결과)", "DE\n(위험)", "SE\n(안전)", "CE\n(폐기)"]
    type_vals   = [
        stats["type_re"],
        stats["type_de"],
        stats["type_se"],
        stats["type_ce"],
    ]
    bar_colors = ["#2e75b6", "#ffc000", "#70ad47", "#ff0000"]
    bars = ax.bar(type_labels, type_vals,
                  color=bar_colors, alpha=0.85, edgecolor="white", lw=0.8)
    max_v = max(type_vals) if type_vals else 1
    for bar, v in zip(bars, type_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max_v * 0.015,
                f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("분류 횟수 (전체 학습)", fontsize=10)
    ax.set_title("V7 경험 타입 분류 분포", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#f8f9fb")

    plt.tight_layout()
    out_png = RESULTS_DIR / "dqn_comparison_v7.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  그래프 저장 -> {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="DQN V7 학습")
    parser.add_argument("--episodes", type=int, default=EPISODES_V7,
                        help=f"학습 에피소드 수 (기본 {EPISODES_V7})")
    args = parser.parse_args()

    arr_hist, col_hist, stats = train_v7(args.episodes)
    save_results(arr_hist, col_hist, stats)
    print("\n완료.")


if __name__ == "__main__":
    main()
