"""DQN v6: improved_ddm.py 오염 판정·보상 페널티 실제 학습 루프 연결.

v5-B(연속 CE 가중치) 위에 shift 기반 오염 심각도 페널티를 추가한다.

변경점 (v5-B 대비):
  1. 오염 판정 _assess_shift(): shift → STABLE/RISK/CRITICAL
  2. 비종료 스텝 보상에 오염 페널티 추가:
       RISK     → -0.02  (장애물에 접근 중일 때만)
       CRITICAL → -0.05  (무조건)
  3. prev_obs_dist 추적 신규 추가

페널티 크기 선택 근거:
  v3/v4에서 RISK=-0.5가 밀집 구간 "−0.5×N >> +1.0"으로 학습 붕괴를 일으킨
  이력이 있어, v5 보상 스케일(시간 패널티 -0.01, 도착 +1.0)에 맞게 대폭 축소했다.
  RISK=-0.02, CRITICAL=-0.05로 시작해 붕괴 없이 효과를 검증한다.

미구현 (다음 단계):
  ImprovedDDMBuffer 및 RE/DE/SE/CE 타입 분류 리플레이는 아직 미연결.
  이 파일은 improved_ddm.py의 오염 판정/보상 로직을 학습 루프에 연결한
  중간 단계이며, 타입별 리플레이 버퍼는 v7 이후 과제로 남겨둔다.

실행 방법:
  시험 (200ep):  python src/dqn_comparison_v6.py --episodes 200
  본 실행 (full): python src/dqn_comparison_v6.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from enum import Enum, auto
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch

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
from dqn_comparison import DQNAgent, pgd_attack, MAX_STEPS
from env_test import DroneEnvN, OBSTACLES_8

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

EPISODES_V6 = 1000   # v5와 동일 (직접 비교 목적)

# v5와 동일한 보상 항목 (수정 금지)
CE_SHIFT_MAX  = 50.0
TIME_PENALTY  = -0.01
DIST_APPROACH = +0.01
DIST_RECEDE   = -0.01

# ── 오염 판정 임계값 (improved_ddm.py의 _SHIFT_THRESHOLD_* 와 동일) ─────────
SHIFT_THRESH_RISK     = 10.0   # px
SHIFT_THRESH_CRITICAL = 20.0   # px

# ── 오염 페널티 (v3의 -0.5/-1.0에서 대폭 축소, 2차 조정) ───────────────────
#    1차 시험(200ep): RISK=-0.02가 36.6%/step 발동 → 총 페널티 부담 과다
#    2차 조정: 크기 추가 축소 + RISK에 obs_dist < 3km 근접 조건 추가
PENALTY_RISK         = -0.01   # RISK: 접근중 + 3km 이내 + RISK 구간 shift
PENALTY_CRITICAL     = -0.03   # CRITICAL: 무조건 (단, shift > 20px 구간만)
OBS_DIST_RISK_THRESH =  3.0    # km: RISK 페널티 발동 최대 거리

# ── improved_ddm.py의 CorruptionLevel 재사용 (수정 없이 import) ─────────────
from improved_ddm import CorruptionLevel


# ══════════════════════════════════════════════════════════════════════════════
# 오염 판정 함수
# ══════════════════════════════════════════════════════════════════════════════

def _assess_shift(shift: float) -> CorruptionLevel:
    """PGD shift(px) 기반 3단계 오염 판정.

    improved_ddm.py의 _SHIFT_THRESHOLD_RISK=10.0, _SHIFT_THRESHOLD_CRITICAL=20.0과 일치.
    score 기반 판정은 실제 Faster R-CNN 없이는 사용 불가이므로 shift만 사용.
    """
    if shift <= SHIFT_THRESH_RISK:
        return CorruptionLevel.STABLE
    elif shift <= SHIFT_THRESH_CRITICAL:
        return CorruptionLevel.RISK
    else:
        return CorruptionLevel.CRITICAL


# ══════════════════════════════════════════════════════════════════════════════
# CE deposit 함수 (v5-B와 동일)
# ══════════════════════════════════════════════════════════════════════════════

def _deposit_continuous(shift: float) -> float:
    """연속 가중치 예치 비율 — v5-B와 동일."""
    return max(0.05, 1.0 - shift / CE_SHIFT_MAX)


# ══════════════════════════════════════════════════════════════════════════════
# 학습 루틴
# ══════════════════════════════════════════════════════════════════════════════

def train_v6(n_episodes: int = EPISODES_V6) -> tuple[list[float], list[float], dict]:
    """v5-B + 오염 판정·보상 페널티를 실제로 연결한 학습 루프.

    변경점 요약 (v5-B 대비):
      - prev_obs_dist 추적 신규 추가 (장애물 접근 판정용)
      - 비종료 스텝 보상 블록에 corruption penalty 추가
          RISK + 접근중 + obs_dist < 3km → -0.01
          CRITICAL (shift > 20px)        → -0.03
      - 목표 접근/후퇴 카운트 추가 (거리 보상 실측용)
      - 경험 예치는 v5-B와 동일 (_deposit_continuous, probabilistic)
    """
    env   = DroneEnvN(OBSTACLES_8, fallback_pos=[1.0, 1.0])
    agent = DQNAgent()

    arr_hist:   list[float] = []
    col_hist:   list[float] = []
    window_arr  = deque(maxlen=100)
    window_col  = deque(maxlen=100)

    total_steps    = 0
    ce_blocked     = 0
    valid_exp      = 0
    arrival_count  = 0
    ratio_sum      = 0.0
    risk_count     = 0   # RISK 페널티 실제 발동 횟수
    critical_count = 0   # CRITICAL 페널티 실제 발동 횟수
    # ── 거리 보상 실측 카운터 ──────────────────────────────────────────────────
    nonterminal_steps  = 0   # 비종료 스텝 총 수
    goal_approach_cnt  = 0   # curr_dist < prev_dist (DIST_APPROACH 발동)
    goal_recede_cnt    = 0   # curr_dist > prev_dist (DIST_RECEDE 발동)
    goal_neutral_cnt   = 0   # 동일 거리
    valid_exp_hist: list[int] = []

    print(f"\n[V6] 학습 시작 (8장애물/랜덤, {n_episodes}ep)")
    print(f"     RISK={PENALTY_RISK}(접근중+{OBS_DIST_RISK_THRESH}km이내), "
          f"CRITICAL={PENALTY_CRITICAL}(무조건)")
    print(f"     CE 예치: 연속 가중치 max(0.05, 1-shift/50), probabilistic")

    for ep in range(n_episodes):
        state = env.reset()
        arrived = False
        hit     = False
        prev_obs_dist: float | None = None   # 에피소드 시작 시 이전 값 없음

        for _ in range(MAX_STEPS):
            # ── 스텝 전 목적지 거리 기록 (v5와 동일) ─────────────────────
            prev_dist = math.dist(env.pos, env.goal)

            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)

            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)
            total_steps += 1

            obs_dist = info["obs_dist"]  # 스텝 후 장애물 거리

            # ── 비종료 스텝 보상 블록 ──────────────────────────────────────
            if not (info["goal_reached"] or info["collision"] or info["out_of_bounds"]):
                nonterminal_steps += 1

                # [v5-B 공통] 시간 페널티 + 목표 거리 보상
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

                # [v6 신규] 오염 심각도 기반 추가 페널티
                corruption = _assess_shift(shift)

                if corruption is CorruptionLevel.RISK:
                    # RISK: 장애물 접근 중 + 3km 이내 + 비종료일 때만
                    # (첫 스텝은 prev_obs_dist=None 이므로 자동 건너뜀)
                    if (prev_obs_dist is not None
                            and obs_dist < prev_obs_dist
                            and obs_dist < OBS_DIST_RISK_THRESH):
                        reward += PENALTY_RISK
                        risk_count += 1

                elif corruption is CorruptionLevel.CRITICAL:
                    # CRITICAL: 접근 여부·거리 무관, 무조건 억제
                    reward += PENALTY_CRITICAL
                    critical_count += 1

            # ── CE deposit (v5-B와 동일 — probabilistic 연속 가중치) ──────
            ratio = _deposit_continuous(shift)
            ratio_sum += ratio
            if random.random() < ratio:
                D_obs_next, theta_obs_next = env.get_raw_obstacle()
                c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
                agent.push(c_state, action, reward, c_next, done)
                valid_exp += 1
            else:
                ce_blocked += 1

            agent.update()

            # ── 다음 스텝 준비 ────────────────────────────────────────────
            prev_obs_dist = obs_dist   # 장애물 거리 갱신
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
        valid_exp_hist.append(valid_exp)

        if (ep + 1) % 100 == 0:
            discard_rate  = ce_blocked / total_steps if total_steps > 0 else 0.0
            risk_rate     = risk_count / total_steps if total_steps > 0 else 0.0
            crit_rate     = critical_count / total_steps if total_steps > 0 else 0.0
            print(f"  Ep {ep+1:4d} | 도착률={arr_hist[-1]:.3f}"
                  f" | 충돌률={col_hist[-1]:.3f}"
                  f" | ε={agent.epsilon:.3f}"
                  f" | CE폐기={ce_blocked}({discard_rate:.1%})"
                  f" | 유효경험={valid_exp:,}"
                  f" | RISK={risk_count:,}({risk_rate:.1%})"
                  f" CRIT={critical_count:,}({crit_rate:.1%})")

    avg_ratio    = ratio_sum / total_steps if total_steps > 0 else 0.0
    discard_rate = ce_blocked / total_steps if total_steps > 0 else 0.0

    # ── 거리 보상 실측 계산 ───────────────────────────────────────────────────
    nt = nonterminal_steps or 1
    avg_dist_reward  = (goal_approach_cnt * DIST_APPROACH
                        + goal_recede_cnt  * DIST_RECEDE) / nt
    avg_step_penalty = TIME_PENALTY + avg_dist_reward   # 시간+거리 합산 평균

    stats = {
        "label":              "V6 (연속CE + 오염페널티)",
        "episodes":           n_episodes,
        "total_steps":        total_steps,
        "ce_blocked":         ce_blocked,
        "valid_exp":          valid_exp,
        "arrival_count":      arrival_count,
        "final_arr":          float(arr_hist[-1]),
        "final_col":          float(col_hist[-1]),
        "mean_arr_last100":   float(np.mean(arr_hist[-100:])),
        "peak_arr":           float(max(arr_hist)),
        "peak_ep":            int(np.argmax(arr_hist)) + 1,
        "discard_rate":       discard_rate,
        "avg_deposit_ratio":  float(avg_ratio),
        "valid_exp_hist":     valid_exp_hist,
        # ── V6 신규 지표 ───────────────────────────────────────────────────
        "risk_penalty_count":     risk_count,
        "critical_penalty_count": critical_count,
        "penalty_risk":           PENALTY_RISK,
        "penalty_critical":       PENALTY_CRITICAL,
        "obs_dist_risk_thresh":   OBS_DIST_RISK_THRESH,
        "shift_thresh_risk":      SHIFT_THRESH_RISK,
        "shift_thresh_critical":  SHIFT_THRESH_CRITICAL,
        # ── 거리 보상 실측 ─────────────────────────────────────────────────
        "nonterminal_steps":   nonterminal_steps,
        "goal_approach_cnt":   goal_approach_cnt,
        "goal_recede_cnt":     goal_recede_cnt,
        "goal_neutral_cnt":    goal_neutral_cnt,
        "goal_approach_rate":  goal_approach_cnt / nt,
        "goal_recede_rate":    goal_recede_cnt  / nt,
        "avg_dist_reward":     float(avg_dist_reward),
        "avg_step_penalty":    float(avg_step_penalty),
    }

    torch.save(agent.policy_net.state_dict(), RESULTS_DIR / "v6_policy.pth")
    print(f"\n  [V6] 가중치 저장 -> results/v6_policy.pth")
    return arr_hist, col_hist, stats


# ══════════════════════════════════════════════════════════════════════════════
# V5-B 거리 보상 실측 (저장된 가중치로 추론 — 학습 없음)
# ══════════════════════════════════════════════════════════════════════════════

def _measure_v5b_dist_reward(n_ep: int = 200) -> dict:
    """v5b_policy.pth 가중치로 n_ep 에피소드 추론해서 거리 보상 분포를 측정한다.

    학습은 절대 하지 않는다(epsilon=0.05 고정, update 호출 없음).
    이 함수가 반환하는 approach/recede 비율이 '학습 완료 후 v5-B 정책'의
    실제 목표 접근 패턴이며, 훈련 초기(ε≈1.0)와는 다를 수 있음을 명시한다.
    """
    weights_path = RESULTS_DIR / "v5b_policy.pth"
    if not weights_path.exists():
        return {"error": "v5b_policy.pth 없음"}

    agent = DQNAgent()
    agent.policy_net.load_state_dict(
        torch.load(weights_path, map_location="cpu")
    )
    agent.policy_net.eval()
    agent.epsilon = 0.05   # v5-B 학습 완료 후 최소 epsilon과 동일

    env = DroneEnvN(list(OBSTACLES_8), fallback_pos=[1.0, 1.0])
    approach_cnt = recede_cnt = neutral_cnt = nt = 0

    for _ in range(n_ep):
        state = env.reset()
        for _ in range(MAX_STEPS):
            prev_dist = math.dist(env.pos, env.goal)
            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, _ = pgd_attack(state, D_obs_km, theta_obs_deg)
            action = agent.act(c_state)
            state, _, done, info = env.step(action)

            if not (info["goal_reached"] or info["collision"] or info["out_of_bounds"]):
                nt += 1
                curr_dist = math.dist(env.pos, env.goal)
                if curr_dist < prev_dist:
                    approach_cnt += 1
                elif curr_dist > prev_dist:
                    recede_cnt += 1
                else:
                    neutral_cnt += 1
            if done:
                break

    if nt == 0:
        return {"error": "비종료 스텝 없음"}

    avg_dist   = (approach_cnt * DIST_APPROACH + recede_cnt * DIST_RECEDE) / nt
    avg_total  = TIME_PENALTY + avg_dist
    return {
        "n_episodes":       n_ep,
        "nonterminal_steps": nt,
        "approach_cnt":     approach_cnt,
        "recede_cnt":       recede_cnt,
        "neutral_cnt":      neutral_cnt,
        "approach_rate":    approach_cnt / nt,
        "recede_rate":      recede_cnt   / nt,
        "avg_dist_reward":  avg_dist,
        "avg_step_penalty": avg_total,   # TIME_PENALTY + avg DIST
    }


# ══════════════════════════════════════════════════════════════════════════════
# V5-B 기준선 로드
# ══════════════════════════════════════════════════════════════════════════════

def _load_v5b() -> dict | None:
    p = RESULTS_DIR / "dqn_results_v5.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("v5b")


# ══════════════════════════════════════════════════════════════════════════════
# 결과 저장
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    arr_v6: list[float],
    col_v6: list[float],
    stats: dict,
    is_trial: bool = False,
) -> None:
    v5b   = _load_v5b()
    v5b_m = _measure_v5b_dist_reward(n_ep=200)   # 거리 보상 실측
    n     = len(arr_v6)
    eps   = list(range(1, n + 1))

    # ── v5-B ep100/200 기준값 ────────────────────────────────────────────────
    v5b_hist = v5b.get("arrival_history", []) if v5b else []
    v5b_ep100 = v5b_hist[99]  if len(v5b_hist) >= 100 else None
    v5b_ep200 = v5b_hist[199] if len(v5b_hist) >= 200 else None

    v6_ep100 = arr_v6[99]  if len(arr_v6) >= 100 else None
    v6_ep200 = arr_v6[199] if len(arr_v6) >= 200 else None

    # ── 비교 표 출력 ──────────────────────────────────────────────────────────
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"V6 학습 결과  ({'시험 실행 ' + str(n) + 'ep' if is_trial else '본 실행'})")
    print(sep)
    fmt = "{:<30} {:>16} {:>16}"
    print(fmt.format("항목", "V5-B (기준선)", "V6 (오염페널티)"))
    print("-" * 80)

    def v5b_val(key: str, fmt_str: str = "{:.4f}") -> str:
        if v5b is None:
            return "N/A"
        v = v5b.get(key)
        return fmt_str.format(v) if v is not None else "N/A"

    rows = [
        ("학습 에피소드",       v5b_val("episodes", "{:,}"),       f"{stats['episodes']:,}"),
        ("총 스텝 수",          v5b_val("total_steps", "{:,}"),    f"{stats['total_steps']:,}"),
        ("CE 폐기율",           v5b_val("discard_rate", "{:.1%}"), f"{stats['discard_rate']:.1%}"),
        ("유효 경험 수",        v5b_val("valid_exp", "{:,}"),      f"{stats['valid_exp']:,}"),
        ("도착 에피소드 수",    v5b_val("arrival_count", "{:,}"),  f"{stats['arrival_count']:,}"),
        ("─" * 28, "─" * 14, "─" * 14),
        ("ep100 도착률",
         f"{v5b_ep100:.4f}" if v5b_ep100 is not None else "N/A",
         f"{v6_ep100:.4f}"  if v6_ep100  is not None else "─"),
        ("ep200 도착률",
         f"{v5b_ep200:.4f}" if v5b_ep200 is not None else "N/A",
         f"{v6_ep200:.4f}"  if v6_ep200  is not None else "─"),
        ("최종 도착률 (100ep)", v5b_val("final_arr"),              f"{stats['final_arr']:.4f}"),
        ("최종 충돌률 (100ep)", v5b_val("final_col"),              f"{stats['final_col']:.4f}"),
        ("피크 도착률",         v5b_val("peak_arr"),
         f"{stats['peak_arr']:.4f} (Ep{stats['peak_ep']})"),
        ("─" * 28, "─" * 14, "─" * 14),
        ("RISK 페널티 발동",    "─",   f"{stats['risk_penalty_count']:,}"),
        ("CRITICAL 페널티 발동","─",   f"{stats['critical_penalty_count']:,}"),
        ("RISK 발동률 (/step)", "─",
         f"{stats['risk_penalty_count']/stats['total_steps']:.2%}"),
        ("CRITICAL 발동률 (/step)", "─",
         f"{stats['critical_penalty_count']/stats['total_steps']:.2%}"),
        ("─" * 28, "─" * 14, "─" * 14),
        ("비종료 스텝 중 목표 접근률", "─",
         f"{stats['goal_approach_rate']:.1%}"),
        ("비종료 스텝 중 목표 후퇴률", "─",
         f"{stats['goal_recede_rate']:.1%}"),
        ("실측 평균 거리 보상/step",   "─",
         f"{stats['avg_dist_reward']:+.5f}"),
        ("실측 평균 (시간+거리)/step", "─",
         f"{stats['avg_step_penalty']:+.5f}"),
    ]
    for r in rows:
        print(fmt.format(*r))
    print(sep)

    # ── v5-B 거리 보상 실측 결과 ─────────────────────────────────────────────
    print(f"\n[V5-B 거리 보상 실측 | v5b_policy.pth, {v5b_m.get('n_episodes',0)}ep, "
          f"epsilon=0.05 추론]")
    if "error" in v5b_m:
        print(f"  오류: {v5b_m['error']}")
    else:
        print(f"  비종료 스텝: {v5b_m['nonterminal_steps']:,}")
        print(f"  목표 접근률: {v5b_m['approach_rate']:.1%}  "
              f"({v5b_m['approach_cnt']:,}스텝 × +0.01)")
        print(f"  목표 후퇴률: {v5b_m['recede_rate']:.1%}  "
              f"({v5b_m['recede_cnt']:,}스텝 × -0.01)")
        print(f"  평균 거리 보상/step:    {v5b_m['avg_dist_reward']:+.5f}")
        print(f"  평균 (시간+거리)/step: {v5b_m['avg_step_penalty']:+.5f}  "
              f"  ← 이전 -0.02 가정 대비 실제값")

    # ── ep100/200 기준 비교 ────────────────────────────────────────────────
    if v5b_ep200 is not None and v6_ep200 is not None:
        diff = v6_ep200 - v5b_ep200
        trend = "↑ 상승" if (v6_ep100 is not None and v6_ep200 > v6_ep100) else "↓ 하락"
        print(f"\n[ep200 도착률 비교]")
        print(f"  V5-B ep200: {v5b_ep200:.4f}")
        print(f"  V6   ep200: {v6_ep200:.4f}  (차이 {diff:+.4f})  V6 추세: {trend}")

    print(f"\n  ※ RISK 조건: {SHIFT_THRESH_RISK}<shift<={SHIFT_THRESH_CRITICAL}px"
          f" + 접근중 + obs<{OBS_DIST_RISK_THRESH}km")
    print(f"  ※ CRITICAL 조건: shift>{SHIFT_THRESH_CRITICAL}px 무조건")
    print(f"  ※ 페널티: RISK={PENALTY_RISK}, CRITICAL={PENALTY_CRITICAL}"
          f"  (v3 -0.5 대비 각 1/50, 1/17)")

    # ── JSON 저장 (시험 실행은 저장 생략) ────────────────────────────────────
    if not is_trial:
        out_json = RESULTS_DIR / "dqn_results_v6.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "penalty_risk":       PENALTY_RISK,
                    "penalty_critical":   PENALTY_CRITICAL,
                    "shift_thresh_risk":  SHIFT_THRESH_RISK,
                    "shift_thresh_critical": SHIFT_THRESH_CRITICAL,
                    "time_penalty":       TIME_PENALTY,
                    "dist_approach":      DIST_APPROACH,
                    "dist_recede":        DIST_RECEDE,
                    "ce_deposit":         "continuous max(0.05, 1-shift/50)",
                    "note": (
                        "improved_ddm.py의 오염 판정/보상 로직을 실제 학습 루프에 연결. "
                        "페널티 크기는 v3/v4에서 -0.5가 학습 붕괴를 일으킨 이력을 반영해 축소. "
                        "타입별 리플레이 버퍼(ImprovedDDMBuffer)는 아직 미연결."
                    ),
                },
                "v6": {
                    **{k: v for k, v in stats.items() if k != "valid_exp_hist"},
                    "arrival_history":   [round(v, 4) for v in arr_v6],
                    "collision_history": [round(v, 4) for v in col_v6],
                },
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  수치 저장 -> {out_json}")

    # ── 그래프 ────────────────────────────────────────────────────────────────
    v5b_arr = v5b.get("arrival_history", [])[:n] if v5b else []
    v5b_col = v5b.get("collision_history", [])[:n] if v5b else []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#f5f6fa")
    title_suffix = f" [시험 실행 {n}ep]" if is_trial else f" [{n}ep 본 실행]"
    fig.suptitle(
        f"DQN v6: improved_ddm 오염 판정·보상 페널티 연결{title_suffix}\n"
        f"RISK={PENALTY_RISK}(접근중+{OBS_DIST_RISK_THRESH}km이내), "
        f"CRITICAL={PENALTY_CRITICAL}(무조건) | CE: max(0.05, 1-shift/50)",
        fontsize=10, fontweight="bold",
    )

    # 왼쪽: 도착률
    ax = axes[0]
    if v5b_arr:
        ax.plot(range(1, len(v5b_arr) + 1), v5b_arr,
                color="moccasin", lw=1.4, ls="--", alpha=0.85,
                label=f"V5-B 기준선  (최종 {v5b['final_arr']:.3f})")
    ax.plot(eps, arr_v6, color="steelblue", lw=1.8, alpha=0.9,
            label=f"V6 오염페널티  (최종 {stats['final_arr']:.3f}, "
                  f"피크 {stats['peak_arr']:.3f}@Ep{stats['peak_ep']})")
    ax.axhline(0.7400, color="gray", ls=":", lw=1.0, alpha=0.5,
               label="기준 (CE없음 순수DQN 0.740)")
    ax.set_title("도착률 비교 (V5-B vs V6)", fontsize=10, fontweight="bold")
    ax.set_xlabel("에피소드")
    ax.set_ylabel("최근 100ep 도착률")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n)
    ax.set_ylim(-0.02, 1.02)
    ax.set_facecolor("#f8f9fb")

    # 오른쪽: 충돌률
    ax = axes[1]
    if v5b_col:
        ax.plot(range(1, len(v5b_col) + 1), v5b_col,
                color="moccasin", lw=1.4, ls="--", alpha=0.85,
                label=f"V5-B 기준선  (최종 {v5b['final_col']:.3f})")
    ax.plot(eps, col_v6, color="darkorange", lw=1.8, alpha=0.9,
            label=f"V6 오염페널티  (최종 {stats['final_col']:.3f})")
    ax.set_title("충돌률 비교 (V5-B vs V6)", fontsize=10, fontweight="bold")
    ax.set_xlabel("에피소드")
    ax.set_ylabel("최근 100ep 충돌률")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n)
    ax.set_ylim(-0.02, 1.02)
    ax.set_facecolor("#f8f9fb")

    plt.tight_layout()
    suffix  = "_trial" if is_trial else ""
    out_png = RESULTS_DIR / f"dqn_comparison_v6{suffix}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  그래프 저장 -> {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="DQN v6 학습")
    parser.add_argument(
        "--episodes", type=int, default=EPISODES_V6,
        help=f"학습 에피소드 수 (기본 {EPISODES_V6}, 시험 실행은 200 권장)",
    )
    args = parser.parse_args()
    is_trial = args.episodes < EPISODES_V6

    print("=" * 60)
    print("DQN v6: improved_ddm 오염 판정·보상 페널티 실제 연결")
    print(f"환경: 8장애물 랜덤 (v5 동일)")
    print(f"{'[시험 실행] ' if is_trial else ''}에피소드: {args.episodes}")
    print(f"RISK={PENALTY_RISK}(접근중), CRITICAL={PENALTY_CRITICAL}(무조건)")
    print("=" * 60)

    arr, col, stats = train_v6(n_episodes=args.episodes)
    save_results(arr, col, stats, is_trial=is_trial)

    print("\n완료.")


if __name__ == "__main__":
    main()
