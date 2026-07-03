"""DQN 비교 v3: 기존 DDM vs 연속 가중치 개선 DDM.

v2 대비 세 가지 개선:
1. 연속 가중치 CE: deposit_ratio = max(0.05, 1.0 - shift/50)  (확률적 저장)
2. 회피 보너스: RISK 탐지 범위(3km) 안에서 장애물에서 멀어질 때 +0.1
3. 장애물 8개 (v2: 5개), 고정 시작(1.0,1.0)·목적지(13.5,16.5)로 밀집 구간 통과 강제

버전 A (기존 DDM):
  - 이진 CE 없음 (모든 경험 저장)
  - 회피 보너스 적용 (새 환경 공통)
  - RISK 페널티 없음

버전 B (연속 가중치 개선 DDM):
  - 연속 가중치 CE (shift 클수록 낮은 확률로 저장)
  - 회피 보너스 적용
  - RISK 페널티 -0.5

저장: results/dqn_comparison_v3.png, results/dqn_results_v3.json
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch

# dqn_comparison 에서 재사용 가능한 컴포넌트만 임포트
sys.path.insert(0, str(Path(__file__).parent))
from dqn_comparison import (
    DQNAgent, pgd_attack,
    SPACE_W, SPACE_H, STEP_SIZE, GOAL_RADIUS, OBS_RADIUS,
    MAX_STEPS, DIAG,
)

# ── 한글 폰트 ────────────────────────────────────────────────────────────────
_prefer = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "Gulim", "Dotum"]
_avail  = {f.name for f in fm.fontManager.ttflist}
for _f in _prefer:
    if _f in _avail:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# V3 환경 설정
# ══════════════════════════════════════════════════════════════════════════════

# 8개 장애물: 2열 벽(wall-1 y≈6~7, wall-2 y≈11~12) + 보조 2개
# 벽 내 장애물 간격 3.54–3.64km → 통과 가능한 좁은 틈(0.54~0.64km) 확보
# 최소 장애물 간 거리: 3.54km — 모든 쌍이 충돌 반경(1.5km×2=3.0km) 이상
OBSTACLES_V3: list[tuple[float, float]] = [
    (4.0,  6.0),   # wall-1 좌
    (7.5,  7.0),   # wall-1 중
    (11.0, 6.5),   # wall-1 우
    (3.5, 11.0),   # wall-2 좌
    (7.0, 12.0),   # wall-2 중
    (10.5, 11.5),  # wall-2 우
    (8.5,  3.0),   # 하단 보조 (경로 우회 강제)
    (9.0, 15.0),   # 상단 보조 (목적지 접근 제한)
]
START_V3        = [1.0,  1.0]    # 고정 시작점 (좌하단)
GOAL_V3         = [13.5, 16.5]  # 고정 목적지 (우상단)
RISK_DIST_V3    = OBS_RADIUS * 2.0   # 3.0 km
AVOIDANCE_BONUS = 0.1
CE_SHIFT_MAX    = 50.0    # shift ≥ 50px → ratio = 0.05 (하한)
EPISODES_V3     = 1000


# ══════════════════════════════════════════════════════════════════════════════
# 1. V3 드론 환경
# ══════════════════════════════════════════════════════════════════════════════

class DroneEnvV3:
    """V3 환경: 장애물 8개, 고정 시작·목적지, 헤딩만 랜덤."""

    def __init__(self) -> None:
        self.pos:     list[float] = list(START_V3)
        self.heading: float       = 0.0
        self.goal:    list[float] = list(GOAL_V3)
        self.steps:   int         = 0

    def reset(self) -> list[float]:
        self.pos     = list(START_V3)
        self.heading = random.uniform(0.0, 360.0)
        self.goal    = list(GOAL_V3)
        self.steps   = 0
        return self._state()

    def _min_obs_dist(self, x: float, y: float) -> float:
        return min(math.dist([x, y], list(o)) for o in OBSTACLES_V3)

    def _nearest_obstacle(self) -> tuple[float, float]:
        x, y     = self.pos
        best_d   = float("inf")
        best_obs = OBSTACLES_V3[0]
        for obs in OBSTACLES_V3:
            d = math.dist([x, y], list(obs))
            if d < best_d:
                best_d, best_obs = d, obs
        ox, oy    = best_obs
        abs_angle = math.degrees(math.atan2(oy - y, ox - x))
        rel_angle = (abs_angle - self.heading + 180.0) % 360.0 - 180.0
        return best_d, rel_angle

    def _state(self) -> list[float]:
        x, y   = self.pos
        gx, gy = self.goal
        D_goal     = math.dist(self.pos, self.goal) / DIAG
        abs_g_ang  = math.degrees(math.atan2(gy - y, gx - x))
        theta_goal = (abs_g_ang - self.heading + 180.0) % 360.0 - 180.0
        D_obs, theta_obs = self._nearest_obstacle()
        return [D_goal, theta_goal / 180.0, D_obs / DIAG, theta_obs / 180.0]

    def get_raw_obstacle(self) -> tuple[float, float]:
        return self._nearest_obstacle()

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        delta = {0: 0, 1: 15, 2: -15, 3: 30, 4: -30}[action]
        self.heading = (self.heading + delta) % 360.0
        rad = math.radians(self.heading)
        self.pos[0] += STEP_SIZE * math.cos(rad)
        self.pos[1] += STEP_SIZE * math.sin(rad)
        self.steps  += 1

        x, y         = self.pos
        obs_dist     = self._min_obs_dist(x, y)
        goal_reached = math.dist(self.pos, self.goal) < GOAL_RADIUS
        collision    = obs_dist < OBS_RADIUS
        out_of_bounds = not (0.0 <= x <= SPACE_W and 0.0 <= y <= SPACE_H)
        timeout      = self.steps >= MAX_STEPS
        done         = goal_reached or collision or out_of_bounds or timeout

        if goal_reached:
            reward = 1.0
        elif collision or out_of_bounds:
            reward = -1.0
        else:
            reward = 0.0

        info = {
            "goal_reached":  goal_reached,
            "collision":     collision,
            "out_of_bounds": out_of_bounds,
            "obs_dist":      obs_dist,
        }
        return self._state(), reward, done, info


# ══════════════════════════════════════════════════════════════════════════════
# 2. 연속 가중치 CE
# ══════════════════════════════════════════════════════════════════════════════

def _deposit_ratio(shift: float) -> float:
    """shift → 경험 저장 확률 [0.05, 1.0].

    shift  5px  →  0.90
    shift 15px  →  0.70
    shift 25px  →  0.50
    shift 50px+ →  0.05 (하한)
    """
    return max(0.05, 1.0 - shift / CE_SHIFT_MAX)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 에피소드 실행 (공통 루프)
# ══════════════════════════════════════════════════════════════════════════════

def _run_episode(
    env: DroneEnvV3,
    agent: DQNAgent,
    use_continuous_ce: bool,
    use_risk_penalty: bool,
) -> tuple[bool, bool, int, int]:
    """단일 에피소드 실행.

    Returns:
        arrived:   도착 성공 여부
        hit:       충돌/이탈 여부
        total_ep:  이 에피소드의 총 스텝 수
        valid_ep:  실제 메모리에 저장된 유효 경험 수
    """
    state   = env.reset()
    arrived = hit = False
    total_ep = valid_ep = 0

    for _ in range(MAX_STEPS):
        obs_dist_before = env._min_obs_dist(*env.pos)

        D_obs_km, theta_obs_deg = env.get_raw_obstacle()
        c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)

        action = agent.act(c_state)
        next_state, reward, done, info = env.step(action)
        total_ep += 1

        # ── RISK 페널티 (버전 B) ────────────────────────────────────────────
        if use_risk_penalty and info["obs_dist"] < RISK_DIST_V3 and not info["goal_reached"]:
            reward += -0.5

        # ── 회피 보너스 (두 버전 공통) ─────────────────────────────────────
        # RISK 범위 안에 있다가 장애물에서 멀어지면 +0.1
        if (obs_dist_before < RISK_DIST_V3
                and info["obs_dist"] > obs_dist_before
                and not info["goal_reached"]):
            reward += AVOIDANCE_BONUS

        # ── 경험 저장 (CE 방식 선택) ─────────────────────────────────────
        if use_continuous_ce:
            deposit = random.random() < _deposit_ratio(shift)
        else:
            deposit = True

        if deposit:
            D_obs_next, theta_obs_next = env.get_raw_obstacle()
            c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
            agent.push(c_state, action, reward, c_next, done)
            valid_ep += 1

        agent.update()
        state = next_state

        if info["goal_reached"]:
            arrived = True
            break
        if info["collision"] or info["out_of_bounds"] or done:
            hit = info["collision"] or info["out_of_bounds"]
            break

    return arrived, hit, total_ep, valid_ep


# ══════════════════════════════════════════════════════════════════════════════
# 4. 학습 함수
# ══════════════════════════════════════════════════════════════════════════════

def _train(
    use_continuous_ce: bool,
    use_risk_penalty: bool,
    tag: str,
) -> tuple[list[float], list[float], dict]:
    """공통 학습 루프. (arr_hist, col_hist, stats) 반환."""
    env   = DroneEnvV3()
    agent = DQNAgent()

    arr_hist:  list[float] = []
    col_hist:  list[float] = []
    window_arr = deque(maxlen=100)
    window_col = deque(maxlen=100)

    total_steps   = 0
    valid_exp     = 0
    ce_blocked    = 0
    arrival_count = 0
    valid_exp_hist: list[int] = []

    for ep in range(EPISODES_V3):
        arrived, hit, t_ep, v_ep = _run_episode(
            env, agent, use_continuous_ce, use_risk_penalty
        )
        total_steps += t_ep
        valid_exp   += v_ep
        ce_blocked  += (t_ep - v_ep)

        agent.decay_epsilon()

        if arrived:
            arrival_count += 1
        window_arr.append(1 if arrived else 0)
        window_col.append(1 if hit else 0)
        arr_hist.append(sum(window_arr) / len(window_arr))
        col_hist.append(sum(window_col) / len(window_col))
        valid_exp_hist.append(valid_exp)

        if (ep + 1) % 100 == 0:
            blk = ce_blocked / total_steps if total_steps else 0.0
            print(f"  {tag} Ep {ep+1:4d}"
                  f" | 도착률={arr_hist[-1]:.3f}"
                  f" | 충돌률={col_hist[-1]:.3f}"
                  f" | ε={agent.epsilon:.3f}"
                  f" | 유효경험={valid_exp:,}"
                  f" | CE차단={blk:.1%}")

    return arr_hist, col_hist, {
        "total_steps":    total_steps,
        "ce_blocked":     ce_blocked,
        "valid_exp":      valid_exp,
        "valid_exp_hist": valid_exp_hist,
        "arrival_count":  arrival_count,
        "episodes":       EPISODES_V3,
    }


def train_v3a() -> tuple[list[float], list[float], dict]:
    print(f"\n[V3-A] 기존 DDM + 회피 보너스 + 8장애물  [{EPISODES_V3}ep]")
    return _train(use_continuous_ce=False, use_risk_penalty=False, tag="[V3-A]")


def train_v3b() -> tuple[list[float], list[float], dict]:
    # v3 개선 DDM: 연속 가중치 CE + 회피 보너스
    # ※ RISK 페널티(-0.5)는 v3에서 제외:
    #    고정 경로가 장애물 밀집 구간을 통과해야 하므로 -0.5×N스텝 패널티가
    #    목표 보상(+1.0)을 압도하여 학습 자체를 방해함
    print(f"\n[V3-B] 연속 가중치 CE + 회피 보너스 + 8장애물  [{EPISODES_V3}ep]")
    return _train(use_continuous_ce=True, use_risk_penalty=False, tag="[V3-B]")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 결과 저장 + v2 비교표
# ══════════════════════════════════════════════════════════════════════════════

def _safe(v, fmt: str) -> str:
    """숫자면 fmt로, nan/None이면 'N/A' 반환."""
    try:
        return f"{v:{fmt}}"
    except (ValueError, TypeError):
        return "N/A".rjust(len(fmt) + 2)


def save_results_v3(
    arr_a: list[float], col_a: list[float], stats_a: dict,
    arr_b: list[float], col_b: list[float], stats_b: dict,
) -> None:
    final_arr_a = float(np.mean(arr_a[-100:]))
    final_col_a = float(np.mean(col_a[-100:]))
    final_arr_b = float(np.mean(arr_b[-100:]))
    final_col_b = float(np.mean(col_b[-100:]))

    blk_a = stats_a["ce_blocked"] / stats_a["total_steps"] if stats_a["total_steps"] else 0.0
    blk_b = stats_b["ce_blocked"] / stats_b["total_steps"] if stats_b["total_steps"] else 0.0

    # ── v2 결과 로드 ──────────────────────────────────────────────────────────
    v2a: dict = {}
    v2b: dict = {}
    v2_json = RESULTS_DIR / "dqn_results.json"
    if v2_json.exists():
        try:
            with open(v2_json, "r", encoding="utf-8") as f:
                v2data = json.load(f)
            v2a = v2data.get("version_a", {})
            v2b = v2data.get("version_b", {})
        except Exception:
            pass

    def gv2(d: dict, key: str, default=float("nan")):
        return d.get(key, default)

    v2a_ep  = int(gv2(v2a, "episodes",              1000))
    v2b_ep  = int(gv2(v2b, "episodes",              2000))
    v2a_tot = int(gv2(v2a, "total_steps",              0))
    v2b_tot = int(gv2(v2b, "total_steps",              0))
    v2a_ce  = int(gv2(v2a, "ce_blocked",               0))
    v2b_ce  = int(gv2(v2b, "ce_blocked",               0))
    v2a_val = int(gv2(v2a, "valid_exp",         v2a_tot))
    v2b_val = int(gv2(v2b, "valid_exp",               0))
    v2a_arr_cnt = int(gv2(v2a, "arrival_count",        0))
    v2b_arr_cnt = int(gv2(v2b, "arrival_count",        0))
    v2a_arr = float(gv2(v2a, "final_arrival_rate",  float("nan")))
    v2b_arr = float(gv2(v2b, "final_arrival_rate",  float("nan")))
    v2a_col = float(gv2(v2a, "final_collision_rate", float("nan")))
    v2b_col = float(gv2(v2b, "final_collision_rate", float("nan")))
    v2a_blk = v2a_ce / v2a_tot if v2a_tot else 0.0
    v2b_blk = v2b_ce / v2b_tot if v2b_tot else 0.0

    # ── 비교 표 출력 ──────────────────────────────────────────────────────────
    sep  = "=" * 76
    sep2 = "-" * 76
    C = 14   # 컬럼 너비
    def row(label, v2av, v2bv, v3av, v3bv, fw="s"):
        return f"{label:<24} {v2av:>{C}} {v2bv:>{C}} {v3av:>{C}} {v3bv:>{C}}"

    print(f"\n{sep}")
    print("v2 결과 vs v3 결과 비교표")
    print(sep)
    print(row("항목", "v2-A(기존)", "v2-B(개선)", "v3-A(기존)", "v3-B(연속CE)"))
    print(sep2)
    print(row("학습 에피소드",
              f"{v2a_ep:,}", f"{v2b_ep:,}",
              f"{stats_a['episodes']:,}", f"{stats_b['episodes']:,}"))
    print(row("총 경험 수집량",
              f"{v2a_tot:,}", f"{v2b_tot:,}",
              f"{stats_a['total_steps']:,}", f"{stats_b['total_steps']:,}"))
    print(row("CE 폐기 경험 수",
              f"{v2a_ce:,}", f"{v2b_ce:,}",
              f"{stats_a['ce_blocked']:,}", f"{stats_b['ce_blocked']:,}"))
    print(row("CE 차단 방식",
              "없음", "이진(10px)", "없음", "연속가중치"))
    print(row("CE 폐기율",
              f"{v2a_blk:.1%}", f"{v2b_blk:.1%}",
              f"{blk_a:.1%}", f"{blk_b:.1%}"))
    print(row("유효 경험 수",
              f"{v2a_val:,}", f"{v2b_val:,}",
              f"{stats_a['valid_exp']:,}", f"{stats_b['valid_exp']:,}"))
    print(row("도착 에피소드 수",
              f"{v2a_arr_cnt:,}", f"{v2b_arr_cnt:,}",
              f"{stats_a['arrival_count']:,}", f"{stats_b['arrival_count']:,}"))
    print(sep2)
    print(row("최종 도착률 (100ep)",
              _safe(v2a_arr, ".4f"), _safe(v2b_arr, ".4f"),
              f"{final_arr_a:.4f}", f"{final_arr_b:.4f}"))
    print(row("최종 충돌률 (100ep)",
              _safe(v2a_col, ".4f"), _safe(v2b_col, ".4f"),
              f"{final_col_a:.4f}", f"{final_col_b:.4f}"))
    print(row("환경",
              "5장애물/랜덤", "5장애물/랜덤",
              "8장애물/고정", "8장애물/고정"))
    print(row("회피 보너스", "없음", "없음", "+0.1", "+0.1"))
    print(row("RISK 페널티",  "없음", "-0.5", "없음", "없음(※)"))
    print(f"  ※ v3-B RISK 패널티 제외: 밀집 구간 강제 통과 경로에서 -0.5×N >> +1.0 도착 보상 → 학습 불가")
    print(sep)

    # ── 그래프 ───────────────────────────────────────────────────────────────
    eps = list(range(1, EPISODES_V3 + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#f5f6fa")

    # 왼쪽: v3 학습 곡선
    ax1.plot(eps, arr_a, color="#e74c3c", linewidth=1.6, alpha=0.9,
             label=f"V3-A 기존 DDM  →  최종 {final_arr_a:.3f}")
    ax1.plot(eps, arr_b, color="#2980b9", linewidth=1.6, alpha=0.9,
             label=f"V3-B 연속CE DDM  →  최종 {final_arr_b:.3f}")
    ax1.set_xlabel("에피소드", fontsize=11)
    ax1.set_ylabel("최근 100ep 도착률", fontsize=11)
    ax1.set_title(
        "V3 학습 성능 비교\n"
        f"(8장애물·고정 경로·회피 보너스 | {EPISODES_V3}ep)",
        fontsize=11, fontweight="bold",
    )
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, EPISODES_V3)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_facecolor("#f8f9fb")

    # 오른쪽: v2 vs v3 최종 도착률 막대 비교
    labels_bar = [
        "v2-A\n기존\n(1000ep\n5장애물)",
        "v2-B\n이진CE\n(2000ep\n5장애물)",
        "v3-A\n기존\n(1000ep\n8장애물)",
        "v3-B\n연속CE\n(1000ep\n8장애물)",
    ]
    arr_vals = [
        v2a_arr if not math.isnan(v2a_arr) else 0.0,
        v2b_arr if not math.isnan(v2b_arr) else 0.0,
        final_arr_a,
        final_arr_b,
    ]
    col_vals = [
        v2a_col if not math.isnan(v2a_col) else 0.0,
        v2b_col if not math.isnan(v2b_col) else 0.0,
        final_col_a,
        final_col_b,
    ]
    x = np.arange(len(labels_bar))
    w = 0.35
    bars_a = ax2.bar(x - w/2, arr_vals, w, label="도착률",
                     color=["#e74c3c", "#c0392b", "#e67e22", "#d35400"], alpha=0.85)
    bars_c = ax2.bar(x + w/2, col_vals, w, label="충돌률",
                     color=["#f1948a", "#d5dbdb", "#abebc6", "#a9cce3"], alpha=0.75)

    for bar, v in zip(bars_a, arr_vals):
        if v > 0.02:
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color="darkred")
    for bar, v in zip(bars_c, col_vals):
        if v > 0.02:
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=8, color="steelblue")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_bar, fontsize=8.5)
    ax2.set_ylabel("비율", fontsize=11)
    ax2.set_title("v2 vs v3 최종 성능 비교\n(최종 100ep 평균)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0.0, 1.2)
    ax2.set_facecolor("#f8f9fb")

    plt.tight_layout()
    out_png = RESULTS_DIR / "dqn_comparison_v3.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\n그래프 저장 → {out_png}")

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "dqn_results_v3.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "v3_a": {
                    "description": "기존 DDM + 회피 보너스 + 8장애물 + 고정 시작/목적지",
                    "final_arrival_rate":   round(final_arr_a, 4),
                    "final_collision_rate": round(final_col_a, 4),
                    **{k: v for k, v in stats_a.items() if k != "valid_exp_hist"},
                    "obstacles": len(OBSTACLES_V3),
                    "start": START_V3,
                    "goal":  GOAL_V3,
                },
                "v3_b": {
                    "description": "연속 가중치 CE + RISK 페널티 + 회피 보너스 + 8장애물",
                    "ce_mode":   f"continuous: max(0.05, 1.0 - shift/{CE_SHIFT_MAX})",
                    "final_arrival_rate":   round(final_arr_b, 4),
                    "final_collision_rate": round(final_col_b, 4),
                    **{k: v for k, v in stats_b.items() if k != "valid_exp_hist"},
                    "obstacles": len(OBSTACLES_V3),
                    "start": START_V3,
                    "goal":  GOAL_V3,
                },
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"수치 저장 → {out_json}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("DQN 비교 v3: 기존 DDM vs 연속 가중치 개선 DDM")
    print(f"환경: {SPACE_W}km × {SPACE_H}km | 장애물 {len(OBSTACLES_V3)}개 (r={OBS_RADIUS}km)")
    print(f"시작: {START_V3}  →  목적지: {GOAL_V3}  (고정)")
    print(f"CE 방식 (B): 연속 가중치  max(0.05, 1.0 - shift/{CE_SHIFT_MAX:.0f})")
    print(f"회피 보너스 (공통): +{AVOIDANCE_BONUS}  |  RISK 페널티: v3에서 제외 (밀집 환경 과적용 방지)")
    print(f"에피소드: {EPISODES_V3} × 2버전  |  최대 스텝/ep: {MAX_STEPS}")
    print("=" * 70)

    arr_a, col_a, stats_a = train_v3a()
    arr_b, col_b, stats_b = train_v3b()
    save_results_v3(arr_a, col_a, stats_a, arr_b, col_b, stats_b)
    print("\n완료.")


if __name__ == "__main__":
    main()
