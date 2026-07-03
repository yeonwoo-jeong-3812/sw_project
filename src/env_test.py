"""환경 단독 테스트: 5장애물/랜덤 vs 8장애물/랜덤.

실험 목적:
  v3 실험(8장애물 + 고정 시작/목적지)의 성능 저하 원인 분리.
  - 가설 A: 장애물 수 자체(8개)가 문제
  - 가설 B: 고정 경로(항상 양쪽 벽 통과)가 문제

공통 조건:
  - CE 필터링 없음 (모든 경험 저장)
  - 회피 보너스 없음
  - PGD ε=0.03 공격 적용 (v2-A 동일)
  - 1000 에피소드
"""

from __future__ import annotations

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

# ── 공유 import (DQNAgent, pgd_attack) ─────────────────────────────────────
from dqn_comparison import (
    DQNAgent, pgd_attack,
    SPACE_W, SPACE_H, STEP_SIZE, GOAL_RADIUS, OBS_RADIUS,
    MAX_STEPS, DIAG,
)

# ══════════════════════════════════════════════════════════════════════════════
# 장애물 레이아웃
# ══════════════════════════════════════════════════════════════════════════════

OBSTACLES_5: list[tuple[float, float]] = [
    (3.0,  3.0),
    (7.5,  9.0),
    (12.0, 6.0),
    (5.0, 15.0),
    (10.0, 12.0),
]

# v3과 동일한 8개 배치 (2중 벽 구조), 단 이번엔 시작/목적지는 랜덤
OBSTACLES_8: list[tuple[float, float]] = [
    (4.0,  6.0),   # wall-1 좌
    (7.5,  7.0),   # wall-1 중
    (11.0, 6.5),   # wall-1 우
    (3.5, 11.0),   # wall-2 좌
    (7.0, 12.0),   # wall-2 중
    (10.5, 11.5),  # wall-2 우
    (8.5,  3.0),   # 하단 보조
    (9.0, 15.0),   # 상단 보조
]

EPISODES_TEST = 1000


# ══════════════════════════════════════════════════════════════════════════════
# 설정 가능한 드론 환경 (장애물 파라미터화)
# ══════════════════════════════════════════════════════════════════════════════

class DroneEnvN:
    """장애물 리스트를 파라미터로 받는 드론 환경.

    시작/목적지 모두 랜덤 (장애물 안전 거리 OBS_RADIUS*2 이상).
    """

    def __init__(
        self,
        obstacles: list[tuple[float, float]],
        fallback_pos: list[float] | None = None,
    ) -> None:
        self.obstacles    = obstacles
        self.fallback_pos = fallback_pos or [1.0, 1.0]
        self.pos:     list[float] = [0.0, 0.0]
        self.heading: float       = 0.0
        self.goal:    list[float] = [0.0, 0.0]
        self.steps:   int         = 0

    # ── 초기화 ────────────────────────────────────────────────────────────────

    def reset(self) -> list[float]:
        self.pos     = self._random_free_pos()
        self.heading = random.uniform(0.0, 360.0)
        self.goal    = self._random_free_pos(min_dist_from=self.pos, min_dist=3.0)
        self.steps   = 0
        return self._state()

    def _min_obs_dist(self, x: float, y: float) -> float:
        return min(math.dist([x, y], list(o)) for o in self.obstacles)

    def _random_free_pos(
        self,
        min_dist_from: list[float] | None = None,
        min_dist: float = 0.0,
        max_tries: int = 2000,
    ) -> list[float]:
        for _ in range(max_tries):
            x = random.uniform(0.5, SPACE_W - 0.5)
            y = random.uniform(0.5, SPACE_H - 0.5)
            if self._min_obs_dist(x, y) <= OBS_RADIUS * 2:
                continue
            if min_dist_from is not None:
                if math.dist([x, y], min_dist_from) < min_dist:
                    continue
            return [x, y]
        return list(self.fallback_pos)

    # ── 상태 계산 ─────────────────────────────────────────────────────────────

    def _nearest_obstacle(self) -> tuple[float, float]:
        x, y     = self.pos
        best_d   = float("inf")
        best_obs = self.obstacles[0]
        for obs in self.obstacles:
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
        return [
            D_goal,
            theta_goal / 180.0,
            D_obs / DIAG,
            theta_obs / 180.0,
        ]

    def get_raw_obstacle(self) -> tuple[float, float]:
        return self._nearest_obstacle()

    # ── 스텝 ──────────────────────────────────────────────────────────────────

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        delta = {0: 0, 1: 15, 2: -15, 3: 30, 4: -30}[action]
        self.heading = (self.heading + delta) % 360.0
        rad = math.radians(self.heading)
        self.pos[0] += STEP_SIZE * math.cos(rad)
        self.pos[1] += STEP_SIZE * math.sin(rad)
        self.steps  += 1

        x, y = self.pos
        obs_dist      = self._min_obs_dist(x, y)
        goal_reached  = math.dist(self.pos, self.goal) < GOAL_RADIUS
        collision     = obs_dist < OBS_RADIUS
        out_of_bounds = not (0.0 <= x <= SPACE_W and 0.0 <= y <= SPACE_H)
        timeout       = self.steps >= MAX_STEPS
        done          = goal_reached or collision or out_of_bounds or timeout

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
# 학습 루틴 (CE 없음, 회피보너스 없음, 순수 DQN)
# ══════════════════════════════════════════════════════════════════════════════

def train_env(
    obstacles: list[tuple[float, float]],
    label: str,
    fallback_pos: list[float] | None = None,
) -> tuple[list[float], list[float], dict]:
    """순수 DQN 학습 (CE 필터 없음, 회피보너스 없음, PGD 공격 O).

    Args:
        obstacles:    사용할 장애물 리스트
        label:        로그/그래프 표시 이름
        fallback_pos: 랜덤 위치 생성 실패 시 fallback

    Returns:
        arr_hist:  에피소드별 최근 100ep 도착률
        col_hist:  에피소드별 최근 100ep 충돌률
        stats:     학습 통계
    """
    env   = DroneEnvN(obstacles, fallback_pos=fallback_pos)
    agent = DQNAgent()

    arr_hist:   list[float] = []
    col_hist:   list[float] = []
    window_arr  = deque(maxlen=100)
    window_col  = deque(maxlen=100)

    total_steps   = 0
    arrival_count = 0
    peak_arr      = 0.0

    print(f"\n[{label}] 학습 시작 "
          f"(장애물 {len(obstacles)}개, 랜덤 시작/목적지, {EPISODES_TEST}ep)")

    for ep in range(EPISODES_TEST):
        state   = env.reset()
        arrived = False
        hit     = False

        for _ in range(MAX_STEPS):
            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, _shift = pgd_attack(state, D_obs_km, theta_obs_deg)

            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)
            total_steps += 1

            D_obs_next, theta_obs_next = env.get_raw_obstacle()
            c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)

            # 모든 경험 저장 (CE 없음)
            agent.push(c_state, action, reward, c_next, done)
            agent.update()

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
        peak_arr = max(peak_arr, arr_hist[-1])

        if (ep + 1) % 100 == 0:
            print(f"  Ep {ep+1:4d} | 도착률={arr_hist[-1]:.3f}"
                  f" | 충돌률={col_hist[-1]:.3f}"
                  f" | ε={agent.epsilon:.3f}"
                  f" | 총경험={total_steps:,}")

    peak_ep = int(np.argmax(arr_hist)) + 1
    stats = {
        "label":         label,
        "obstacles":     len(obstacles),
        "episodes":      EPISODES_TEST,
        "total_steps":   total_steps,
        "arrival_count": arrival_count,
        "final_arr":     float(arr_hist[-1]),
        "final_col":     float(col_hist[-1]),
        "mean_arr_last100": float(np.mean(arr_hist[-100:])),
        "peak_arr":      float(peak_arr),
        "peak_ep":       peak_ep,
    }
    return arr_hist, col_hist, stats


# ══════════════════════════════════════════════════════════════════════════════
# 결과 저장
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    arr_5: list[float], col_5: list[float], stats_5: dict,
    arr_8: list[float], col_8: list[float], stats_8: dict,
) -> None:
    """비교 표 출력 + 그래프·JSON 저장."""

    eps = list(range(1, EPISODES_TEST + 1))

    # ── 표 출력 ──────────────────────────────────────────────────────────────
    sep = "=" * 68
    print(f"\n{sep}")
    print("환경 테스트 결과 비교 (CE 없음, 회피보너스 없음, PGD ε=0.03)")
    print(sep)
    fmt = "{:<22} {:>20} {:>20}"
    h5  = f"Env-1 (5장애물/랜덤)"
    h8  = f"Env-2 (8장애물/랜덤)"
    print(fmt.format("항목", h5, h8))
    print("-" * 68)
    rows = [
        ("학습 에피소드",       f"{stats_5['episodes']:,}",     f"{stats_8['episodes']:,}"),
        ("장애물 수",            f"{stats_5['obstacles']}개",    f"{stats_8['obstacles']}개"),
        ("총 경험 수집량",       f"{stats_5['total_steps']:,}", f"{stats_8['total_steps']:,}"),
        ("도착 에피소드 수",     f"{stats_5['arrival_count']:,}",f"{stats_8['arrival_count']:,}"),
        ("최종 도착률 (100ep)", f"{stats_5['final_arr']:.4f}", f"{stats_8['final_arr']:.4f}"),
        ("최종 충돌률 (100ep)", f"{stats_5['final_col']:.4f}", f"{stats_8['final_col']:.4f}"),
        ("피크 도착률",          f"{stats_5['peak_arr']:.4f}",  f"{stats_8['peak_arr']:.4f}"),
        ("피크 도착 에피소드",   f"Ep {stats_5['peak_ep']}",    f"Ep {stats_8['peak_ep']}"),
    ]
    for name, v5, v8 in rows:
        print(fmt.format(name, v5, v8))
    print(sep)

    # v2-A 참조값 표시 (동일 환경 비교)
    print(f"\n※ v2-A 참조: 5장애물/랜덤/1000ep → 최종 도착률 0.8755 (CE 이진차단 포함)")
    print(f"※ v3-A 참조: 8장애물/고정경로/1000ep → 최종 도착률 0.0000 (피크 0.47 at ep400)")
    print(f"\n분석 포인트:")
    gap = stats_5['final_arr'] - stats_8['final_arr']
    if gap > 0.3:
        print(f"  → Env-2 도착률이 {gap:.3f} 낮음 → 장애물 수 자체가 유의미한 영향")
    elif gap > 0.1:
        print(f"  → Env-2 도착률이 {gap:.3f} 낮음 → 장애물 수가 부분적 영향")
    else:
        print(f"  → 도착률 차이 {gap:.3f} → 장애물 수보다 고정 경로가 주 원인")

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "env_test_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "env1_5obs_random": {**stats_5,
                                 "arrival_history": [round(v, 4) for v in arr_5],
                                 "collision_history": [round(v, 4) for v in col_5]},
            "env2_8obs_random": {**stats_8,
                                 "arrival_history": [round(v, 4) for v in arr_8],
                                 "collision_history": [round(v, 4) for v in col_8]},
            "reference": {
                "v2a_5obs_random_final_arr": 0.8755,
                "v3a_8obs_fixed_final_arr":  0.0000,
                "v3a_8obs_fixed_peak_arr":   0.47,
                "v3a_8obs_fixed_peak_ep":    400,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n수치 저장 → {out_json}")

    # ── 그래프 (2×1) ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    # ── (상단) 도착률 학습 곡선 ────────────────────────────────────────────────
    ax1.plot(eps, arr_5, color="steelblue",  linewidth=1.6, alpha=0.9,
             label=f"Env-1  5장애물/랜덤  (최종 {stats_5['final_arr']:.3f}, "
                   f"피크 {stats_5['peak_arr']:.3f}@Ep{stats_5['peak_ep']})")
    ax1.plot(eps, arr_8, color="darkorange", linewidth=1.6, alpha=0.9,
             label=f"Env-2  8장애물/랜덤  (최종 {stats_8['final_arr']:.3f}, "
                   f"피크 {stats_8['peak_arr']:.3f}@Ep{stats_8['peak_ep']})")

    # v3-A 참조선
    ax1.axhline(0.0000, color="red",   linestyle=":", linewidth=1.0, alpha=0.6,
                label="v3-A 참조 (8장애물/고정경로 최종: 0.000)")
    ax1.axhline(0.8755, color="green", linestyle=":", linewidth=1.0, alpha=0.6,
                label="v2-A 참조 (5장애물/랜덤     최종: 0.876)")

    ax1.set_ylabel("최근 100ep 도착률", fontsize=11)
    ax1.set_title(
        "환경 단독 테스트: 5장애물/랜덤 vs 8장애물/랜덤\n"
        "(CE 없음 · 회피보너스 없음 · PGD ε=0.03 공격 · 1000ep)",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(fontsize=9.5, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, EPISODES_TEST)
    ax1.set_ylim(-0.02, 1.02)

    # ── (하단) 충돌률 학습 곡선 ────────────────────────────────────────────────
    ax2.plot(eps, col_5, color="steelblue",  linewidth=1.6, alpha=0.9,
             label=f"Env-1  5장애물/랜덤  (최종 충돌률 {stats_5['final_col']:.3f})")
    ax2.plot(eps, col_8, color="darkorange", linewidth=1.6, alpha=0.9,
             label=f"Env-2  8장애물/랜덤  (최종 충돌률 {stats_8['final_col']:.3f})")

    ax2.set_xlabel("에피소드", fontsize=11)
    ax2.set_ylabel("최근 100ep 충돌률", fontsize=11)
    ax2.set_title("충돌률 비교", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9.5, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, EPISODES_TEST)
    ax2.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    out_png = RESULTS_DIR / "env_test.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"그래프 저장 → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("환경 단독 테스트: 장애물 수 vs 고정 경로 영향 분리")
    print(f"공간: {SPACE_W}km × {SPACE_H}km | OBS_RADIUS={OBS_RADIUS}km | "
          f"STEP={STEP_SIZE}km | MAX_STEPS={MAX_STEPS}")
    print(f"CE 없음 | 회피보너스 없음 | PGD ε=0.03 공격 적용")
    print("=" * 60)

    arr_5, col_5, stats_5 = train_env(
        OBSTACLES_5, label="Env-1 (5장애물/랜덤)",
    )
    arr_8, col_8, stats_8 = train_env(
        OBSTACLES_8, label="Env-2 (8장애물/랜덤)",
        fallback_pos=[1.0, 1.0],  # 8장애물 배치에서 안전한 fallback
    )

    save_results(arr_5, col_5, stats_5, arr_8, col_8, stats_8)
    print("\n완료.")


if __name__ == "__main__":
    main()
