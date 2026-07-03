"""비행 경로 시각화: 기존 DDM vs 개선 DDM (임계값 20px).

동일한 10개 테스트 시나리오에서 두 에이전트를 실행하고
비행 경로·장애물·PGD 오염 위치를 10행 × 2열 그리드로 비교한다.

왼쪽 열 : 기존 DDM   (PGD 공격, 모든 경험 저장)
오른쪽 열: 개선 DDM  (PGD 공격, CE 임계값 20px 적용)

표시 항목:
  - 드론 경로      : 파란 실선
  - 장애물         : 빨간 원 (반경 1.5km)
  - 목적지         : 노란 별
  - 충돌/이탈 지점 : 빨간 X
  - 목적지 도달    : 초록 삼각형
  - PGD 오염 위치  : 주황색 점 (shift > 20px)
"""

from __future__ import annotations

import math
import random
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

# ── 한글 폰트 ────────────────────────────────────────────────────────────────
_prefer = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic',
           'Gulim', 'Dotum', 'Batang']
_available = {f.name for f in fm.fontManager.ttflist}
for _font in _prefer:
    if _font in _available:
        plt.rcParams['font.family'] = _font
        break
plt.rcParams['axes.unicode_minus'] = False

# ── dqn_comparison 모듈 임포트 ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from dqn_comparison import (
    DroneEnv, DQNAgent, pgd_attack,
    OBSTACLES, SPACE_W, SPACE_H, OBS_RADIUS, GOAL_RADIUS,
    MAX_STEPS, EPISODES, RISK_DIST,
    PGD_D_ERR_RATE, PGD_T_ERR_DEG, PGD_MEAN_SHIFT, PGD_STD_SHIFT,
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 상수 ─────────────────────────────────────────────────────────────────────
TEST_EPISODES  = 10       # 테스트 에피소드 수
CE_THRESH_B    = 20.0     # 개선 DDM CE 임계값 (px)
VIS_CE_THRESH  = 20.0     # 시각화에서 "오염" 기준 (두 에이전트 동일하게 표시)
SCENARIO_SEED  = 7777     # 테스트 시나리오 고정 시드
ATTACK_SEED    = 9999     # 공격 노이즈 고정 시드 (양쪽 동일 노이즈)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 에이전트 학습
# ══════════════════════════════════════════════════════════════════════════════

def _run_train_loop(
    agent: DQNAgent,
    env: DroneEnv,
    use_ce_filter: bool,
    ce_threshold: float = CE_THRESH_B,
    tag: str = "",
) -> None:
    """공통 학습 루프.

    Args:
        use_ce_filter: True → CE 필터링 + RISK 페널티 (개선 DDM)
                       False → 모든 경험 저장 (기존 DDM)
    """
    window_arr = deque(maxlen=100)
    window_col = deque(maxlen=100)

    for ep in range(EPISODES):
        state   = env.reset()
        arrived = False
        hit     = False

        for _ in range(MAX_STEPS):
            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)

            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)

            if use_ce_filter and info["obs_dist"] < RISK_DIST and not info["goal_reached"]:
                reward += -0.5

            should_store = (not use_ce_filter) or (shift <= ce_threshold)
            if should_store:
                D_obs_n, theta_obs_n = env.get_raw_obstacle()
                c_next, _ = pgd_attack(next_state, D_obs_n, theta_obs_n)
                agent.push(c_state, action, reward, c_next, done)

            agent.update()
            state = next_state

            if info["goal_reached"]:
                arrived = True
                break
            if info["collision"] or info["out_of_bounds"] or done:
                hit = info["collision"] or info["out_of_bounds"]
                break

        agent.decay_epsilon()
        window_arr.append(1 if arrived else 0)
        window_col.append(1 if hit else 0)

        if (ep + 1) % 250 == 0:
            print(f"  {tag} Ep {ep+1:4d} | "
                  f"도착률={sum(window_arr)/len(window_arr):.3f} | "
                  f"ε={agent.epsilon:.3f}")

    agent.epsilon = 0.0   # 테스트 시 greedy


def train_classic_ddm() -> DQNAgent:
    """기존 DDM 에이전트 학습 (1000 에피소드)."""
    print("[1/2] 기존 DDM 학습 중...")
    random.seed(42); np.random.seed(42)
    agent = DQNAgent()
    env   = DroneEnv()
    _run_train_loop(agent, env, use_ce_filter=False, tag="[기존DDM]")
    print("      완료\n")
    return agent


def train_improved_ddm(threshold: float = CE_THRESH_B) -> DQNAgent:
    """개선 DDM 에이전트 학습 (임계값 threshold px, 1000 에피소드)."""
    print(f"[2/2] 개선 DDM 학습 중 (임계값={threshold}px)...")
    random.seed(42); np.random.seed(42)
    agent = DQNAgent()
    env   = DroneEnv()
    _run_train_loop(agent, env, use_ce_filter=True,
                    ce_threshold=threshold, tag="[개선DDM]")
    print("      완료\n")
    return agent


# ══════════════════════════════════════════════════════════════════════════════
# 2. 테스트 시나리오 생성 및 에피소드 실행
# ══════════════════════════════════════════════════════════════════════════════

def generate_scenarios(n: int, seed: int) -> list[dict]:
    """n개의 고정 테스트 시나리오 (시작 위치·헤딩·목적지) 생성."""
    random.seed(seed)
    np.random.seed(seed)
    env = DroneEnv()
    scenarios = []
    for _ in range(n):
        env.reset()
        scenarios.append({
            "pos":     list(env.pos),
            "heading": env.heading,
            "goal":    list(env.goal),
        })
    return scenarios


def run_episode(
    agent: DQNAgent,
    scenario: dict,
    attack_seed: int,
) -> dict:
    """고정 시나리오에서 한 에피소드를 실행하고 경로 데이터를 반환한다.

    PGD 노이즈는 attack_seed로 고정하여 양쪽 에이전트가 동일한 공격에 노출된다.

    Returns dict:
      path          : [(x, y), ...]          – 전체 이동 경로
      corrupted_pts : [(x, y), ...]          – shift > VIS_CE_THRESH 위치
      goal          : [gx, gy]
      start         : [sx, sy]
      result        : 'arrived'|'collision'|'out_of_bounds'|'timeout'
      end_pos       : (x, y)
      n_corrupt     : int  – 오염 스텝 수
    """
    random.seed(attack_seed)
    np.random.seed(attack_seed)

    env = DroneEnv()
    env.pos     = list(scenario["pos"])
    env.heading = scenario["heading"]
    env.goal    = list(scenario["goal"])
    env.steps   = 0
    state = env._state()

    path: list[tuple[float, float]]          = [(env.pos[0], env.pos[1])]
    corrupted_pts: list[tuple[float, float]] = []
    result  = "timeout"
    end_pos = (env.pos[0], env.pos[1])

    for _ in range(MAX_STEPS):
        D_obs_km, theta_obs_deg = env.get_raw_obstacle()
        c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)

        if shift > VIS_CE_THRESH:
            corrupted_pts.append((env.pos[0], env.pos[1]))

        action = agent.act(c_state)
        next_state, reward, done, info = env.step(action)
        path.append((env.pos[0], env.pos[1]))
        state = next_state

        if info["goal_reached"]:
            result  = "arrived"
            end_pos = (env.pos[0], env.pos[1])
            break
        if info["collision"]:
            result  = "collision"
            end_pos = (env.pos[0], env.pos[1])
            break
        if info["out_of_bounds"]:
            result  = "out_of_bounds"
            end_pos = (env.pos[0], env.pos[1])
            break

    return {
        "path":          path,
        "corrupted_pts": corrupted_pts,
        "goal":          list(scenario["goal"]),
        "start":         list(scenario["pos"]),
        "result":        result,
        "end_pos":       end_pos,
        "n_corrupt":     len(corrupted_pts),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. 단일 서브플롯 렌더링
# ══════════════════════════════════════════════════════════════════════════════

_RESULT_LABEL = {
    "arrived":      "도착 성공",
    "collision":    "충돌",
    "out_of_bounds":"이탈",
    "timeout":      "시간초과",
}
_RESULT_COLOR = {
    "arrived":      "darkgreen",
    "collision":    "crimson",
    "out_of_bounds":"saddlebrown",
    "timeout":      "gray",
}


def draw_episode(ax: plt.Axes, ep: dict, ep_num: int, agent_label: str) -> None:
    """단일 에피소드 비행 경로를 ax에 그린다."""
    ax.set_facecolor("#f5f5f5")
    ax.set_xlim(0, SPACE_W)
    ax.set_ylim(0, SPACE_H)
    ax.set_aspect("equal")

    # 그리드
    ax.grid(True, alpha=0.2, linewidth=0.4, color="gray")

    # 장애물 (빨간 원)
    for ox, oy in OBSTACLES:
        ax.add_patch(mpatches.Circle(
            (ox, oy), OBS_RADIUS,
            facecolor="#ff4444", alpha=0.22, edgecolor="#cc0000",
            linewidth=0.8, zorder=2,
        ))
        ax.plot(ox, oy, "+", color="#cc0000", markersize=5,
                markeredgewidth=0.8, zorder=3)

    # 목적지 영역 (연노랑 원)
    gx, gy = ep["goal"]
    ax.add_patch(mpatches.Circle(
        (gx, gy), GOAL_RADIUS,
        facecolor="gold", alpha=0.18, edgecolor="goldenrod",
        linewidth=0.8, zorder=2,
    ))
    # 목적지 별
    ax.plot(gx, gy, "*", color="goldenrod", markersize=13,
            markeredgecolor="#996600", markeredgewidth=0.6, zorder=7)

    # 비행 경로 (파란 실선)
    xs = [p[0] for p in ep["path"]]
    ys = [p[1] for p in ep["path"]]
    ax.plot(xs, ys, color="steelblue", linewidth=1.1, alpha=0.85, zorder=4)

    # 시작점 삼각형
    sx, sy = ep["start"]
    ax.plot(sx, sy, "^", color="steelblue", markersize=7,
            markeredgecolor="navy", markeredgewidth=0.6, zorder=6)

    # PGD 오염 위치 (주황색 점)
    if ep["corrupted_pts"]:
        cxs = [p[0] for p in ep["corrupted_pts"]]
        cys = [p[1] for p in ep["corrupted_pts"]]
        ax.scatter(cxs, cys, c="darkorange", s=14, alpha=0.75,
                   zorder=5, linewidths=0)

    # 종료 표시
    ex, ey = ep["end_pos"]
    result = ep["result"]
    if result == "arrived":
        ax.plot(ex, ey, "^", color="limegreen", markersize=10,
                markeredgecolor="darkgreen", markeredgewidth=1.0, zorder=8)
    else:
        ax.plot(ex, ey, "x", color="crimson", markersize=10,
                markeredgewidth=2.0, zorder=8)

    # 서브플롯 제목
    result_str = _RESULT_LABEL[result]
    tc         = _RESULT_COLOR[result]
    n_c        = ep["n_corrupt"]
    ax.set_title(
        f"{agent_label}  에피소드 {ep_num}\n"
        f"{result_str}  |  오염 {n_c}건",
        fontsize=8, color=tc, fontweight="bold", pad=2,
    )

    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 6, 12, 18])
    ax.tick_params(labelsize=5.5)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 전체 그림 구성 및 저장
# ══════════════════════════════════════════════════════════════════════════════

def make_legend_handles() -> list:
    return [
        plt.Line2D([0], [0], color="steelblue", linewidth=2,
                   label="비행 경로"),
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor="steelblue", markersize=8,
                   markeredgecolor="navy", label="시작점"),
        mpatches.Patch(facecolor="#ff4444", alpha=0.4,
                       edgecolor="#cc0000", label="장애물 (r=1.5km)"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="goldenrod", markersize=12,
                   label="목적지"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="darkorange", markersize=7,
                   label=f"PGD 오염 (shift>{VIS_CE_THRESH:.0f}px)"),
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor="limegreen", markersize=8,
                   markeredgecolor="darkgreen", label="도착 성공"),
        plt.Line2D([0], [0], marker="x", color="crimson",
                   markersize=8, markeredgewidth=2, label="충돌 / 이탈"),
    ]


def visualize(results_a: list[dict], results_b: list[dict]) -> None:
    n = len(results_a)
    fig, axes = plt.subplots(
        n, 2,
        figsize=(10, n * 1.95),
        squeeze=False,
        gridspec_kw={"wspace": 0.22, "hspace": 0.60},
    )

    for i in range(n):
        draw_episode(axes[i][0], results_a[i], i + 1, "기존 DDM")
        draw_episode(axes[i][1], results_b[i], i + 1, "개선 DDM")

    # 열 헤더 — 첫 행 위쪽 여백에 배치
    axes[0][0].annotate(
        "◀ 기존 DDM (PGD 공격, 전체 저장)",
        xy=(0.5, 1.22), xycoords="axes fraction",
        ha="center", fontsize=9, fontweight="bold", color="steelblue",
    )
    axes[0][1].annotate(
        "개선 DDM (CE 임계값 20px) ▶",
        xy=(0.5, 1.22), xycoords="axes fraction",
        ha="center", fontsize=9, fontweight="bold", color="darkorange",
    )

    # 전체 제목
    fig.suptitle(
        "기존 DDM vs 개선 DDM 비행 경로 비교\n(PGD ε=0.03 공격 환경)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # 하단 공통 범례
    fig.legend(
        handles=make_legend_handles(),
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.012),
        framealpha=0.9,
    )

    out_png = RESULTS_DIR / "flight_comparison.png"
    plt.savefig(out_png, dpi=85, bbox_inches="tight")
    plt.close()
    print(f"\n그래프 저장 → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results_a: list[dict], results_b: list[dict]) -> None:
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"{'에피소드':>6}  {'기존 DDM 결과':<18}  {'개선 DDM 결과':<18}")
    print("-" * 58)
    for i, (ra, rb) in enumerate(zip(results_a, results_b)):
        la = _RESULT_LABEL[ra["result"]]
        lb = _RESULT_LABEL[rb["result"]]
        print(f"  {i+1:>4}    {la:<18}  {lb:<18}")
    print("-" * 58)
    arr_a = sum(1 for r in results_a if r["result"] == "arrived")
    arr_b = sum(1 for r in results_b if r["result"] == "arrived")
    print(f"  도착 성공:  기존 DDM {arr_a}/{len(results_a)}회  |  "
          f"개선 DDM {arr_b}/{len(results_b)}회")
    print(sep)


def main() -> None:
    print("=" * 60)
    print("비행 경로 시각화: 기존 DDM vs 개선 DDM (임계값 20px)")
    print(f"테스트 에피소드: {TEST_EPISODES}  |  학습: 각 {EPISODES}에피소드")
    print("=" * 60)

    # ── 학습 ─────────────────────────────────────────────────────────────────
    agent_a = train_classic_ddm()
    agent_b = train_improved_ddm(threshold=CE_THRESH_B)

    # ── 테스트 시나리오 생성 (고정 시드) ────────────────────────────────────
    scenarios = generate_scenarios(TEST_EPISODES, seed=SCENARIO_SEED)
    print(f"[테스트] {TEST_EPISODES}개 시나리오 실행 중...")

    results_a: list[dict] = []
    results_b: list[dict] = []
    for i, sc in enumerate(scenarios):
        seed = ATTACK_SEED + i * 13   # 에피소드별 공격 시드 (양쪽 동일)
        ep_a = run_episode(agent_a, sc, attack_seed=seed)
        ep_b = run_episode(agent_b, sc, attack_seed=seed)
        results_a.append(ep_a)
        results_b.append(ep_b)
        print(f"  [{i+1:2d}] 기존={_RESULT_LABEL[ep_a['result']]:<6}  "
              f"개선={_RESULT_LABEL[ep_b['result']]:<6}  "
              f"오염 기존={ep_a['n_corrupt']}건 / 개선={ep_b['n_corrupt']}건")

    # ── 요약 출력 ────────────────────────────────────────────────────────────
    print_summary(results_a, results_b)

    # ── 시각화 ───────────────────────────────────────────────────────────────
    visualize(results_a, results_b)
    print("완료.")


if __name__ == "__main__":
    main()
