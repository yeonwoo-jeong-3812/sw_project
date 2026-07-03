"""V5-A / V5-B 일반화 테스트 (장애물 랜덤 재배치).

실행 흐름:
  1. 가중치 파일(results/v5a_policy.pth, results/v5b_policy.pth) 존재 확인
     → 없으면 V5와 동일한 설정으로 학습 1회 실행 후 저장 (최초 1회만)
  2. 가중치 로드 → epsilon=0.0 (탐색 없음, 추론 전용)
  3. 매 에피소드마다 8개 장애물 랜덤 재배치 → 100 에피소드 추론
  4. 비교 표 출력 + results/generalization_test.png 저장

기존 V5 결과 (고정 장애물, dqn_comparison_v5.py 학습 결과):
  V5-A (이진 CE, 10px 임계값)  : 도착률 49%
  V5-B (연속 CE, max(0.05,...)) : 도착률 78%
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch

# ── 한글 폰트 ──────────────────────────────────────────────────────────────
_prefer = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Gulim', 'Dotum']
_available = {f.name for f in fm.fontManager.ttflist}
for _font in _prefer:
    if _font in _available:
        plt.rcParams['font.family'] = _font
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from dqn_comparison import (
    DQNAgent, pgd_attack,
    SPACE_W, SPACE_H, OBS_RADIUS, MAX_STEPS,
)
from env_test import DroneEnvN, OBSTACLES_8

# ── 재현성 ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── V5 동일 상수 ──────────────────────────────────────────────────────────
CE_BINARY_THRESH = 10.0   # 이진 CE 임계값 (px)
CE_SHIFT_MAX     = 50.0   # 연속 CE 정규화 기준 (px)
TIME_PENALTY     = -0.01
DIST_APPROACH    = +0.01
DIST_RECEDE      = -0.01
EPISODES_V5      = 1000   # V5 학습 에피소드 수

# ── 설정 ──────────────────────────────────────────────────────────────────
N_TEST_EPISODES = 100
N_OBSTACLES     = 8
V5A_WEIGHTS     = RESULTS_DIR / "v5a_policy.pth"
V5B_WEIGHTS     = RESULTS_DIR / "v5b_policy.pth"

# 기존 V5 도착률 (dqn_comparison_v5.py 학습 결과 — 고정 장애물 배치)
V5_ORIGINAL = {"v5a": 0.49, "v5b": 0.78}

# ── CE deposit 함수 ────────────────────────────────────────────────────────
def _deposit_binary(shift: float) -> float:
    return 0.0 if shift > CE_BINARY_THRESH else 1.0

def _deposit_continuous(shift: float) -> float:
    return max(0.05, 1.0 - shift / CE_SHIFT_MAX)

# ── 랜덤 장애물 배치 ──────────────────────────────────────────────────────
def random_obstacles(
    n: int = N_OBSTACLES,
    min_dist: float = OBS_RADIUS * 2.0,  # 3.0 km
    margin: float = 1.5,
    max_tries: int = 5000,
) -> list[tuple[float, float]]:
    """n개 장애물을 min_dist 이상 간격을 두고 랜덤 배치.
    배치 실패 시 기존 OBSTACLES_8 반환.
    """
    obstacles: list[tuple[float, float]] = []
    tries = 0
    while len(obstacles) < n and tries < max_tries:
        tries += 1
        x = random.uniform(margin, SPACE_W - margin)
        y = random.uniform(margin, SPACE_H - margin)
        if all(math.dist([x, y], list(o)) >= min_dist for o in obstacles):
            obstacles.append((x, y))
    return obstacles if len(obstacles) == n else list(OBSTACLES_8)

# ── V5 학습 + 가중치 저장 (가중치 파일 없을 때만 실행) ──────────────────
def _train_and_save(
    deposit_fn,
    is_probabilistic: bool,
    save_path: Path,
    label: str,
) -> None:
    """V5와 동일한 하이퍼파라미터로 학습 후 policy_net 가중치를 저장한다."""
    print(f"  [{label}] 가중치 파일 없음 → V5 동일 설정으로 학습 ({EPISODES_V5}ep)")
    env   = DroneEnvN(list(OBSTACLES_8), fallback_pos=[1.0, 1.0])
    agent = DQNAgent()

    for ep in range(EPISODES_V5):
        state = env.reset()
        for _ in range(MAX_STEPS):
            prev_dist = math.dist(env.pos, env.goal)
            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)
            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)

            # V5 보상 조형 (시간 페널티 + 거리 기반 보상)
            if not (info["goal_reached"] or info["collision"] or info["out_of_bounds"]):
                reward += TIME_PENALTY
                curr_dist = math.dist(env.pos, env.goal)
                if curr_dist < prev_dist:
                    reward += DIST_APPROACH
                elif curr_dist > prev_dist:
                    reward += DIST_RECEDE

            # CE 필터링 후 경험 저장
            ratio = deposit_fn(shift)
            if is_probabilistic:
                should_deposit = (random.random() < ratio)
            else:
                should_deposit = (ratio > 0.0)

            if should_deposit:
                D_obs_next, theta_obs_next = env.get_raw_obstacle()
                c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
                agent.push(c_state, action, reward, c_next, done)

            agent.update()
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        if (ep + 1) % 200 == 0:
            print(f"    Ep {ep+1}/{EPISODES_V5}  ε={agent.epsilon:.3f}")

    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"  [{label}] 가중치 저장 완료 → {save_path}\n")

# ── 가중치 로드 (없으면 학습 후 저장) ────────────────────────────────────
def load_agent(
    weights_path: Path,
    deposit_fn,
    is_probabilistic: bool,
    label: str,
) -> DQNAgent:
    if not weights_path.exists():
        _train_and_save(deposit_fn, is_probabilistic, weights_path, label)

    agent = DQNAgent()
    agent.policy_net.load_state_dict(
        torch.load(weights_path, map_location="cpu")
    )
    agent.policy_net.eval()
    agent.epsilon = 0.0  # 추론 시 탐색 비활성화
    print(f"  [{label}] 가중치 로드 완료: {weights_path.name}")
    return agent

# ── 추론 실행 ─────────────────────────────────────────────────────────────
def run_test(agent: DQNAgent, label: str) -> list[int]:
    """매 에피소드마다 장애물을 랜덤 재배치한 환경에서 100 에피소드 추론.

    반환: 에피소드별 성공(1) / 실패(0) 리스트
    """
    results: list[int] = []
    print(f"\n[{label}] 일반화 테스트  |  100 에피소드  |  장애물 매회 랜덤 재배치")

    for ep in range(N_TEST_EPISODES):
        new_obs = random_obstacles()
        env     = DroneEnvN(new_obs, fallback_pos=[1.0, 1.0])
        state   = env.reset()
        arrived = False

        for _ in range(MAX_STEPS):
            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, _ = pgd_attack(state, D_obs_km, theta_obs_deg)
            action = agent.act(c_state)          # epsilon=0 → greedy 선택
            state, _, done, info = env.step(action)
            if info["goal_reached"]:
                arrived = True
                break
            if done:
                break

        results.append(1 if arrived else 0)
        if (ep + 1) % 25 == 0:
            rate = sum(results) / len(results)
            print(f"  Ep {ep+1:3d}  누적 도착률 {rate:.1%}")

    final = sum(results) / len(results)
    print(f"  [{label}] 최종 도착률: {final:.1%}")
    return results

# ── 표 출력 + 그래프 저장 ─────────────────────────────────────────────────
def report(results_a: list[int], results_b: list[int]) -> None:
    arr_a = sum(results_a) / len(results_a)
    arr_b = sum(results_b) / len(results_b)

    # ── 비교 표 ───────────────────────────────────────────────────────────
    sep = "=" * 66
    fmt = "{:<26} {:>12} {:>12} {:>12}"
    print(f"\n{sep}")
    print("  일반화 테스트 결과  |  장애물 랜덤 재배치  |  100 에피소드")
    print(sep)
    print(fmt.format("항목", "V5-A (이진 CE)", "V5-B (연속 CE)", "차이 (B-A)"))
    print("-" * 66)
    print(fmt.format(
        "기존 V5 도착률 (고정 장애물)",
        f"{V5_ORIGINAL['v5a']:.1%}",
        f"{V5_ORIGINAL['v5b']:.1%}",
        f"{V5_ORIGINAL['v5b'] - V5_ORIGINAL['v5a']:+.1%}",
    ))
    print(fmt.format(
        "일반화 도착률 (랜덤 장애물)",
        f"{arr_a:.1%}",
        f"{arr_b:.1%}",
        f"{arr_b - arr_a:+.1%}",
    ))
    print(fmt.format(
        "도착률 변화 (일반화 − 기존)",
        f"{arr_a - V5_ORIGINAL['v5a']:+.1%}",
        f"{arr_b - V5_ORIGINAL['v5b']:+.1%}",
        "",
    ))
    print(sep)

    # ── 그래프 ────────────────────────────────────────────────────────────
    eps   = list(range(1, N_TEST_EPISODES + 1))
    cum_a = [sum(results_a[:i]) / i for i in range(1, N_TEST_EPISODES + 1)]
    cum_b = [sum(results_b[:i]) / i for i in range(1, N_TEST_EPISODES + 1)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#f5f6fa")
    fig.suptitle(
        "V5-A vs V5-B  일반화 테스트 (8개 장애물 매 에피소드 랜덤 재배치)\n"
        "epsilon=0 추론 전용  |  100 에피소드",
        fontsize=12, fontweight="bold",
    )

    # 왼쪽: 누적 도착률 추이
    ax = axes[0]
    ax.scatter(eps, results_a, color="steelblue", s=12, alpha=0.35, zorder=2)
    ax.scatter(eps, results_b, color="darkorange", s=12, alpha=0.35, zorder=2)
    ax.plot(eps, cum_a, color="steelblue",   lw=2.0, label=f"V5-A 이진CE  (최종 {arr_a:.1%})")
    ax.plot(eps, cum_b, color="darkorange",  lw=2.0, label=f"V5-B 연속CE  (최종 {arr_b:.1%})")
    ax.axhline(V5_ORIGINAL["v5a"], color="steelblue",  ls="--", lw=1.1, alpha=0.5,
               label=f"V5-A 기존 {V5_ORIGINAL['v5a']:.0%} (고정)")
    ax.axhline(V5_ORIGINAL["v5b"], color="darkorange", ls="--", lw=1.1, alpha=0.5,
               label=f"V5-B 기존 {V5_ORIGINAL['v5b']:.0%} (고정)")
    ax.set_xlabel("에피소드", fontsize=11)
    ax.set_ylabel("누적 도착률", fontsize=11)
    ax.set_title("누적 도착률 추이 (점=에피소드 결과, 선=누적)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, N_TEST_EPISODES)
    ax.set_ylim(-0.05, 1.1)
    ax.set_facecolor("#f8f9fb")

    # 오른쪽: 기존 vs 일반화 막대 비교
    ax = axes[1]
    cats = ["V5-A\n(이진 CE\n10px 임계)", "V5-B\n(연속 CE\nmax(0.05,…))"]
    x = np.arange(len(cats))
    w = 0.35
    bars_orig = ax.bar(x - w/2, [V5_ORIGINAL["v5a"], V5_ORIGINAL["v5b"]],
                       w, label="기존 (고정 장애물)", color=["#5b9bd5", "#ed7d31"], alpha=0.65)
    bars_gen  = ax.bar(x + w/2, [arr_a, arr_b],
                       w, label="일반화 (랜덤 장애물)", color=["#2e75b6", "#c55a11"], alpha=0.90)

    for bar, v in zip(bars_orig, [V5_ORIGINAL["v5a"], V5_ORIGINAL["v5b"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.0%}", ha="center", va="bottom", fontsize=10, color="#555")
    for bar, v in zip(bars_gen, [arr_a, arr_b]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("도착률", fontsize=11)
    ax.set_title("기존 V5 vs 일반화 도착률 비교", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.0, 1.15)
    ax.set_facecolor("#f8f9fb")

    plt.tight_layout()
    out_path = RESULTS_DIR / "generalization_test.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n그래프 저장 완료 → {out_path}")

# ── 메인 ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 66)
    print("V5-A / V5-B  일반화 테스트")
    print(f"환경: 8개 장애물 매 에피소드 랜덤 재배치  |  {N_TEST_EPISODES} 에피소드")
    print(f"모드: epsilon=0  추론 전용  (학습 없음)")
    print("=" * 66)

    print("\n[1단계] 모델 가중치 준비")
    agent_a = load_agent(V5A_WEIGHTS, _deposit_binary,     is_probabilistic=False, label="V5-A")
    agent_b = load_agent(V5B_WEIGHTS, _deposit_continuous, is_probabilistic=True,  label="V5-B")

    print("\n[2단계] 일반화 테스트 실행")
    results_a = run_test(agent_a, "V5-A")
    results_b = run_test(agent_b, "V5-B")

    print("\n[3단계] 결과 정리")
    report(results_a, results_b)

    print("\n완료.")

if __name__ == "__main__":
    main()
