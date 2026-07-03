"""고정 장애물 + epsilon=0 추론 테스트 (V5-A vs V5-B).

로드 대상:
  results/v5a_policy.pth  — V5-A (이진 CE) 학습 가중치
  results/v5b_policy.pth  — V5-B (연속 CE) 학습 가중치

  ※ 가중치 파일이 없으면 오류를 출력하고 종료한다. (재학습 없음)

환경: env_test.py의 OBSTACLES_8 고정 배치 (dqn_comparison_v5.py와 동일)
정책: epsilon=0 순수 그리디
에피소드: 100

2×2 비교표 출력 + results/fixed_vs_random_epsilon0.png 저장

출력 표 구조:
                  고정 장애물               랜덤 장애물
  eps-greedy   V5-A 49.0% / V5-B 78.0%   (미측정)
  eps=0        (이번 측정)                V5-A 47.0% / V5-B 89.0%  ← generalization_test.py

※ 랜덤+eps=0 수치는 generalization_test.py 결과 참고값
"""

from __future__ import annotations

import math
import sys
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
sys.path.insert(0, str(Path(__file__).parent))

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
V5A_WEIGHTS = RESULTS_DIR / "v5a_policy.pth"
V5B_WEIGHTS = RESULTS_DIR / "v5b_policy.pth"

from dqn_comparison import DQNAgent, pgd_attack, MAX_STEPS
from env_test import DroneEnvN, OBSTACLES_8

# ── 설정 ──────────────────────────────────────────────────────────────────
N_TEST_EPISODES = 100

# 참고값 (학습 중 eps-greedy 정책 기준, dqn_comparison_v5.py 결과)
V5_EPSGREEDY_FIXED = {"v5a": 0.490, "v5b": 0.780}

# 참고값 (eps=0 + 랜덤 장애물, generalization_test.py 결과)
V5_EPS0_RANDOM = {"v5a": 0.470, "v5b": 0.890}

# ── 가중치 로드 (없으면 오류 종료) ────────────────────────────────────────
def _load_agent(weights_path: Path, label: str) -> DQNAgent:
    if not weights_path.exists():
        print(f"\n[오류] 가중치 파일을 찾을 수 없습니다: {weights_path}")
        print("  → dqn_comparison_v5.py를 먼저 실행해서 가중치를 생성하세요.")
        sys.exit(1)

    agent = DQNAgent()
    agent.policy_net.load_state_dict(
        torch.load(weights_path, map_location="cpu")
    )
    agent.policy_net.eval()
    agent.epsilon = 0.0  # 순수 그리디
    print(f"  [{label}] 가중치 로드 완료: {weights_path.name}")
    return agent

# ── 추론 실행 (고정 장애물) ────────────────────────────────────────────────
def run_test(agent: DQNAgent, label: str) -> list[int]:
    """OBSTACLES_8 고정 배치에서 100 에피소드 eps=0 추론.

    반환: 에피소드별 성공(1) / 실패(0) 리스트
    """
    env     = DroneEnvN(list(OBSTACLES_8), fallback_pos=[1.0, 1.0])
    results: list[int] = []
    print(f"\n[{label}] 고정 장애물  |  100 에피소드  |  epsilon=0 (순수 그리디)")

    for ep in range(N_TEST_EPISODES):
        state   = env.reset()
        arrived = False

        for _ in range(MAX_STEPS):
            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, _ = pgd_attack(state, D_obs_km, theta_obs_deg)
            action = agent.act(c_state)          # epsilon=0 → greedy
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

# ── 2×2 비교 표 출력 + 그래프 저장 ──────────────────────────────────────
def report(arr_a: float, arr_b: float) -> None:
    # ── 2×2 비교 표 ───────────────────────────────────────────────────────
    sep  = "=" * 72
    dash = "-" * 72
    col  = "{:<20} {:>24} {:>24}"
    print(f"\n{sep}")
    print("  고정 장애물 vs 랜덤 장애물  ×  eps-greedy vs eps=0 비교표")
    print(sep)
    print(col.format("", "고정 장애물", "랜덤 장애물"))
    print(dash)
    print(col.format(
        "eps-greedy (학습 중)",
        f"V5-A {V5_EPSGREEDY_FIXED['v5a']:.1%}  /  V5-B {V5_EPSGREEDY_FIXED['v5b']:.1%}",
        "(미측정)",
    ))
    print(col.format(
        "eps=0 (순수 그리디)",
        f"V5-A {arr_a:.1%}  /  V5-B {arr_b:.1%}",
        f"V5-A {V5_EPS0_RANDOM['v5a']:.1%}  /  V5-B {V5_EPS0_RANDOM['v5b']:.1%}",
    ))
    print(sep)
    print("  ※ eps-greedy 고정: dqn_comparison_v5.py 학습 결과")
    print("  ※ eps=0 랜덤:      generalization_test.py 결과 참고값")

    delta_a = arr_a - V5_EPSGREEDY_FIXED["v5a"]
    delta_b = arr_b - V5_EPSGREEDY_FIXED["v5b"]
    print(f"\n  eps=0 적용 효과 (고정 장애물, greedy vs greedy+학습epsilon 비교):")
    print(f"    V5-A: {V5_EPSGREEDY_FIXED['v5a']:.1%} → {arr_a:.1%}  ({delta_a:+.1%})")
    print(f"    V5-B: {V5_EPSGREEDY_FIXED['v5b']:.1%} → {arr_b:.1%}  ({delta_b:+.1%})")

    # ── 막대 그래프 (이번 측정값 vs 랜덤+eps=0) ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#f5f6fa")
    fig.suptitle(
        "epsilon=0 (순수 그리디)  |  고정 장애물 vs 랜덤 장애물 도착률 비교\n"
        "V5-A (이진 CE 10px)  vs  V5-B (연속 CE max(0.05,…))",
        fontsize=11, fontweight="bold",
    )

    labels_model = ["V5-A\n(이진 CE)", "V5-B\n(연속 CE)"]
    x = np.arange(len(labels_model))
    w = 0.38

    # 왼쪽: eps=0 고정 vs eps=0 랜덤
    ax = axes[0]
    fixed_vals  = [arr_a, arr_b]
    random_vals = [V5_EPS0_RANDOM["v5a"], V5_EPS0_RANDOM["v5b"]]

    bars_f = ax.bar(x - w/2, fixed_vals,  w,
                    label="고정 장애물 (이번 측정)", color=["#2e75b6", "#c55a11"], alpha=0.90)
    bars_r = ax.bar(x + w/2, random_vals, w,
                    label="랜덤 장애물 (generalization_test)",
                    color=["#9dc3e6", "#f4b183"], alpha=0.80)

    for bar, v in zip(bars_f, fixed_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, v in zip(bars_r, random_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_model, fontsize=10)
    ax.set_ylabel("도착률", fontsize=11)
    ax.set_title("eps=0 정책  |  고정 vs 랜덤 장애물", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.0, 1.15)
    ax.set_facecolor("#f8f9fb")

    # 오른쪽: 동일 환경(고정 장애물)에서 eps-greedy vs eps=0
    ax = axes[1]
    eg_vals  = [V5_EPSGREEDY_FIXED["v5a"], V5_EPSGREEDY_FIXED["v5b"]]
    e0_vals  = [arr_a, arr_b]

    bars_eg = ax.bar(x - w/2, eg_vals, w,
                     label="eps-greedy (학습 중 측정)",
                     color=["#5b9bd5", "#ed7d31"], alpha=0.70)
    bars_e0 = ax.bar(x + w/2, e0_vals, w,
                     label="eps=0 (순수 그리디, 이번 측정)",
                     color=["#2e75b6", "#c55a11"], alpha=0.90)

    for bar, v in zip(bars_eg, eg_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10, color="#555")
    for bar, v in zip(bars_e0, e0_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_model, fontsize=10)
    ax.set_ylabel("도착률", fontsize=11)
    ax.set_title("고정 장애물  |  eps-greedy vs eps=0 정책", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.0, 1.15)
    ax.set_facecolor("#f8f9fb")

    plt.tight_layout()
    out_path = RESULTS_DIR / "fixed_vs_random_epsilon0.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n그래프 저장 완료 → {out_path}")

# ── 메인 ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 72)
    print("고정 장애물  |  epsilon=0 순수 그리디 추론  |  V5-A vs V5-B")
    print(f"환경: OBSTACLES_8 고정 배치  |  {N_TEST_EPISODES} 에피소드")
    print("가중치 재학습 없음 - 파일 없으면 즉시 종료")
    print("=" * 72)

    print("\n[1단계] 가중치 로드")
    agent_a = _load_agent(V5A_WEIGHTS, "V5-A")
    agent_b = _load_agent(V5B_WEIGHTS, "V5-B")

    print("\n[2단계] 추론 실행")
    results_a = run_test(agent_a, "V5-A")
    results_b = run_test(agent_b, "V5-B")

    print("\n[3단계] 결과 정리")
    arr_a = sum(results_a) / len(results_a)
    arr_b = sum(results_b) / len(results_b)
    report(arr_a, arr_b)

    print("\n완료.")

if __name__ == "__main__":
    main()
