"""DQN 비교 v4: 이진 CE 차단 vs 연속 CE 가중치 (8장애물/랜덤 환경).

실험 목적:
  env_test.py 검증 결과, 8장애물+랜덤 환경에서 순수 DQN 74% 도착.
  같은 환경에서 CE 처리 방식(이진 vs 연속)의 단독 효과를 격리 비교.

버전 A: 이진 차단
  - shift > 10px → 저장 안 함
  - 그 외 → 100% 저장 (v2-B와 동일 CE 전략, 단 RISK 패널티 제거)

버전 B: 연속 가중치
  - deposit_ratio = max(0.05, 1.0 - shift / 50.0)
  - shift 5px  → 0.90 / shift 15px → 0.70
  - shift 25px → 0.50 / shift 50px+ → 0.05
  - 확률적 deposit: random.random() < deposit_ratio

공통 조건:
  - 8장애물, 랜덤 시작/목적지 (env_test Env-2 동일)
  - RISK 패널티 없음 (밀집 환경 학습 붕괴 방지)
  - 회피 보너스 없음
  - PGD ε=0.03 공격 적용
  - 1000 에피소드
"""

from __future__ import annotations

import json
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

# ── 공유 컴포넌트 import ────────────────────────────────────────────────────
from dqn_comparison import DQNAgent, pgd_attack, MAX_STEPS
from env_test import DroneEnvN, OBSTACLES_8

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

CE_BINARY_THRESH  = 10.0   # 이진 차단 임계값 (px)
CE_SHIFT_MAX      = 50.0   # 연속 가중치 분모 (px)
EPISODES_V4       = 1000


# ══════════════════════════════════════════════════════════════════════════════
# CE deposit 비율 함수
# ══════════════════════════════════════════════════════════════════════════════

def _deposit_ratio_binary(shift: float) -> float:
    return 0.0 if shift > CE_BINARY_THRESH else 1.0


def _deposit_ratio_continuous(shift: float) -> float:
    return max(0.05, 1.0 - shift / CE_SHIFT_MAX)


# ══════════════════════════════════════════════════════════════════════════════
# 공통 학습 루틴
# ══════════════════════════════════════════════════════════════════════════════

def _train(
    deposit_fn,       # shift → deposit_ratio (0~1)
    label: str,
    is_probabilistic: bool,  # True: random() < ratio, False: ratio == 1.0 or 0.0
) -> tuple[list[float], list[float], dict]:
    """CE 방식만 다른 공통 DQN 학습 루틴.

    Args:
        deposit_fn:       shift → deposit_ratio
        label:            로그/그래프 표시 이름
        is_probabilistic: 연속 가중치 방식이면 True (확률적 deposit)

    Returns:
        arr_hist, col_hist, stats
    """
    env   = DroneEnvN(OBSTACLES_8, fallback_pos=[1.0, 1.0])
    agent = DQNAgent()

    arr_hist:  list[float] = []
    col_hist:  list[float] = []
    window_arr = deque(maxlen=100)
    window_col = deque(maxlen=100)

    total_steps    = 0
    ce_blocked     = 0
    valid_exp      = 0
    arrival_count  = 0
    ratio_sum      = 0.0   # 평균 deposit_ratio 계산용
    valid_exp_hist: list[int] = []

    print(f"\n[{label}] 학습 시작 (8장애물/랜덤, {EPISODES_V4}ep)")

    for ep in range(EPISODES_V4):
        state   = env.reset()
        arrived = False
        hit     = False

        for _ in range(MAX_STEPS):
            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)

            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)
            total_steps += 1

            # CE deposit 판정
            ratio     = deposit_fn(shift)
            ratio_sum += ratio
            if is_probabilistic:
                should_deposit = (random.random() < ratio)
            else:
                should_deposit = (ratio > 0.0)

            if should_deposit:
                D_obs_next, theta_obs_next = env.get_raw_obstacle()
                c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
                agent.push(c_state, action, reward, c_next, done)
                valid_exp += 1
            else:
                ce_blocked += 1

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
        valid_exp_hist.append(valid_exp)

        if (ep + 1) % 100 == 0:
            discard_rate = ce_blocked / total_steps if total_steps > 0 else 0.0
            print(f"  Ep {ep+1:4d} | 도착률={arr_hist[-1]:.3f}"
                  f" | 충돌률={col_hist[-1]:.3f}"
                  f" | ε={agent.epsilon:.3f}"
                  f" | CE폐기={ce_blocked}({discard_rate:.1%})"
                  f" | 유효경험={valid_exp:,}")

    avg_ratio = ratio_sum / total_steps if total_steps > 0 else 0.0
    stats = {
        "label":          label,
        "episodes":       EPISODES_V4,
        "total_steps":    total_steps,
        "ce_blocked":     ce_blocked,
        "valid_exp":      valid_exp,
        "arrival_count":  arrival_count,
        "final_arr":      float(arr_hist[-1]),
        "final_col":      float(col_hist[-1]),
        "mean_arr_last100": float(np.mean(arr_hist[-100:])),
        "peak_arr":       float(max(arr_hist)),
        "peak_ep":        int(np.argmax(arr_hist)) + 1,
        "discard_rate":   ce_blocked / total_steps if total_steps > 0 else 0.0,
        "avg_deposit_ratio": float(avg_ratio),
        "valid_exp_hist": valid_exp_hist,
    }
    return arr_hist, col_hist, stats


def train_v4a() -> tuple[list[float], list[float], dict]:
    """버전 A: 이진 차단 (shift > 10px → 저장 안 함)."""
    return _train(
        deposit_fn=_deposit_ratio_binary,
        label="V4-A (이진 CE 10px)",
        is_probabilistic=False,
    )


def train_v4b() -> tuple[list[float], list[float], dict]:
    """버전 B: 연속 가중치 (max(0.05, 1.0 - shift/50))."""
    return _train(
        deposit_fn=_deposit_ratio_continuous,
        label="V4-B (연속 CE 가중치)",
        is_probabilistic=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 결과 저장
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    arr_a: list[float], col_a: list[float], stats_a: dict,
    arr_b: list[float], col_b: list[float], stats_b: dict,
) -> None:
    """비교 표 출력 + 그래프·JSON 저장."""

    # ── 표 출력 ──────────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("v4 학습 결과 비교  (8장애물/랜덤 | RISK패널티 없음 | 회피보너스 없음)")
    print(sep)
    fmt = "{:<22} {:>22} {:>22}"
    print(fmt.format("항목", "V4-A (이진 CE 10px)", "V4-B (연속 CE 가중치)"))
    print("-" * 68)

    rows = [
        ("학습 에피소드",
         f"{stats_a['episodes']:,}", f"{stats_b['episodes']:,}"),
        ("총 경험 수집량",
         f"{stats_a['total_steps']:,}", f"{stats_b['total_steps']:,}"),
        ("CE 폐기 경험 수",
         f"{stats_a['ce_blocked']:,}", f"{stats_b['ce_blocked']:,}"),
        ("CE 폐기율",
         f"{stats_a['discard_rate']:.1%}", f"{stats_b['discard_rate']:.1%}"),
        ("평균 deposit_ratio",
         f"{stats_a['avg_deposit_ratio']:.3f}", f"{stats_b['avg_deposit_ratio']:.3f}"),
        ("유효 경험 수",
         f"{stats_a['valid_exp']:,}", f"{stats_b['valid_exp']:,}"),
        ("도착 에피소드 수",
         f"{stats_a['arrival_count']:,}", f"{stats_b['arrival_count']:,}"),
        ("─" * 20, "─" * 20, "─" * 20),
        ("최종 도착률 (100ep)",
         f"{stats_a['final_arr']:.4f}", f"{stats_b['final_arr']:.4f}"),
        ("최종 충돌률 (100ep)",
         f"{stats_a['final_col']:.4f}", f"{stats_b['final_col']:.4f}"),
        ("피크 도착률",
         f"{stats_a['peak_arr']:.4f} (Ep{stats_a['peak_ep']})",
         f"{stats_b['peak_arr']:.4f} (Ep{stats_b['peak_ep']})"),
    ]
    for r in rows:
        print(fmt.format(*r))
    print(sep)

    # 참조값
    print(f"\n비교 참조:")
    print(f"  - env_test Env-2 (CE 없음, 동일 환경): 최종 도착률 0.7400")
    print(f"  - v2-B (이진 CE 10px, 5장애물/랜덤, RISK패널티, 2000ep): 0.3422")

    # ── JSON ─────────────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "dqn_results_v4.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "v4a": {
                **{k: v for k, v in stats_a.items() if k != "valid_exp_hist"},
                "arrival_history":   [round(v, 4) for v in arr_a],
                "collision_history": [round(v, 4) for v in col_a],
            },
            "v4b": {
                **{k: v for k, v in stats_b.items() if k != "valid_exp_hist"},
                "arrival_history":   [round(v, 4) for v in arr_b],
                "collision_history": [round(v, 4) for v in col_b],
            },
            "reference": {
                "env_test_8obs_random_no_ce":  0.7400,
                "v2b_binary_ce_5obs_2000ep":   0.3422,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n수치 저장 → {out_json}")

    # ── 그래프 (3×1) ─────────────────────────────────────────────────────────
    eps = list(range(1, EPISODES_V4 + 1))
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    # ── (1) 도착률 학습 곡선 ─────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(eps, arr_a, color="steelblue",  linewidth=1.6, alpha=0.9,
             label=f"V4-A 이진CE 10px  (최종 {stats_a['final_arr']:.3f}, "
                   f"피크 {stats_a['peak_arr']:.3f}@Ep{stats_a['peak_ep']})")
    ax1.plot(eps, arr_b, color="darkorange", linewidth=1.6, alpha=0.9,
             label=f"V4-B 연속CE 가중치 (최종 {stats_b['final_arr']:.3f}, "
                   f"피크 {stats_b['peak_arr']:.3f}@Ep{stats_b['peak_ep']})")
    ax1.axhline(0.7400, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
                label="기준선: CE 없음 순수DQN (0.740)")
    ax1.set_ylabel("최근 100ep 도착률", fontsize=11)
    ax1.set_title(
        "DQN 비교 v4: 이진 CE vs 연속 CE 가중치\n"
        "(8장애물/랜덤 | RISK패널티 없음 | 회피보너스 없음 | PGD ε=0.03 | 1000ep)",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(fontsize=9.5, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, EPISODES_V4)
    ax1.set_ylim(-0.02, 1.02)

    # ── (2) 충돌률 학습 곡선 ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(eps, col_a, color="steelblue",  linewidth=1.6, alpha=0.9,
             label=f"V4-A 이진CE 10px  (최종 충돌률 {stats_a['final_col']:.3f})")
    ax2.plot(eps, col_b, color="darkorange", linewidth=1.6, alpha=0.9,
             label=f"V4-B 연속CE 가중치 (최종 충돌률 {stats_b['final_col']:.3f})")
    ax2.set_ylabel("최근 100ep 충돌률", fontsize=11)
    ax2.set_title("충돌률 비교", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9.5, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, EPISODES_V4)
    ax2.set_ylim(-0.02, 1.02)

    # ── (3) 유효 경험 누적 ───────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(eps, stats_a["valid_exp_hist"], color="steelblue",  linewidth=1.6, alpha=0.9,
             label=f"V4-A 이진CE  (유효경험 {stats_a['valid_exp']:,}개)")
    ax3.plot(eps, stats_b["valid_exp_hist"], color="darkorange", linewidth=1.6, alpha=0.9,
             label=f"V4-B 연속CE  (유효경험 {stats_b['valid_exp']:,}개)")
    ax3.set_xlabel("에피소드", fontsize=11)
    ax3.set_ylabel("누적 유효 경험 수", fontsize=11)
    ax3.set_title("누적 유효 경험 수 (리플레이 메모리에 실제 저장된 경험)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9.5, loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, EPISODES_V4)

    plt.tight_layout()
    out_png = RESULTS_DIR / "dqn_comparison_v4.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"그래프 저장 → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("DQN 비교 v4: 이진 CE 차단 vs 연속 CE 가중치")
    print("환경: 8장애물, 랜덤 시작/목적지 (env_test Env-2 동일)")
    print("CE 비교 조건 외 모든 설정 동일")
    print("=" * 60)

    arr_a, col_a, stats_a = train_v4a()
    arr_b, col_b, stats_b = train_v4b()
    save_results(arr_a, col_a, stats_a, arr_b, col_b, stats_b)
    print("\n완료.")


if __name__ == "__main__":
    main()
