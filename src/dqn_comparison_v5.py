"""DQN 비교 v5: 시간 페널티 + 거리 기반 보상 추가 (8장애물/랜덤 환경).

보상 함수 변경:
  v4 기준에서 비종료 스텝(reward=0)에 아래 두 항목 추가:
    - 시간 페널티: -0.01 / 스텝  → 빙빙 돌기 억제, 효율적 경로 유도
    - 거리 보상:  +0.01 (목적지에 가까워짐) / -0.01 (멀어짐) → 밀집 환경 dense 신호

최종 보상 구조 (비종료 스텝):
    r = -0.01(시간) + {+0.01 | -0.01}(거리)
    → 목적지 방향 이동: 0.00 / 역방향 이동: -0.02

종료 보상 유지:
    목적지 도달: +1.0 / 충돌·이탈: -1.0

버전 A: 이진 CE 차단 (shift > 10px → 저장 안 함)
버전 B: 연속 CE 가중치 (max(0.05, 1.0 - shift/50))

환경: 8장애물, 랜덤 시작/목적지 (v4 동일)
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

# ── 공유 컴포넌트 import ────────────────────────────────────────────────────
from dqn_comparison import DQNAgent, pgd_attack, MAX_STEPS
from env_test import DroneEnvN, OBSTACLES_8

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

CE_BINARY_THRESH = 10.0
CE_SHIFT_MAX     = 50.0
EPISODES_V5      = 1000

# ── 새 보상 항목 ─────────────────────────────────────────────────────────────
TIME_PENALTY   = -0.01   # 비종료 스텝 매 스텝 적용
DIST_APPROACH  = +0.01   # 목적지에 가까워진 경우
DIST_RECEDE    = -0.01   # 목적지에서 멀어진 경우


# ══════════════════════════════════════════════════════════════════════════════
# 공통 학습 루틴
# ══════════════════════════════════════════════════════════════════════════════

def _train(
    deposit_fn,
    label: str,
    is_probabilistic: bool,
) -> tuple[list[float], list[float], dict]:
    """시간 페널티·거리 보상이 추가된 DQN 학습.

    Args:
        deposit_fn:       shift → deposit_ratio (0~1)
        label:            로그/그래프 표시 이름
        is_probabilistic: 연속 가중치면 True

    Returns:
        arr_hist, col_hist, stats
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
    valid_exp_hist: list[int] = []

    print(f"\n[{label}] 학습 시작 (8장애물/랜덤, {EPISODES_V5}ep, "
          f"시간패널티={TIME_PENALTY} / 거리보상=+/-{abs(DIST_APPROACH)})")

    for ep in range(EPISODES_V5):
        state   = env.reset()
        arrived = False
        hit     = False

        for _ in range(MAX_STEPS):
            # 스텝 전 목적지 거리 기록
            prev_dist = math.dist(env.pos, env.goal)

            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)

            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)
            total_steps += 1

            # ── 보상 추가 (비종료 스텝만) ─────────────────────────────────
            if not (info["goal_reached"] or info["collision"] or info["out_of_bounds"]):
                # 시간 페널티
                reward += TIME_PENALTY

                # 거리 기반 보상
                curr_dist = math.dist(env.pos, env.goal)
                if curr_dist < prev_dist:
                    reward += DIST_APPROACH
                elif curr_dist > prev_dist:
                    reward += DIST_RECEDE

            # ── CE deposit 판정 ────────────────────────────────────────────
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
        "label":             label,
        "episodes":          EPISODES_V5,
        "total_steps":       total_steps,
        "ce_blocked":        ce_blocked,
        "valid_exp":         valid_exp,
        "arrival_count":     arrival_count,
        "final_arr":         float(arr_hist[-1]),
        "final_col":         float(col_hist[-1]),
        "mean_arr_last100":  float(np.mean(arr_hist[-100:])),
        "peak_arr":          float(max(arr_hist)),
        "peak_ep":           int(np.argmax(arr_hist)) + 1,
        "discard_rate":      ce_blocked / total_steps if total_steps > 0 else 0.0,
        "avg_deposit_ratio": float(avg_ratio),
        "valid_exp_hist":    valid_exp_hist,
    }
    return arr_hist, col_hist, stats, agent


def _deposit_binary(shift: float) -> float:
    return 0.0 if shift > CE_BINARY_THRESH else 1.0


def _deposit_continuous(shift: float) -> float:
    return max(0.05, 1.0 - shift / CE_SHIFT_MAX)


def train_v5a() -> tuple[list[float], list[float], dict]:
    arr_hist, col_hist, stats, agent = _train(_deposit_binary,     "V5-A (이진 CE 10px)",  is_probabilistic=False)
    torch.save(agent.policy_net.state_dict(), RESULTS_DIR / "v5a_policy.pth")
    return arr_hist, col_hist, stats


def train_v5b() -> tuple[list[float], list[float], dict]:
    arr_hist, col_hist, stats, agent = _train(_deposit_continuous, "V5-B (연속 CE 가중치)", is_probabilistic=True)
    torch.save(agent.policy_net.state_dict(), RESULTS_DIR / "v5b_policy.pth")
    return arr_hist, col_hist, stats


# ══════════════════════════════════════════════════════════════════════════════
# 결과 저장 (v4 비교 포함)
# ══════════════════════════════════════════════════════════════════════════════

def _load_v4_stats() -> tuple[dict | None, dict | None]:
    """v4 JSON에서 비교 통계를 읽는다. 파일 없으면 None 반환."""
    p = RESULTS_DIR / "dqn_results_v4.json"
    if not p.exists():
        return None, None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("v4a"), d.get("v4b")


def save_results(
    arr_a: list[float], col_a: list[float], stats_a: dict,
    arr_b: list[float], col_b: list[float], stats_b: dict,
) -> None:
    """비교 표 출력 + 그래프·JSON 저장."""

    v4a, v4b = _load_v4_stats()

    # ── 표 출력 ──────────────────────────────────────────────────────────────
    sep = "=" * 84
    print(f"\n{sep}")
    print("v5 학습 결과 비교  (8장애물/랜덤 | RISK패널티 없음 | 회피보너스 없음)")
    print(f"보상 추가: 시간패널티 {TIME_PENALTY}/step, "
          f"거리보상 +/-{abs(DIST_APPROACH)}/step")
    print(sep)
    fmt = "{:<24} {:>14} {:>14} {:>14} {:>14}"
    print(fmt.format("항목", "V4-A(이진CE)", "V5-A(이진CE)", "V4-B(연속CE)", "V5-B(연속CE)"))
    print("-" * 84)

    def v4_val(d: dict | None, key: str, fmt_str: str = "{:.4f}") -> str:
        if d is None:
            return "N/A"
        v = d.get(key)
        return fmt_str.format(v) if v is not None else "N/A"

    rows = [
        ("학습 에피소드",
         v4_val(v4a, "episodes", "{:,}"),
         f"{stats_a['episodes']:,}",
         v4_val(v4b, "episodes", "{:,}"),
         f"{stats_b['episodes']:,}"),
        ("총 경험 수집량",
         v4_val(v4a, "total_steps", "{:,}"),
         f"{stats_a['total_steps']:,}",
         v4_val(v4b, "total_steps", "{:,}"),
         f"{stats_b['total_steps']:,}"),
        ("CE 폐기율",
         v4_val(v4a, "discard_rate", "{:.1%}"),
         f"{stats_a['discard_rate']:.1%}",
         v4_val(v4b, "discard_rate", "{:.1%}"),
         f"{stats_b['discard_rate']:.1%}"),
        ("유효 경험 수",
         v4_val(v4a, "valid_exp", "{:,}"),
         f"{stats_a['valid_exp']:,}",
         v4_val(v4b, "valid_exp", "{:,}"),
         f"{stats_b['valid_exp']:,}"),
        ("도착 에피소드 수",
         v4_val(v4a, "arrival_count", "{:,}"),
         f"{stats_a['arrival_count']:,}",
         v4_val(v4b, "arrival_count", "{:,}"),
         f"{stats_b['arrival_count']:,}"),
        ("─" * 22, "─" * 12, "─" * 12, "─" * 12, "─" * 12),
        ("최종 도착률 (100ep)",
         v4_val(v4a, "final_arr"),
         f"{stats_a['final_arr']:.4f}",
         v4_val(v4b, "final_arr"),
         f"{stats_b['final_arr']:.4f}"),
        ("최종 충돌률 (100ep)",
         v4_val(v4a, "final_col"),
         f"{stats_a['final_col']:.4f}",
         v4_val(v4b, "final_col"),
         f"{stats_b['final_col']:.4f}"),
        ("피크 도착률",
         v4_val(v4a, "peak_arr"),
         f"{stats_a['peak_arr']:.4f} (Ep{stats_a['peak_ep']})",
         v4_val(v4b, "peak_arr"),
         f"{stats_b['peak_arr']:.4f} (Ep{stats_b['peak_ep']})"),
    ]
    for r in rows:
        print(fmt.format(*r))
    print(sep)

    # v4 대비 개선 여부
    if v4a and v4b:
        delta_a = stats_a["final_arr"] - v4a["final_arr"]
        delta_b = stats_b["final_arr"] - v4b["final_arr"]
        sign_a  = "+" if delta_a >= 0 else ""
        sign_b  = "+" if delta_b >= 0 else ""
        print(f"\n보상 추가 효과 (v4 → v5):")
        print(f"  V4-A → V5-A: {v4a['final_arr']:.4f} → {stats_a['final_arr']:.4f}"
              f"  ({sign_a}{delta_a:.4f})")
        print(f"  V4-B → V5-B: {v4b['final_arr']:.4f} → {stats_b['final_arr']:.4f}"
              f"  ({sign_b}{delta_b:.4f})")
    print(f"\n※ 비교 기준: env_test Env-2 (CE 없음, 순수DQN) = 0.7400")

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "dqn_results_v5.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "reward_config": {
                "time_penalty":  TIME_PENALTY,
                "dist_approach": DIST_APPROACH,
                "dist_recede":   DIST_RECEDE,
                "note":          "비종료 스텝에만 적용 (terminal +1/-1 유지)",
            },
            "v5a": {
                **{k: v for k, v in stats_a.items() if k != "valid_exp_hist"},
                "arrival_history":   [round(v, 4) for v in arr_a],
                "collision_history": [round(v, 4) for v in col_a],
            },
            "v5b": {
                **{k: v for k, v in stats_b.items() if k != "valid_exp_hist"},
                "arrival_history":   [round(v, 4) for v in arr_b],
                "collision_history": [round(v, 4) for v in col_b],
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n수치 저장 → {out_json}")

    # ── 그래프 (2×2) ─────────────────────────────────────────────────────────
    eps = list(range(1, EPISODES_V5 + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "DQN 비교 v5: 시간 페널티(-0.01/step) + 거리 보상(+/-0.01/step) 추가\n"
        "(8장애물/랜덤 | RISK패널티 없음 | 회피보너스 없음 | PGD ε=0.03 | 1000ep)",
        fontsize=11, fontweight="bold",
    )

    # v4 히스토리 로드 (있을 경우)
    v4a_arr = v4a.get("arrival_history", []) if v4a else []
    v4b_arr = v4b.get("arrival_history", []) if v4b else []

    # ── (좌상) V*-A 도착률 비교 ───────────────────────────────────────────────
    ax = axes[0, 0]
    if v4a_arr:
        ax.plot(range(1, len(v4a_arr) + 1), v4a_arr,
                color="lightsteelblue", linewidth=1.2, alpha=0.8, linestyle="--",
                label=f"V4-A 이진CE  (최종 {v4a['final_arr']:.3f})")
    ax.plot(eps, arr_a,
            color="steelblue", linewidth=1.8, alpha=0.9,
            label=f"V5-A 이진CE  (최종 {stats_a['final_arr']:.3f}, "
                  f"피크 {stats_a['peak_arr']:.3f}@Ep{stats_a['peak_ep']})")
    ax.axhline(0.7400, color="gray", linestyle=":", linewidth=1.0, alpha=0.6,
               label="기준 (CE없음 순수DQN 0.740)")
    ax.set_title("이진 CE 10px 도착률 (V4 vs V5)", fontsize=10, fontweight="bold")
    ax.set_ylabel("최근 100ep 도착률")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, EPISODES_V5)
    ax.set_ylim(-0.02, 1.02)

    # ── (우상) V*-B 도착률 비교 ───────────────────────────────────────────────
    ax = axes[0, 1]
    if v4b_arr:
        ax.plot(range(1, len(v4b_arr) + 1), v4b_arr,
                color="moccasin", linewidth=1.2, alpha=0.8, linestyle="--",
                label=f"V4-B 연속CE  (최종 {v4b['final_arr']:.3f})")
    ax.plot(eps, arr_b,
            color="darkorange", linewidth=1.8, alpha=0.9,
            label=f"V5-B 연속CE  (최종 {stats_b['final_arr']:.3f}, "
                  f"피크 {stats_b['peak_arr']:.3f}@Ep{stats_b['peak_ep']})")
    ax.axhline(0.7400, color="gray", linestyle=":", linewidth=1.0, alpha=0.6,
               label="기준 (CE없음 순수DQN 0.740)")
    ax.set_title("연속 CE 가중치 도착률 (V4 vs V5)", fontsize=10, fontweight="bold")
    ax.set_ylabel("최근 100ep 도착률")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, EPISODES_V5)
    ax.set_ylim(-0.02, 1.02)

    # ── (좌하) V5 도착률 직접 비교 ────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(eps, arr_a, color="steelblue",  linewidth=1.6, alpha=0.9,
            label=f"V5-A 이진CE  (최종 {stats_a['final_arr']:.3f})")
    ax.plot(eps, arr_b, color="darkorange", linewidth=1.6, alpha=0.9,
            label=f"V5-B 연속CE  (최종 {stats_b['final_arr']:.3f})")
    ax.set_title("V5-A vs V5-B 도착률", fontsize=10, fontweight="bold")
    ax.set_xlabel("에피소드")
    ax.set_ylabel("최근 100ep 도착률")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, EPISODES_V5)
    ax.set_ylim(-0.02, 1.02)

    # ── (우하) V5 충돌률 직접 비교 ────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(eps, col_a, color="steelblue",  linewidth=1.6, alpha=0.9,
            label=f"V5-A 이진CE  (최종 {stats_a['final_col']:.3f})")
    ax.plot(eps, col_b, color="darkorange", linewidth=1.6, alpha=0.9,
            label=f"V5-B 연속CE  (최종 {stats_b['final_col']:.3f})")
    ax.set_title("V5-A vs V5-B 충돌률", fontsize=10, fontweight="bold")
    ax.set_xlabel("에피소드")
    ax.set_ylabel("최근 100ep 충돌률")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, EPISODES_V5)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    out_png = RESULTS_DIR / "dqn_comparison_v5.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"그래프 저장 → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("DQN 비교 v5: 시간 페널티 + 거리 기반 보상 추가")
    print(f"환경: 8장애물 랜덤 (v4 동일)")
    print(f"보상 변경: 비종료 스텝마다 {TIME_PENALTY}(시간)"
          f" + {DIST_APPROACH}/{DIST_RECEDE}(거리)")
    print("=" * 60)

    arr_a, col_a, stats_a = train_v5a()
    arr_b, col_b, stats_b = train_v5b()
    save_results(arr_a, col_a, stats_a, arr_b, col_b, stats_b)
    print("\n완료.")


if __name__ == "__main__":
    main()
