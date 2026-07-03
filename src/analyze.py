"""FRDDM-DQN 논문 수식 기반 상태 벡터 오차 분석.

attack_results.json의 최고 신뢰도 박스 중심 좌표에
이미지 스케일 변환 수식을 적용하여 D'OtoU 오차율과
theta'_o 오차를 epsilon별로 계산·시각화한다.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 경로 ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
IN_JSON     = RESULTS_DIR / "attack_results.json"
OUT_PNG     = RESULTS_DIR / "state_vector_error.png"

# ── FRDDM-DQN 파라미터 ─────────────────────────────────────────────────────
TAU          = 0.05
IMG_W        = 1000          # 이미지 폭 (px)
IMG_H        = 500           # 이미지 높이 (px)
# UAV 위치: 이미지 중앙 하단 (픽셀 → 스케일 좌표)
X_U = TAU * (IMG_W / 2)     # 0.05 × 500  =  25.0
Y_U = -TAU * IMG_H          # 0.05 × 500  = -25.0  (y축 반전)

EPSILONS = [0, 0.01, 0.03, 0.05, 0.07, 0.10]

# JSON에 저장된 key 형식: str(0) → "0", str(0.10) → "0.1"
EPS_KEYS = [str(e) for e in EPSILONS]


# ── 수식 ──────────────────────────────────────────────────────────────────
def to_state(x_px: float, y_px: float) -> tuple[float, float]:
    """픽셀 중심 좌표 → (D'OtoU, theta'_o[deg]).

    x'_o = tau * x_px
    y'_o = -tau * y_px          (이미지 y는 아래가 양수, 월드 좌표는 위가 양수)
    D'OtoU = ||U - O||₂
    theta'_o = atan2(y'_U - y'_o, x'_U - x'_o)
    """
    xo = TAU * x_px
    yo = -TAU * y_px
    dx = X_U - xo
    dy = Y_U - yo
    d     = math.sqrt(dx * dx + dy * dy)
    theta = math.degrees(math.atan2(dy, dx))
    return d, theta


def angle_diff(a: float, b: float) -> float:
    """두 각도 차이를 [0, 180] 범위로 반환."""
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff


# ── 메인 ──────────────────────────────────────────────────────────────────
def main() -> None:
    with open(IN_JSON, encoding="utf-8") as f:
        data = json.load(f)

    filenames = sorted(data.keys())

    # epsilon=0 기준 상태 벡터 (최고 신뢰도 박스 = 인덱스 0)
    ref_d:     dict[str, float] = {}
    ref_theta: dict[str, float] = {}
    for fn in filenames:
        centers = data[fn]["0"]["centers"]
        cx, cy  = centers[0]
        ref_d[fn], ref_theta[fn] = to_state(cx, cy)

    # ── epsilon별 오차 계산 ────────────────────────────────────────────────
    # per_d[fn][eps_idx], per_t[fn][eps_idx]
    per_d: dict[str, list] = {fn: [] for fn in filenames}
    per_t: dict[str, list] = {fn: [] for fn in filenames}

    for key in EPS_KEYS:
        for fn in filenames:
            centers = data[fn].get(key, {}).get("centers", [])
            if centers:
                cx, cy  = centers[0]
                d_adv, t_adv = to_state(cx, cy)
                d_err = abs(d_adv - ref_d[fn]) / ref_d[fn] * 100
                t_err = angle_diff(t_adv, ref_theta[fn])
            else:
                d_err = float("nan")
                t_err = float("nan")
            per_d[fn].append(d_err)
            per_t[fn].append(t_err)

    # 이미지 평균 (NaN 제외)
    avg_d = [float(np.nanmean([per_d[fn][j] for fn in filenames])) for j in range(len(EPSILONS))]
    avg_t = [float(np.nanmean([per_t[fn][j] for fn in filenames])) for j in range(len(EPSILONS))]

    # ── 표 출력 ────────────────────────────────────────────────────────────
    sep = "=" * 56
    print(sep)
    print(f"{'epsilon':>8} | {'D_OtoU err rate (%)':>22} | {'theta_o err (deg)':>18}")
    print("-" * 56)
    for eps, de, te in zip(EPSILONS, avg_d, avg_t):
        print(f"{eps:>8.2f} | {de:>22.4f} | {te:>18.4f}")
    print(sep)

    print("\n[이미지별 D_OtoU 오차율 (%)]")
    print(f"  {'':14}", " ".join(f"{e:>6.2f}" for e in EPSILONS))
    for fn in filenames:
        row = " ".join(f"{v:>6.2f}" if not np.isnan(v) else "   nan" for v in per_d[fn])
        print(f"  {fn:<14} {row}")

    print("\n[이미지별 theta_o 오차 (deg)]")
    print(f"  {'':14}", " ".join(f"{e:>6.2f}" for e in EPSILONS))
    for fn in filenames:
        row = " ".join(f"{v:>6.2f}" if not np.isnan(v) else "   nan" for v in per_t[fn])
        print(f"  {fn:<14} {row}")

    # ── 기준값 출력 ────────────────────────────────────────────────────────
    print(f"\n[UAV 위치] x'_U={X_U:.2f}, y'_U={Y_U:.2f}  (tau={TAU}, "
          f"image {IMG_W}x{IMG_H})")
    print("[epsilon=0 기준 상태 벡터]")
    for fn in filenames:
        print(f"  {fn}: D'OtoU={ref_d[fn]:.4f}, theta'_o={ref_theta[fn]:.4f} deg")

    # ── 그래프 ────────────────────────────────────────────────────────────
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    fig.suptitle(
        "FRDDM-DQN State Vector Error under FGSM Attack\n"
        rf"($\tau$={TAU}, image {IMG_W}$\times${IMG_H}, "
        rf"UAV at $x'_U$={X_U:.1f}, $y'_U$={Y_U:.1f})",
        fontsize=13, fontweight="bold",
    )

    # 개별 이미지 선
    for i, fn in enumerate(filenames):
        ax1.plot(EPSILONS, per_d[fn], "o--", color=colors[i],
                 alpha=0.45, linewidth=1.2, markersize=5, label=fn)
        ax2.plot(EPSILONS, per_t[fn], "o--", color=colors[i],
                 alpha=0.45, linewidth=1.2, markersize=5, label=fn)

    # 평균 (굵은 선)
    ax1.plot(EPSILONS, avg_d, "s-", color="black", linewidth=2.5,
             markersize=9, label="Average", zorder=5)
    ax2.plot(EPSILONS, avg_t, "s-", color="black", linewidth=2.5,
             markersize=9, label="Average", zorder=5)

    # 평균 레이블
    for eps, de in zip(EPSILONS, avg_d):
        ax1.annotate(f"{de:.2f}%", (eps, de),
                     textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8.5, fontweight="bold")
    for eps, te in zip(EPSILONS, avg_t):
        ax2.annotate(f"{te:.2f}", (eps, te),
                     textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8.5, fontweight="bold")

    ax1.set_ylabel("D'OtoU Error Rate (%)", fontsize=12)
    ax1.set_title("(a) Distance D'OtoU Error Rate", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.set_xlabel("Epsilon", fontsize=12)
    ax2.set_ylabel(r"$\theta'_o$ Error (degrees)", fontsize=12)
    ax2.set_title(r"(b) Bearing Angle $\theta'_o$ Error", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(EPSILONS)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"\n그래프 저장 -> {OUT_PNG}")


if __name__ == "__main__":
    main()
