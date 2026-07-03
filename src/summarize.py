"""종합 결과 그래프 생성 스크립트.

attack_results.json에서 실험 데이터를 읽고
FRDDM-DQN 수식을 재적용하여 4개 서브플롯을 하나의 PNG로 출력한다.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 한글 폰트 (Windows) ────────────────────────────────────────────────────
plt.rcParams["font.family"]       = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False   # 마이너스 부호 깨짐 방지

# ── 경로 ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
RESULTS_DIR  = ROOT / "results"
IN_JSON      = RESULTS_DIR / "attack_results.json"
OUT_PNG      = RESULTS_DIR / "final_summary.png"

# ── FRDDM-DQN 파라미터 ─────────────────────────────────────────────────────
TAU   = 0.05
IMG_W = 1000
IMG_H = 500
X_U   = TAU * (IMG_W / 2)   #  25.0
Y_U   = -TAU * IMG_H         # -25.0

EPSILONS  = [0, 0.01, 0.03, 0.05, 0.07, 0.10]
EPS_KEYS  = [str(e) for e in EPSILONS]
EPS_LABEL = [str(e) for e in EPSILONS]

COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]


# ── 수식 ──────────────────────────────────────────────────────────────────
def to_state(cx: float, cy: float) -> tuple[float, float]:
    xo, yo = TAU * cx, -TAU * cy
    d      = math.sqrt((X_U - xo) ** 2 + (Y_U - yo) ** 2)
    theta  = math.degrees(math.atan2(Y_U - yo, X_U - xo))
    return d, theta

def angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff


# ── 데이터 로드 및 계산 ───────────────────────────────────────────────────
with open(IN_JSON, encoding="utf-8") as f:
    raw = json.load(f)

filenames = sorted(raw.keys())
n_img     = len(filenames)
n_eps     = len(EPSILONS)

# 이미지별 × epsilon별 → [n_img][n_eps]
bbox_err  = {fn: [] for fn in filenames}   # 픽셀 중심 오차
d_err_pct = {fn: [] for fn in filenames}   # D'OtoU 오차율 (%)
t_err_deg = {fn: [] for fn in filenames}   # theta 오차 (도)

for fn in filenames:
    # epsilon=0 기준값
    ref_center   = raw[fn]["0"]["centers"][0]
    ref_d, ref_t = to_state(*ref_center)

    for key in EPS_KEYS:
        entry   = raw[fn].get(key, {})
        centers = entry.get("centers", [])

        # (1) 픽셀 중심 오차: attack.py가 저장한 nearest-neighbor 평균값 그대로 사용
        px_raw = entry.get("center_error_px")
        px_err = float(px_raw) if px_raw is not None else float("nan")

        # (2) 상태 벡터 오차: 최고 신뢰도 박스 기준으로 재계산
        if centers:
            cx, cy       = centers[0]
            d_adv, t_adv = to_state(cx, cy)
            de = abs(d_adv - ref_d) / ref_d * 100 if ref_d > 0 else 0.0
            te = angle_diff(t_adv, ref_t)
        else:
            de = te = float("nan")

        bbox_err[fn].append(px_err)
        d_err_pct[fn].append(de)
        t_err_deg[fn].append(te)

# 5장 이미지 평균
avg_bbox = [float(np.nanmean([bbox_err[fn][j]  for fn in filenames])) for j in range(n_eps)]
avg_d    = [float(np.nanmean([d_err_pct[fn][j] for fn in filenames])) for j in range(n_eps)]
avg_t    = [float(np.nanmean([t_err_deg[fn][j] for fn in filenames])) for j in range(n_eps)]


# ══════════════════════════════════════════════════════════════════════════
# 그래프 렌더링
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "FGSM 공격이 FRDDM-DQN 상태 벡터에 미치는 영향",
    fontsize=17, fontweight="bold", y=0.98,
)

# 2×2 그리드: 왼쪽 두 칸은 넓게, 오른쪽 파이차트는 정사각형 유지
gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                      left=0.07, right=0.97, top=0.92, bottom=0.07)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])


def _line_plot(ax, per_img_data, avg_data, ylabel, title, annotate_fmt="{:.2f}"):
    """이미지별 점선 + 평균 실선 공통 렌더러."""
    for i, fn in enumerate(filenames):
        short = fn.replace("vehicle_", "v").replace(".png", "")
        ax.plot(EPSILONS, per_img_data[fn], "o--", color=COLORS[i],
                alpha=0.40, linewidth=1.3, markersize=5, label=short)

    ax.plot(EPSILONS, avg_data, "s-", color="black",
            linewidth=2.6, markersize=9, label="평균", zorder=5)

    for eps, val in zip(EPSILONS, avg_data):
        if not math.isnan(val):
            ax.annotate(
                annotate_fmt.format(val),
                (eps, val),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8.5, fontweight="bold",
            )

    ax.set_xticks(EPSILONS)
    ax.set_xlim(-0.005, 0.11)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Epsilon", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left", ncol=2,
              framealpha=0.7, columnspacing=0.8, handlelength=1.5)
    ax.grid(True, alpha=0.3)


# ── 서브플롯 1: bbox 중심 오차 (픽셀) ────────────────────────────────────
_line_plot(
    ax1, bbox_err, avg_bbox,
    ylabel="중심 좌표 오차 (픽셀)",
    title="① 바운딩 박스 중심 좌표 오차",
    annotate_fmt="{:.1f}",
)

# ── 서브플롯 2: D'OtoU 오차율 (%) ────────────────────────────────────────
_line_plot(
    ax2, d_err_pct, avg_d,
    ylabel="D\'OtoU 오차율 (%)",
    title="② D\'OtoU 오차율",
    annotate_fmt="{:.2f}%",
)

# ── 서브플롯 3: theta'_o 오차 (도) ───────────────────────────────────────
_line_plot(
    ax3, t_err_deg, avg_t,
    ylabel="theta'_o 오차 (도)",
    title="③ 방위각 theta'_o 오차",
    annotate_fmt="{:.2f}°",
)

# ── 서브플롯 4: DDM 경험 분류 비율 파이차트 ──────────────────────────────
# 예치율: RE=1.0, DE=0.8, SE=0.2, CE=0.0
# 정규화(총합 2.0 기준): RE=50%, DE=40%, SE=10%, CE=0%
ddm_labels  = ["RE\n결과 경험", "DE\n위험 경험", "SE\n안전 경험"]
ddm_values  = [1.0, 0.8, 0.2]                       # CE 제외 (0.0)
ddm_pct     = [v / sum(ddm_values) * 100 for v in ddm_values]
ddm_colors  = ["#2196F3", "#FF9800", "#4CAF50"]
ddm_explode = [0.04, 0.04, 0.04]

wedges, texts, autotexts = ax4.pie(
    ddm_values,
    labels=None,
    colors=ddm_colors,
    explode=ddm_explode,
    autopct="%1.0f%%",
    pctdistance=0.72,
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.8),  # 도넛
    textprops=dict(fontsize=10),
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
    at.set_color("white")

# 도넛 중앙 텍스트
ax4.text(0, 0.10, "DDM\n예치율", ha="center", va="center",
         fontsize=11, fontweight="bold", color="#333333")

# 범례 (CE 포함)
legend_handles = [
    mpatches.Patch(color=ddm_colors[0], label=f"RE  예치율 100% ({ddm_pct[0]:.0f}%)"),
    mpatches.Patch(color=ddm_colors[1], label=f"DE  예치율  80% ({ddm_pct[1]:.0f}%)"),
    mpatches.Patch(color=ddm_colors[2], label=f"SE  예치율  20% ({ddm_pct[2]:.0f}%)"),
    mpatches.Patch(color="#BDBDBD",     label="CE  예치율   0%  ← 저장 차단"),
]
ax4.legend(handles=legend_handles, loc="lower center",
           bbox_to_anchor=(0.5, -0.18), fontsize=9,
           framealpha=0.85, edgecolor="#CCCCCC")

# CE 차단 강조 텍스트
ax4.text(0, -0.72,
         "CE (오염 경험)  →  pCE = 0.0  →  리플레이 메모리 저장 차단",
         ha="center", va="center", fontsize=9.5,
         color="#D32F2F", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFEBEE",
                   edgecolor="#EF9A9A", linewidth=1.2))

ax4.set_title("④ 개선 DDM 경험 분류 예치율", fontsize=12,
              fontweight="bold", pad=12)


# ── 공통 하단 설명 ─────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    f"모델: Faster R-CNN (MobileNet V3 Large FPN) | 에폭: 3 | "
    f"τ = {TAU} | 이미지 크기: {IMG_W}×{IMG_H} | "
    f"UAV 위치: x'_U = {X_U:.1f}, y'_U = {Y_U:.1f}",
    ha="center", fontsize=9, color="#555555",
    style="italic",
)

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장 완료 → {OUT_PNG}")
