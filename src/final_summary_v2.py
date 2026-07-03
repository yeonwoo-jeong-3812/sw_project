"""FRDDM-DQN 최종 결과 요약 그림 생성 — V2 (V6~V9 결과 추가).

final_summary.py 원본은 수정하지 않는다. 6개 패널 중 아래 2개만 확장:
  (3) 임계값 전략별 도착률 막대그래프 — V6~V9 추가
  (4) 학습 곡선 — V5-A/V5-B 2줄 대신 V5-B/V6/V7/V8/V9 5줄

(1) FGSM/PGD bbox 오차, (2) PGD 탐지 실패율, (5) confusion matrix,
(6) 5개 시나리오 결과는 원본 코드 그대로 유지한다.

V6~V9 도착률 값은 ce_evaluation_v2.py의 load_threshold_data()와 동일한
매핑 규칙을 따른다 (threshold_summary_table_v2.png와 동일 출처/로직):
  - V6/V7/V8: 각 json의 final_arr (학습 중 최종 도착률)
  - V9 (패널3, 막대그래프): eval_a_final (항상 공격 조건, 조건 A) —
    나머지 모든 막대가 항상 공격 조건에서 측정되었으므로 같은 조건끼리
    비교하기 위함.
  - V9 (패널4, 학습 곡선): 학습 중 arrival_history/final_arr을 그대로
    사용 — 이 학습 곡선은 확률적 공격(p=0.5) 조건에서 나온 것이므로,
    패널3의 eval_a_final(조건 A, 즉 100% 공격 평가)과는 다른 조건의
    수치이다. 두 곳 모두 우연히 0.78로 같게 나왔지만 의미가 다르므로
    패널4에 별도 각주로 명시한다.

기존 결과 파일만 재사용 (새 실험 없음):
  results/attack_results.json        → FGSM bbox 오차
  results/attack_results_pgd.json    → PGD  bbox 오차
  results/detection_failure.json     → PGD 탐지 실패율
  results/dqn_results_v5.json        → V5 학습 곡선/최종 도착률
  results/dqn_results_v6.json        → V6 학습 곡선/최종 도착률
  results/dqn_results_v7.json        → V7 학습 곡선/최종 도착률
  results/dqn_results_v8.json        → V8 학습 곡선/최종 도착률
  results/dqn_results_v9.json        → V9 학습 곡선/최종 도착률 + eval_a_final
  (CM·시나리오 결과는 기존 수치 직접 사용)

참조:
  results/flight_animation_v5.gif    → V5 비행 경로 시각화
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

_prefer = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "Gulim"]
_avail  = {f.name for f in fm.fontManager.ttflist}
for _f in _prefer:
    if _f in _avail:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = "results"
IMAGES = ["vehicle_01.png", "vehicle_02.png", "vehicle_03.png",
          "vehicle_04.png", "vehicle_05.png"]
EPSILONS = [0, 0.01, 0.03, 0.05, 0.07, 0.1]
EPS_LABELS = ["0", "0.01", "0.03", "0.05", "0.07", "0.1"]


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(name: str) -> dict:
    with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _eps_key(eps) -> str:
    return "0" if eps == 0 else str(eps)


def mean_shift_per_eps(atk_data: dict) -> list[float]:
    """각 epsilon 값에 대해 5장 이미지의 center_error_px 평균을 반환한다."""
    means = []
    for eps in EPSILONS:
        key = _eps_key(eps)
        vals = [
            float(atk_data[img][key]["center_error_px"])
            for img in IMAGES
            if img in atk_data and key in atk_data[img]
        ]
        means.append(float(np.mean(vals)) if vals else 0.0)
    return means


def rolling_mean(arr: list[float], window: int = 50) -> list[float]:
    out = []
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(arr[lo: i + 1])))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 메인 그리기
# ─────────────────────────────────────────────────────────────────────────────

def build_final_summary() -> None:
    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    fgsm_data = _load_json("attack_results.json")
    pgd_data  = _load_json("attack_results_pgd.json")
    det_data  = _load_json("detection_failure.json")
    v5_data   = _load_json("dqn_results_v5.json")
    v6_data   = _load_json("dqn_results_v6.json")["v6"]
    v7_data   = _load_json("dqn_results_v7.json")["v7"]
    v8_data   = _load_json("dqn_results_v8.json")["v8"]
    v9_data   = _load_json("dqn_results_v9.json")["v9"]

    fgsm_shifts = mean_shift_per_eps(fgsm_data)
    pgd_shifts  = mean_shift_per_eps(pgd_data)

    pgd_miss  = det_data["pgd"]["miss_rate"]
    pgd_ghost = det_data["pgd"]["ghost_rate"]
    pgd_total = det_data["pgd"]["total_failure_rate"]

    v5b_hist = v5_data["v5b"]["arrival_history"]
    v5b_smooth = rolling_mean(v5b_hist, 50)
    episodes = list(range(1, len(v5b_hist) + 1))

    # ── Figure 구성 ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#f5f6fa")

    gs = fig.add_gridspec(
        2, 3,
        hspace=0.46, wspace=0.38,
        left=0.06, right=0.97,
        top=0.88, bottom=0.07,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    _PANEL_STYLE = dict(facecolor="#ffffff", edgecolor="#cccccc", linewidth=0.8)
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#cccccc")

    # ── 패널 번호 레이블 ──────────────────────────────────────────────────────
    for idx, (ax, lbl) in enumerate(zip(
        [ax1, ax2, ax3, ax4, ax5, ax6],
        ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)"],
    )):
        ax.text(-0.12, 1.06, lbl, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="#2c3e50")

    x_idx = np.arange(len(EPSILONS))

    # ══════════════════════════════════════════════════════════════════════════
    # 서브플롯 1: FGSM vs PGD 바운딩 박스 오차 (원본 그대로)
    # ══════════════════════════════════════════════════════════════════════════
    ax1.plot(x_idx, fgsm_shifts, "o-", color="#e67e22", linewidth=2,
             markersize=6, label="FGSM")
    ax1.plot(x_idx, pgd_shifts,  "s-", color="#2980b9", linewidth=2,
             markersize=6, label="PGD")
    ax1.fill_between(x_idx, fgsm_shifts, alpha=0.12, color="#e67e22")
    ax1.fill_between(x_idx, pgd_shifts,  alpha=0.12, color="#2980b9")

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(EPS_LABELS, fontsize=9)
    ax1.set_xlabel("epsilon (ε)", fontsize=10)
    ax1.set_ylabel("평균 bbox 오차 (px)", fontsize=10)
    ax1.set_title("FGSM vs PGD\n바운딩 박스 중심 오차", fontsize=11, fontweight="bold", pad=8)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_ylim(bottom=0)

    # PGD가 FGSM보다 평균적으로 몇 배 큰지 표시
    ratio = np.mean([p / f if f > 0 else 0 for p, f in
                     zip(pgd_shifts[1:], fgsm_shifts[1:])])
    ax1.text(0.98, 0.05,
             f"PGD : FGSM\n~{ratio:.1f}x 오차",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=8.5, color="#2980b9",
             bbox=dict(boxstyle="round,pad=0.3", fc="#eaf4fb", alpha=0.9))

    # ══════════════════════════════════════════════════════════════════════════
    # 서브플롯 2: PGD 탐지 실패율 (Miss / Ghost / Total) (원본 그대로)
    # ══════════════════════════════════════════════════════════════════════════
    ax2.plot(x_idx, pgd_miss,  "^--", color="#c0392b", linewidth=1.8,
             markersize=6, label="Miss율")
    ax2.plot(x_idx, pgd_ghost, "o-",  color="#8e44ad", linewidth=2,
             markersize=6, label="Ghost율")
    ax2.plot(x_idx, pgd_total, "s-",  color="#2c3e50", linewidth=2.2,
             markersize=6, label="전체 실패율")
    ax2.fill_between(x_idx, pgd_total, alpha=0.07, color="#2c3e50")

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(EPS_LABELS, fontsize=9)
    ax2.set_xlabel("epsilon (ε)", fontsize=10)
    ax2.set_ylabel("실패율", fontsize=10)
    ax2.set_ylim(-0.05, 1.10)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax2.set_title("PGD 공격\n탐지 실패율 (Miss / Ghost / Total)", fontsize=11, fontweight="bold", pad=8)
    ax2.legend(fontsize=9, loc="center right")
    ax2.grid(True, alpha=0.3, linewidth=0.5)

    ax2.annotate("eps>=0.03\nTotal~100%",
                 xy=(2, pgd_total[2]), xytext=(3.2, 0.72),
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
                 fontsize=8, color="#2c3e50")

    # ══════════════════════════════════════════════════════════════════════════
    # 서브플롯 3: 임계값 전략별 도착률 (V6~V9 추가)
    # ══════════════════════════════════════════════════════════════════════════
    labels3 = [
        "기준선\nCE없음", "V4-A\n이진CE", "V4-B\n연속CE", "V5-A\n이진CE", "V5-B\n연속CE",
        "V6\n연속CE", "V7\nCE분류", "V8\n연속생존", "V9\np=0.5(A)",
    ]
    arr_vals = [
        0.74, 0.31, 0.65, 0.49, 0.78,
        v6_data["final_arr"], v7_data["final_arr"], v8_data["final_arr"],
        v9_data["eval_a_final"],
    ]
    colors3 = [
        "#95a5a6", "#e74c3c", "#f39c12", "#3498db", "#16a085",
        "#8e44ad", "#935116", "#27ae60", "#2c3e50",
    ]
    bar_x = np.arange(len(labels3))

    bars = ax3.bar(bar_x, arr_vals, color=colors3, width=0.62,
                   edgecolor="white", linewidth=1.2, zorder=3)

    # 기준선 점선
    ax3.axhline(0.74, color="#95a5a6", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2)

    for bar, val in zip(bars, arr_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                 f"{val*100:.0f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    # V8 강조 화살표 (전체 중 최고 도착률)
    best_idx = int(np.argmax(arr_vals))
    ax3.annotate("최고\n성능",
                 xy=(best_idx, arr_vals[best_idx]), xytext=(best_idx - 0.9, 0.96),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.3),
                 fontsize=8.5, color="#27ae60", fontweight="bold")

    ax3.set_xticks(bar_x)
    ax3.set_xticklabels(labels3, fontsize=7.8)
    ax3.set_ylabel("최종 도착률 (100에피소드 평균)", fontsize=9.5)
    ax3.set_ylim(0, 1.08)
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=9)
    ax3.set_title("임계값 전략별 도착률 비교\n(8장애물 랜덤 환경, 1000 에피소드)", fontsize=11, fontweight="bold", pad=8)
    ax3.grid(True, axis="y", alpha=0.3, linewidth=0.5, zorder=1)

    # ══════════════════════════════════════════════════════════════════════════
    # 서브플롯 4: V5-B / V6 / V7 / V8 / V9 학습 곡선
    # ══════════════════════════════════════════════════════════════════════════
    curve_specs = [
        ("V5-B", v5_data["v5b"], "#16a085"),
        ("V6",   v6_data,        "#8e44ad"),
        ("V7",   v7_data,        "#935116"),
        ("V8",   v8_data,        "#27ae60"),
        ("V9",   v9_data,        "#2c3e50"),
    ]

    for name, d, color in curve_specs:
        hist = d["arrival_history"]
        smooth = rolling_mean(hist, 50)
        ax4.plot(episodes, hist, color=color, alpha=0.15, linewidth=0.7)
        ax4.plot(episodes, smooth, color=color, linewidth=2.0,
                 label=f"{name}  {d['final_arr']*100:.0f}%")

    ax4.set_xlim(0, 1020)
    ax4.set_ylim(-0.05, 1.10)
    ax4.set_xlabel("에피소드", fontsize=10)
    ax4.set_ylabel("도착률 (50ep 이동평균)", fontsize=10)
    ax4.set_title("V5-B~V9 학습 곡선\n(CE 필터링 전략 비교)", fontsize=11, fontweight="bold", pad=8)
    ax4.legend(fontsize=8.5, loc="upper left", ncol=2, title="최종 도착률", title_fontsize=8)
    ax4.grid(True, alpha=0.3, linewidth=0.5)

    # 필수 각주: V9는 학습 중 50% 확률 공격 조건 (다른 버전과 난이도 자체가 다름)
    note4 = (
        "v9는 학습 내내 50% 확률로만 공격받는 조건이며, 나머지 버전은 항상 공격받는 조건이다.\n"
        "두 조건은 과제 난이도 자체가 다르므로 곡선의 높낮이만으로 우열을 판단하지 않는다."
    )
    ax4.text(0.5, -0.30, note4, transform=ax4.transAxes, ha="center",
             fontsize=7.6, style="italic", color="#7f8c8d")

    # ══════════════════════════════════════════════════════════════════════════
    # 서브플롯 5: CE 판별기 Confusion Matrix (원본 그대로)
    # ══════════════════════════════════════════════════════════════════════════
    TP, FP, TN, FN = 25, 0, 2, 3
    matrix = np.array([[TN, FP], [FN, TP]])

    im = ax5.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=28)
    ax5.set_xticks([0, 1])
    ax5.set_yticks([0, 1])
    ax5.set_xticklabels(["CE없음\n(≤10px)", "CE있음\n(>10px)"], fontsize=9)
    ax5.set_yticklabels(["실패없음", "실패있음\n(IoU기준)"], fontsize=9)
    ax5.set_xlabel("CE 판별기 예측", fontsize=10)
    ax5.set_ylabel("실제 탐지 실패", fontsize=10)
    ax5.set_title("CE 판별기 Confusion Matrix\n(독립 기준: IoU vs shift>10px)", fontsize=11, fontweight="bold", pad=8)

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            c = "white" if matrix[i, j] > thresh else "black"
            ax5.text(j, i, f"{cell_labels[i][j]}\n{matrix[i, j]}",
                     ha="center", va="center", fontsize=13, fontweight="bold", color=c)

    # 성능 지표 텍스트 (우측)
    acc  = (TP + TN) / (TP + FP + TN + FN)
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    f1   = 2 * TP / (2 * TP + FP + FN)
    metrics_str = (
        f"Accuracy   {acc*100:.1f}%\n"
        f"Sensitivity {sens*100:.1f}%\n"
        f"Specificity {spec*100:.1f}%\n"
        f"F1 Score   {f1:.3f}"
    )
    ax5.text(1.55, 0.5, metrics_str,
             transform=ax5.transAxes, fontsize=8.8,
             va="center", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#eaf4fb", alpha=0.95,
                       edgecolor="#aed6f1"))

    ax5.text(0.5, -0.26, "FN=3: eps=0 baseline ghost (CE필터 한계)",
             transform=ax5.transAxes, ha="center", fontsize=8,
             style="italic", color="#7f8c8d")

    # ══════════════════════════════════════════════════════════════════════════
    # 서브플롯 6: V5 시나리오별 결과 막대그래프 (원본 그대로)
    # ══════════════════════════════════════════════════════════════════════════
    sc_labels = ["#1\n[2.3,2.5]\n→[11.6,2.4]",
                 "#2\n[5.5,2.4]\n→[3.1,14.9]",
                 "#3\n[12.7,14.8]\n→[2.2,2.4]",
                 "#4\n[5.9,15.9]\n→[4.1,2.9]",
                 "#5\n[13.0,9.0]\n→[11.5,2.7]"]

    # 1=도착, 0=실패 (시나리오별)
    v5a_res = [0, 0, 0, 0, 1]   # out / out / col / out / arrived
    v5b_res = [1, 1, 1, 1, 0]   # arr / arr / arr / arr / collision

    sc_x = np.arange(len(sc_labels))
    w = 0.32

    def _bar_color(val: int, is_a: bool) -> str:
        if val == 1:
            return "#27ae60"
        return "#e74c3c" if is_a else "#c0392b"

    for i, (va, vb) in enumerate(zip(v5a_res, v5b_res)):
        ax6.bar(i - w / 2, va if va else -0.08, width=w,
                color=_bar_color(va, True),
                edgecolor="white", linewidth=0.8)
        ax6.bar(i + w / 2, vb if vb else -0.08, width=w,
                color=_bar_color(vb, False),
                edgecolor="white", linewidth=0.8)

    # 결과 레이블 (도착=O, 실패=x)
    result_kr = {
        (0, 0): "이탈", (0, 1): "충돌",
        (1, 0): "out", (1, 1): "도착",
    }
    va_labels = ["이탈", "이탈", "충돌", "이탈", "도착"]
    vb_labels = ["도착", "도착", "도착", "도착", "충돌"]

    for i in range(5):
        va, vb = v5a_res[i], v5b_res[i]
        ax6.text(i - w / 2, (0.08 if va else 0.02),
                 va_labels[i], ha="center", fontsize=8.5, fontweight="bold",
                 color="darkgreen" if va else "#c0392b")
        ax6.text(i + w / 2, (0.08 if vb else 0.02),
                 vb_labels[i], ha="center", fontsize=8.5, fontweight="bold",
                 color="darkgreen" if vb else "#c0392b")

    ax6.set_xticks(sc_x)
    ax6.set_xticklabels(sc_labels, fontsize=7.5)
    ax6.set_ylim(-0.2, 1.3)
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(["실패", "도착"], fontsize=10)
    ax6.set_title("V5 시나리오별 결과\n(5개 시나리오 테스트)", fontsize=11, fontweight="bold", pad=8)
    ax6.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    legend_handles6 = [
        mpatches.Patch(color="#27ae60", label="도착 성공"),
        mpatches.Patch(color="#e74c3c", label="실패 (이탈/충돌)"),
    ]
    ax6.legend(handles=legend_handles6, fontsize=8.5, loc="upper right")

    # 도착 수 요약
    ax6.text(0.5, 1.18,
             f"V5-A: 1/5 도착   |   V5-B: 4/5 도착",
             transform=ax6.transAxes, ha="center", fontsize=9.5, fontweight="bold",
             color="#2c3e50",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fdfefe", alpha=0.9,
                       edgecolor="#bdc3c7"))

    # 범례 패치 (V5-A / V5-B 구분)
    ax6.text(-0.02, 0.5, "V5-A", transform=ax6.transAxes,
             fontsize=8, color="#3498db", fontweight="bold",
             va="center", rotation=90)
    ax6.text(1.02, 0.5, "V5-B", transform=ax6.transAxes,
             fontsize=8, color="#27ae60", fontweight="bold",
             va="center", rotation=90)

    # ══════════════════════════════════════════════════════════════════════════
    # 전체 제목 및 참조 파일 목록
    # ══════════════════════════════════════════════════════════════════════════
    fig.suptitle(
        "FRDDM-DQN 적대적 공격 취약성 분석 및 개선 DDM 설계 — 최종 결과 (V2: V6~V9 포함)",
        fontsize=15, fontweight="bold", y=0.955,
        color="#1a252f",
    )

    ref_text = (
        "참조 파일: attack_results.json | attack_results_pgd.json | "
        "detection_failure.json | dqn_results_v5~v9.json | "
        "threshold_summary_table_v2.png | ce_confusion_matrix_v2.png | "
        "flight_animation_v5.gif"
    )
    fig.text(0.5, 0.005, ref_text,
             ha="center", fontsize=7.5, color="#7f8c8d", style="italic")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    out = os.path.join(RESULTS_DIR, "final_summary_v2.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[저장] {out}")


def main() -> None:
    print("최종 결과 요약 그림 생성 중 (V2)...")
    build_final_summary()
    print("완료.")


if __name__ == "__main__":
    main()
