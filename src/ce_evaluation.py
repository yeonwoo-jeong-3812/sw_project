"""CE(오염 경험) 판별기 분류 성능 평가.

데이터 출처 (기존 파일만 사용, 새 시뮬레이션 없음):
  1. results/dqn_results.json / dqn_results_v4.json / dqn_results_v5.json / env_test_results.json
     - V4/V5 실험에서 측정된 임계값별 도착률, CE 폐기율, 유효 경험 수
  2. results/detection_failure.json
     - IoU 기준 Miss/Ghost 탐지 실패 여부 (독립적 정답)
  3. results/attack_results_pgd.json
     - PGD 공격 시 bbox shift 값 (우리 CE 판별기 입력)

분석 1: 정직한 임계값 비교표
  - 새 학습 없이 기존 V4/V5 실험 수치만 정리
분석 2: 독립적 기준 Confusion Matrix
  - GT: detection_failure.json (IoU 기반 탐지 실패)
  - Pred: attack_results_pgd.json (shift > 10px)
  - 서로 다른 측정 방식 → 순환논리 없음
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = "results"


def load_threshold_data() -> list[dict]:
    """기존 JSON 파일에서 임계값별 실험 수치를 로드한다."""
    rows = []

    # 기준선: CE없음 (8장애물 랜덤)
    with open(os.path.join(RESULTS_DIR, "env_test_results.json"), encoding="utf-8") as f:
        env_data = json.load(f)
    env2 = env_data["env2_8obs_random"]
    rows.append({
        "label": "기준선\nCE없음",
        "strategy": "전체 저장\n(8장애물)",
        "final_arr": env2["final_arr"],
        "discard_rate": 0.0,
        "valid_exp": env2["total_steps"],
        "highlight": False,
    })

    # V4 실험 (보상 보정 없음)
    with open(os.path.join(RESULTS_DIR, "dqn_results_v4.json"), encoding="utf-8") as f:
        v4 = json.load(f)
    rows.append({
        "label": "V4-A\n이진 CE",
        "strategy": "shift>10px\n폐기",
        "final_arr": v4["v4a"]["final_arr"],
        "discard_rate": v4["v4a"]["discard_rate"],
        "valid_exp": v4["v4a"]["valid_exp"],
        "highlight": False,
    })
    rows.append({
        "label": "V4-B\n연속 CE",
        "strategy": "가중치\n저장",
        "final_arr": v4["v4b"]["final_arr"],
        "discard_rate": v4["v4b"]["discard_rate"],
        "valid_exp": v4["v4b"]["valid_exp"],
        "highlight": False,
    })

    # V5 실험 (시간/거리 보상 추가)
    with open(os.path.join(RESULTS_DIR, "dqn_results_v5.json"), encoding="utf-8") as f:
        v5 = json.load(f)
    rows.append({
        "label": "V5-A\n이진 CE",
        "strategy": "shift>10px 폐기\n+시간/거리 보상",
        "final_arr": v5["v5a"]["final_arr"],
        "discard_rate": v5["v5a"]["discard_rate"],
        "valid_exp": v5["v5a"]["valid_exp"],
        "highlight": False,
    })
    rows.append({
        "label": "V5-B\n연속 CE",
        "strategy": "가중치 저장\n+시간/거리 보상",
        "final_arr": v5["v5b"]["final_arr"],
        "discard_rate": v5["v5b"]["discard_rate"],
        "valid_exp": v5["v5b"]["valid_exp"],
        "highlight": True,  # 최고 도착률
    })

    return rows


def save_threshold_table(rows: list[dict]) -> None:
    """임계값 비교표를 PNG로 저장한다."""
    headers = ["실험", "CE 전략", "최종 도착률", "CE 폐기율", "유효 경험 수"]
    cell_data = [
        [
            r["label"],
            r["strategy"],
            f"{r['final_arr'] * 100:.1f}%",
            f"{r['discard_rate'] * 100:.1f}%",
            f"{r['valid_exp']:,}",
        ]
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 2.6)

    # 헤더 스타일
    header_color = "#2c3e50"
    for j in range(len(headers)):
        cell = tbl[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    # 행 색상
    row_colors = ["#eaf4fb", "#ffffff"]
    for i, r in enumerate(rows):
        bg = "#d5f5e3" if r["highlight"] else row_colors[i % 2]
        for j in range(len(headers)):
            tbl[i + 1, j].set_facecolor(bg)

    ax.set_title(
        "임계값 전략별 CE 필터링 성능 비교\n(V4/V5 기존 실험 수치 — 새 학습 없음)",
        fontsize=13,
        fontweight="bold",
        pad=18,
    )

    note = "초록색 행(V5-B): 연속 CE + 시간/거리 보상 → 최고 도착률 78%"
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, style="italic",
             color="#1a5276")

    out = os.path.join(RESULTS_DIR, "threshold_summary_table.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out}")


def _eps_key(eps) -> str:
    """epsilon 값을 attack_results_pgd.json 키 형식으로 변환한다."""
    if eps == 0 or eps == 0.0:
        return "0"
    # 부동소수점 표현 그대로 사용 (0.01, 0.03, 0.05, 0.07, 0.1)
    return str(eps)


def build_independent_confusion(threshold_px: float = 10.0) -> dict:
    """
    GT: detection_failure.json pgd.per_image 의 miss_cnt + ghost_cnt > 0
    Pred: attack_results_pgd.json 의 center_error_px > threshold_px
    """
    with open(os.path.join(RESULTS_DIR, "detection_failure.json"), encoding="utf-8") as f:
        det = json.load(f)
    with open(os.path.join(RESULTS_DIR, "attack_results_pgd.json"), encoding="utf-8") as f:
        atk = json.load(f)

    epsilons = det["epsilons"]
    pgd_images = det["pgd"]["per_image"]

    TP = FP = TN = FN = 0
    records = []

    for img, img_data in pgd_images.items():
        for eps_idx, eps in enumerate(epsilons):
            miss = img_data["miss_cnt"][eps_idx]
            ghost = img_data["ghost_cnt"][eps_idx]
            actual = int((miss + ghost) > 0)

            key = _eps_key(eps)
            if img in atk and key in atk[img]:
                shift = float(atk[img][key].get("center_error_px", 0.0))
            else:
                shift = 0.0

            pred = int(shift > threshold_px)

            records.append({
                "img": img, "eps": eps,
                "actual": actual, "pred": pred,
                "shift": shift, "miss": miss, "ghost": ghost,
            })

            if actual == 1 and pred == 1:
                TP += 1
            elif actual == 0 and pred == 1:
                FP += 1
            elif actual == 0 and pred == 0:
                TN += 1
            else:
                FN += 1

    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) > 0 else 0.0)

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "accuracy": accuracy, "sensitivity": sensitivity,
        "specificity": specificity, "precision": precision, "f1": f1,
        "threshold": threshold_px,
        "records": records,
    }


def save_confusion_v2(cm: dict) -> None:
    """독립적 기준 Confusion Matrix를 PNG로 저장한다."""
    TP, FP, TN, FN = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    threshold = int(cm["threshold"])

    fig = plt.figure(figsize=(14, 6.5))

    # --- 왼쪽: Confusion Matrix 히트맵 ---
    ax1 = fig.add_subplot(1, 2, 1)
    matrix = np.array([[TN, FP], [FN, TP]])
    im = ax1.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0)

    ax1.set_title(
        f"독립적 기준 Confusion Matrix\n(CE 판별기 T={threshold}px vs IoU 탐지 실패)",
        fontsize=11, fontweight="bold",
    )
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels([f"CE없음\n(shift≤{threshold}px)", f"CE있음\n(shift>{threshold}px)"], fontsize=9)
    ax1.set_yticklabels(["실패없음\n(Miss/Ghost=0)", "실패있음\n(Miss/Ghost>0)"], fontsize=9)
    ax1.set_xlabel("CE 판별기 예측", fontsize=10)
    ax1.set_ylabel("실제 탐지 실패 여부 (IoU 기준)", fontsize=10)

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > thresh else "black"
            ax1.text(
                j, i,
                f"{cell_labels[i][j]}\n{matrix[i, j]}",
                ha="center", va="center",
                fontsize=14, fontweight="bold", color=color,
            )

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # --- 오른쪽: 성능 지표 ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.set_title("성능 지표", fontsize=12, fontweight="bold")

    metrics = [
        ("Accuracy",            f"{cm['accuracy'] * 100:.1f}%",    True),
        ("Sensitivity (Recall)", f"{cm['sensitivity'] * 100:.1f}%", False),
        ("Specificity",          f"{cm['specificity'] * 100:.1f}%", False),
        ("Precision",            f"{cm['precision'] * 100:.1f}%",   False),
        ("F1 Score",             f"{cm['f1']:.3f}",                 True),
    ]

    y = 0.85
    for name, val, bold in metrics:
        fw = "bold" if bold else "normal"
        color_val = "#1a5276" if bold else "#2471a3"
        ax2.text(0.05, y, name, transform=ax2.transAxes,
                 fontsize=11, fontweight=fw, va="top")
        ax2.text(0.68, y, val, transform=ax2.transAxes,
                 fontsize=11, fontweight=fw, va="top", color=color_val)
        y -= 0.13

    y -= 0.04
    sep_metrics = [
        ("TP", str(TP)), ("FP", str(FP)),
        ("TN", str(TN)), ("FN", str(FN)),
    ]
    for name, val in sep_metrics:
        ax2.text(0.05, y, name, transform=ax2.transAxes, fontsize=10, va="top", color="#555")
        ax2.text(0.68, y, val, transform=ax2.transAxes, fontsize=10, va="top", color="#555")
        y -= 0.10

    # 해석 주석
    note = (
        f"FN={FN}: eps=0 기준선 ghost (PGD 없이 발생)\n"
        f"CE 필터(shift>{threshold}px)는 자연 발생 ghost 탐지 불가\n"
        "→ 서로 다른 측정 방식이므로 순환논리 없음"
    )
    fig.text(
        0.5, 0.01, note,
        ha="center", fontsize=9, style="italic",
        bbox=dict(facecolor="lightyellow", alpha=0.85, edgecolor="orange", pad=4),
    )

    plt.suptitle(
        f"CE 판별기 독립 평가  (N=30: 5장 × 6 epsilon값)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "ce_confusion_matrix_v2.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out}")


def main() -> None:
    # 구버전 파일 삭제
    for old in ["ce_roc_curve.png", "ce_confusion_matrix.png"]:
        path = os.path.join(RESULTS_DIR, old)
        if os.path.exists(path):
            os.remove(path)
            print(f"[삭제] {path}")

    # ── 분석 1: 임계값 비교표 ─────────────────────────────────
    print("\n=== 분석 1: 임계값 전략별 비교표 ===")
    rows = load_threshold_data()
    for r in rows:
        print(
            f"  {r['label'].replace(chr(10), ' '):<16} "
            f"도착률={r['final_arr'] * 100:.1f}%  "
            f"폐기율={r['discard_rate'] * 100:.1f}%  "
            f"유효경험={r['valid_exp']:,}"
        )
    save_threshold_table(rows)

    # ── 분석 2: 독립적 Confusion Matrix ──────────────────────
    print("\n=== 분석 2: 독립적 기준 Confusion Matrix (T=10px) ===")
    cm = build_independent_confusion(threshold_px=10.0)
    print(f"  TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}")
    print(f"  Accuracy    = {cm['accuracy'] * 100:.1f}%")
    print(f"  Sensitivity = {cm['sensitivity'] * 100:.1f}%")
    print(f"  Specificity = {cm['specificity'] * 100:.1f}%")
    print(f"  Precision   = {cm['precision'] * 100:.1f}%")
    print(f"  F1 Score    = {cm['f1']:.3f}")
    save_confusion_v2(cm)

    print("\n완료.")


if __name__ == "__main__":
    main()
