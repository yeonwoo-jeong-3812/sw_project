"""CE(오염 경험) 판별기 분류 성능 평가 — V2 (V6~V9 결과 추가).

ce_evaluation.py 원본은 수정하지 않고, load_threshold_data()와
save_threshold_table()만 확장한 버전이다. build_independent_confusion()과
save_confusion_v2()는 원본과 동일하게 그대로 두되, main()에서는 호출하지
않는다 (기존 ce_confusion_matrix_v2.png를 건드리지 않기 위함).

CE 판정 기준(shift>10px)은 V4~V9까지 변경된 적이 없고, 비교 대상
detection_failure.json / attack_results_pgd.json도 DQN 버전과 무관한
오프라인 분석 데이터이므로 분석 2(독립적 confusion matrix)는 그대로
유효하며 재실행할 필요가 없다.

데이터 출처 (기존 파일만 사용, 새 시뮬레이션 없음):
  1. results/dqn_results_v4.json / dqn_results_v5.json / env_test_results.json
     - V4/V5 실험에서 측정된 임계값별 도착률, CE 폐기율, 유효 경험 수
  2. results/dqn_results_v6.json / v7 / v8 / v9
     - V6~V9 실험에서 측정된 동일 지표 (아래 매핑 규칙 참고)

V6~V9 필드 매핑 규칙:
  - final_arr: 각 버전 json에서 그대로 사용
    (단, V9는 학습 중 final_arr이 아니라 eval_a_final을 사용 — 항상 공격
     조건(조건 A)으로, 나머지 모든 행이 항상 공격 조건에서 측정되었기
     때문에 같은 조건끼리 비교하기 위함. eval_b_final은 50% 공격 조건이라
     제외한다.)
  - discard_rate:
      V6: ce_blocked / total_steps (v6은 type_re/de/se 필드가 없음)
      V7/V8: type_ce / total_steps
      V9: discard_count / total_steps
  - valid_exp:
      V6: valid_exp 필드를 직접 사용 (type_re/de/se 필드가 존재하지 않고,
          valid_exp + ce_blocked == total_steps로 정합성 확인됨)
      V7/V8/V9: type_re + type_de + type_se
          (buffer_size_final이 아니라 누적 저장 시도 성공 횟수를 사용.
           각 버전에서 이 합이 buffer_size_final보다 큼을 확인함)
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
    """기존 JSON 파일에서 임계값별 실험 수치를 로드한다 (V4~V9)."""
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
        "highlight": False,
    })

    # V6 실험 (연속 CE + 오염페널티, ImprovedDDMBuffer 이전)
    with open(os.path.join(RESULTS_DIR, "dqn_results_v6.json"), encoding="utf-8") as f:
        v6 = json.load(f)["v6"]
    rows.append({
        "label": "V6\n연속 CE",
        "strategy": "가중치 저장\n+오염페널티",
        "final_arr": v6["final_arr"],
        "discard_rate": v6["ce_blocked"] / v6["total_steps"],
        "valid_exp": v6["valid_exp"],
        "highlight": False,
    })

    # V7 실험 (ImprovedDDMBuffer + CE 분류)
    with open(os.path.join(RESULTS_DIR, "dqn_results_v7.json"), encoding="utf-8") as f:
        v7 = json.load(f)["v7"]
    rows.append({
        "label": "V7\nImprovedDDMBuffer",
        "strategy": "타입별 분류\n(RE/DE/SE/CE)",
        "final_arr": v7["final_arr"],
        "discard_rate": v7["type_ce"] / v7["total_steps"],
        "valid_exp": v7["type_re"] + v7["type_de"] + v7["type_se"],
        "highlight": False,
    })

    # V8 실험 (ImprovedDDMBuffer + 연속 CE 생존)
    with open(os.path.join(RESULTS_DIR, "dqn_results_v8.json"), encoding="utf-8") as f:
        v8 = json.load(f)["v8"]
    rows.append({
        "label": "V8\nImprovedDDMBuffer",
        "strategy": "연속 CE\n생존 반영",
        "final_arr": v8["final_arr"],
        "discard_rate": v8["type_ce"] / v8["total_steps"],
        "valid_exp": v8["type_re"] + v8["type_de"] + v8["type_se"],
        "highlight": True,  # 최고 도착률
    })

    # V9 실험 (확률적 공격 p=0.5 + 연속 CE 생존)
    # final_arr는 학습 중 값이 아니라 eval_a_final(항상 공격 조건, 조건 A)을
    # 사용한다. 나머지 모든 행이 항상 공격 조건에서 측정되었기 때문에
    # 같은 조건끼리 비교하기 위함 (eval_b_final=50% 공격 조건은 제외).
    with open(os.path.join(RESULTS_DIR, "dqn_results_v9.json"), encoding="utf-8") as f:
        v9 = json.load(f)["v9"]
    rows.append({
        "label": "V9\n확률적공격 p=0.5",
        "strategy": "연속 CE 생존\n(평가:조건A)",
        "final_arr": v9["eval_a_final"],
        "discard_rate": v9["discard_count"] / v9["total_steps"],
        "valid_exp": v9["type_re"] + v9["type_de"] + v9["type_se"],
        "highlight": False,
    })

    return rows


def save_threshold_table(rows: list[dict]) -> None:
    """임계값 비교표를 PNG로 저장한다 (V4~V9, threshold_summary_table_v2.png)."""
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

    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=headers,
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 2.3)

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
        "임계값 전략별 CE 필터링 성능 비교\n(V4~V9 기존 실험 수치 — 새 학습 없음)",
        fontsize=13,
        fontweight="bold",
        pad=18,
    )

    note = (
        "초록색 행(V8): 연속 CE 생존 반영(ImprovedDDMBuffer) → 최고 도착률 84%\n"
        "v7~v9의 유효 경험 수는 최종 버퍼 크기가 아닌 누적 저장 성공 횟수이며, "
        "v4/v5의 valid_exp와 계산 방식이 다르므로 절대값 비교 시 유의할 것.\n"
        "v9의 도착률은 항상 공격 조건(조건 A) 평가 기준이며, 학습 시 도착률과는 "
        "별도로 측정된 지표이다 (이번 결과에서는 두 값이 우연히 같게 나왔다)."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=8.5, style="italic",
             color="#1a5276")

    out = os.path.join(RESULTS_DIR, "threshold_summary_table_v2.png")
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

    원본 ce_evaluation.py와 동일 (수정하지 않음). V2에서는 호출하지 않는다.
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
    """독립적 기준 Confusion Matrix를 PNG로 저장한다.

    원본 ce_evaluation.py와 동일 (수정하지 않음). V2에서는 호출하지 않는다
    (기존 ce_confusion_matrix_v2.png를 덮어쓰지 않기 위함).
    """
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
    # ── 분석 1: 임계값 비교표 (V4~V9) ─────────────────────────
    print("\n=== 분석 1: 임계값 전략별 비교표 (V4~V9) ===")
    rows = load_threshold_data()
    for r in rows:
        print(
            f"  {r['label'].replace(chr(10), ' '):<24} "
            f"도착률={r['final_arr'] * 100:.1f}%  "
            f"폐기율={r['discard_rate'] * 100:.1f}%  "
            f"유효경험={r['valid_exp']:,}"
        )
    save_threshold_table(rows)

    # 분석 2(독립적 confusion matrix)는 CE 판정 기준/GT 데이터가 V4~V9
    # 사이에 변경되지 않았으므로 재실행하지 않는다. 기존
    # ce_confusion_matrix_v2.png를 그대로 유지한다.

    print("\n완료.")


if __name__ == "__main__":
    main()
