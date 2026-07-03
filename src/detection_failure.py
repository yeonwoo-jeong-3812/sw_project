"""탐지 실패율 분석 스크립트.

GT 바운딩 박스와의 IoU(임계값 0.5)를 기준으로
FGSM / PGD 각각에 대해 epsilon별 Miss율·Ghost율·Total Failure율을 계산한다.

정의:
  Miss  : GT 박스와 IoU ≥ 0.5인 탐지 박스가 없는 경우 (미검출)
          Miss율  = 미검출 GT 수 / 전체 GT 수
  Ghost : GT 박스와 IoU ≥ 0.5인 GT가 없는 탐지 박스 (허위 탐지)
          Ghost율 = ghost 탐지 수 / max(전체 탐지 수, 1)
  Total : (Miss 수 + Ghost 수) / (GT 수 + 탐지 수)
          ∈ [0, 1]  — 분모를 GT+탐지 합으로 놓아 상한 보장
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 한글 폰트 ─────────────────────────────────────────────────────────────
plt.rcParams["font.family"]        = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
GT_PATH     = ROOT / "data" / "gt_annotations.json"
RESULTS_DIR = ROOT / "results"
FGSM_JSON   = RESULTS_DIR / "attack_results.json"
PGD_JSON    = RESULTS_DIR / "attack_results_pgd.json"
OUT_JSON    = RESULTS_DIR / "detection_failure.json"
OUT_PNG     = RESULTS_DIR / "detection_failure.png"

EPSILONS      = [0, 0.01, 0.03, 0.05, 0.07, 0.10]
IOU_THRESHOLD = 0.5


# ══════════════════════════════════════════════════════════════════════════
# IoU 계산
# ══════════════════════════════════════════════════════════════════════════
def iou(box_a: list, box_b: list) -> float:
    """두 박스 [x1, y1, x2, y2]의 IoU."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


# ══════════════════════════════════════════════════════════════════════════
# 단일 이미지 분석
# ══════════════════════════════════════════════════════════════════════════
def analyze_image(
    gt_boxes: list,
    det_boxes: list,
    thresh: float = IOU_THRESHOLD,
) -> dict:
    """GT 박스 목록 vs 탐지 박스 목록 → Miss/Ghost 집계.

    Returns:
        miss_count  : GT 박스 중 매칭 실패 수
        ghost_count : 탐지 박스 중 GT 매칭 실패 수
        n_gt        : GT 박스 수
        n_det       : 탐지 박스 수
        matched_gt  : 매칭 성공 GT 인덱스 집합
    """
    n_gt  = len(gt_boxes)
    n_det = len(det_boxes)

    if n_gt == 0:
        return dict(miss_count=0, ghost_count=n_det, n_gt=0, n_det=n_det)
    if n_det == 0:
        return dict(miss_count=n_gt, ghost_count=0, n_gt=n_gt, n_det=0)

    # IoU 행렬 (n_gt × n_det)
    iou_mat = np.array(
        [[iou(g, d) for d in det_boxes] for g in gt_boxes],
        dtype=np.float32,
    )

    miss_count  = int(np.sum(iou_mat.max(axis=1) < thresh))   # row: GT
    ghost_count = int(np.sum(iou_mat.max(axis=0) < thresh))   # col: det

    return dict(
        miss_count=miss_count,
        ghost_count=ghost_count,
        n_gt=n_gt,
        n_det=n_det,
    )


# ══════════════════════════════════════════════════════════════════════════
# epsilon별 실패율 계산
# ══════════════════════════════════════════════════════════════════════════
def compute_failure_rates(
    attack_data: dict,
    gt: dict,
    filenames: list[str],
) -> dict:
    """공격 결과 dict에서 epsilon별 실패율을 계산한다."""
    per_img: dict[str, dict] = {
        fn: {"miss": [], "ghost": [], "total": [],
             "miss_cnt": [], "ghost_cnt": [], "n_gt": [], "n_det": []}
        for fn in filenames
    }

    miss_rates:  list[float] = []
    ghost_rates: list[float] = []
    total_rates: list[float] = []

    for eps in EPSILONS:
        key = str(eps)
        eps_miss, eps_ghost, eps_total = [], [], []

        for fn in filenames:
            gt_boxes  = gt[fn]
            det_boxes = attack_data[fn].get(key, {}).get("boxes", [])
            r = analyze_image(gt_boxes, det_boxes)

            # --- 비율 계산 ---
            miss_r  = r["miss_count"]  / r["n_gt"]  if r["n_gt"]  > 0 else 0.0
            ghost_r = r["ghost_count"] / max(r["n_det"], 1)
            denom   = r["n_gt"] + r["n_det"]
            total_r = (r["miss_count"] + r["ghost_count"]) / denom if denom > 0 else 0.0

            eps_miss.append(miss_r)
            eps_ghost.append(ghost_r)
            eps_total.append(total_r)

            per_img[fn]["miss"].append(round(miss_r, 4))
            per_img[fn]["ghost"].append(round(ghost_r, 4))
            per_img[fn]["total"].append(round(total_r, 4))
            per_img[fn]["miss_cnt"].append(r["miss_count"])
            per_img[fn]["ghost_cnt"].append(r["ghost_count"])
            per_img[fn]["n_gt"].append(r["n_gt"])
            per_img[fn]["n_det"].append(r["n_det"])

        miss_rates.append(round(float(np.mean(eps_miss)),  4))
        ghost_rates.append(round(float(np.mean(eps_ghost)), 4))
        total_rates.append(round(float(np.mean(eps_total)), 4))

    return {
        "miss_rate":          miss_rates,
        "ghost_rate":         ghost_rates,
        "total_failure_rate": total_rates,
        "per_image":          per_img,
    }


# ══════════════════════════════════════════════════════════════════════════
# 그래프
# ══════════════════════════════════════════════════════════════════════════
def plot_failure(fgsm_stats: dict, pgd_stats: dict, out_path: Path) -> None:
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    filenames = list(fgsm_stats["per_image"].keys())

    fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
    fig.suptitle(
        "FGSM vs PGD 탐지 실패율 비교",
        fontsize=15, fontweight="bold", y=0.985,
    )

    # per_image 서브키 매핑 (저장 시 짧은 이름 사용)
    _img_key = {"miss_rate": "miss", "ghost_rate": "ghost", "total_failure_rate": "total"}

    rows = [
        (axes[0], "① Miss율  (GT 박스 미검출 비율)",      "miss_rate"),
        (axes[1], "② Ghost율  (허위 탐지 비율)",           "ghost_rate"),
        (axes[2], "③ Total Failure율  (Miss + Ghost 합산)", "total_failure_rate"),
    ]

    for ax, title, key in rows:
        fgsm_vals = fgsm_stats[key]
        pgd_vals  = pgd_stats[key]
        img_k     = _img_key[key]

        # 개별 이미지 배경선
        for i, fn in enumerate(filenames):
            f_img = fgsm_stats["per_image"][fn][img_k]
            p_img = pgd_stats["per_image"][fn][img_k]
            ax.plot(EPSILONS, f_img, "o--", color="#1565C0",
                    alpha=0.18, linewidth=1.0, markersize=3)
            ax.plot(EPSILONS, p_img, "s--", color="#B71C1C",
                    alpha=0.18, linewidth=1.0, markersize=3)

        # 평균 실선
        ax.plot(EPSILONS, fgsm_vals, "o-", color="#1565C0",
                linewidth=2.8, markersize=9,
                label="FGSM  (steps=1)", zorder=5)
        ax.plot(EPSILONS, pgd_vals,  "s-", color="#C62828",
                linewidth=2.8, markersize=9,
                label=f"PGD   (steps=10, alpha=ε/4)", zorder=5)

        # 값 레이블
        for eps, fv in zip(EPSILONS, fgsm_vals):
            ax.annotate(
                f"{fv:.2f}", (eps, fv),
                textcoords="offset points", xytext=(-14, 9),
                ha="center", fontsize=8.5, fontweight="bold", color="#1565C0",
            )
        for eps, pv in zip(EPSILONS, pgd_vals):
            ax.annotate(
                f"{pv:.2f}", (eps, pv),
                textcoords="offset points", xytext=(14, 9),
                ha="center", fontsize=8.5, fontweight="bold", color="#C62828",
            )

        ax.set_ylabel("비율 (0 ~ 1)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=7)
        ax.set_ylim(-0.04, 1.18)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.legend(fontsize=9.5, loc="upper left", framealpha=0.85)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Epsilon", fontsize=12)
    axes[2].set_xticks(EPSILONS)
    axes[2].set_xlim(-0.008, 0.115)

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"그래프 저장 → {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    with open(FGSM_JSON, encoding="utf-8") as f:
        fgsm_data = json.load(f)
    with open(PGD_JSON, encoding="utf-8") as f:
        pgd_data = json.load(f)

    filenames = sorted(gt.keys())

    print("FGSM 탐지 실패 분석 중...")
    fgsm_stats = compute_failure_rates(fgsm_data, gt, filenames)

    print("PGD  탐지 실패 분석 중...")
    pgd_stats = compute_failure_rates(pgd_data, gt, filenames)

    # ── JSON 저장 ─────────────────────────────────────────────────────────
    output = {
        "iou_threshold": IOU_THRESHOLD,
        "epsilons":      EPSILONS,
        "fgsm":          fgsm_stats,
        "pgd":           pgd_stats,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"결과 저장 → {OUT_JSON}\n")

    # ── 표 출력 ───────────────────────────────────────────────────────────
    sep = "=" * 78
    hdr = (f"{'eps':>6} | "
           f"{'F-Miss':>7} {'P-Miss':>7} | "
           f"{'F-Ghost':>8} {'P-Ghost':>8} | "
           f"{'F-Total':>8} {'P-Total':>8}")
    print(sep)
    print(hdr)
    print("-" * 78)
    for i, eps in enumerate(EPSILONS):
        fm = fgsm_stats["miss_rate"][i]
        pm = pgd_stats["miss_rate"][i]
        fg = fgsm_stats["ghost_rate"][i]
        pg = pgd_stats["ghost_rate"][i]
        ft = fgsm_stats["total_failure_rate"][i]
        pt = pgd_stats["total_failure_rate"][i]
        print(f"{eps:>6.2f} | "
              f"{fm:>7.4f} {pm:>7.4f} | "
              f"{fg:>8.4f} {pg:>8.4f} | "
              f"{ft:>8.4f} {pt:>8.4f}")
    print(sep)

    # ── 이미지별 상세 ─────────────────────────────────────────────────────
    for method, stats in [("FGSM", fgsm_stats), ("PGD", pgd_stats)]:
        print(f"\n[{method}] 이미지별 ghost_count (허위 탐지 수)")
        print(f"  {'':14}", " ".join(f"eps={e:.2f}" for e in EPSILONS))
        for fn in filenames:
            ghost_cnt = stats["per_image"][fn]["ghost_cnt"]
            n_det     = stats["per_image"][fn]["n_det"]
            vals = " ".join(f"{g:>5}({d})" for g, d in zip(ghost_cnt, n_det))
            print(f"  {fn:<14} {vals}")

    # ── 그래프 ───────────────────────────────────────────────────────────
    plot_failure(fgsm_stats, pgd_stats, OUT_PNG)


if __name__ == "__main__":
    main()
