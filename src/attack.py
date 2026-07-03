"""FGSM 및 PGD 적대적 공격 스크립트.

학습된 Faster R-CNN(MobileNet 백본)에 FGSM / PGD 공격을 적용하고
epsilon별 바운딩 박스 중심 좌표 오차를 측정·시각화한다.

출력 파일:
  results/attack_results.json     – FGSM 결과 (기존 형식 유지)
  results/attack_results_pgd.json – PGD  결과
  results/attack_graph.png        – FGSM 단독 그래프
  results/fgsm_vs_pgd.png         – FGSM vs PGD 비교 그래프
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ── 한글 폰트 ─────────────────────────────────────────────────────────────
plt.rcParams["font.family"]        = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ──────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
DATA_DIR        = ROOT / "data" / "images"
GT_PATH         = ROOT / "data" / "gt_annotations.json"
RESULTS_DIR     = ROOT / "results"
MODEL_PATH      = RESULTS_DIR / "frcnn_finetuned.pth"
OUT_JSON_FGSM   = RESULTS_DIR / "attack_results.json"
OUT_JSON_PGD    = RESULTS_DIR / "attack_results_pgd.json"
OUT_PNG_FGSM    = RESULTS_DIR / "attack_graph.png"
OUT_PNG_COMPARE = RESULTS_DIR / "fgsm_vs_pgd.png"

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────
NUM_CLASSES  = 2
SCORE_THRESH = 0.3
EPSILONS     = [0, 0.01, 0.03, 0.05, 0.07, 0.10]
PGD_STEPS    = 10          # PGD 반복 횟수
# PGD step_size = epsilon / 4  (epsilon마다 동적으로 계산)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════
# 모델
# ══════════════════════════════════════════════════════════════════════════
def build_model() -> torch.nn.Module:
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


# ══════════════════════════════════════════════════════════════════════════
# 공격 함수
# ══════════════════════════════════════════════════════════════════════════
def fgsm_attack(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    targets: list[dict],
    epsilon: float,
) -> torch.Tensor:
    """FGSM: 1회 gradient sign step."""
    if epsilon == 0:
        return img_tensor.clone()

    adv = img_tensor.clone().detach().requires_grad_(True)
    model.train()
    total_loss = sum(model([adv], targets).values())
    model.zero_grad()
    total_loss.backward()

    return torch.clamp(img_tensor + epsilon * adv.grad.data.sign(), 0.0, 1.0).detach()


def pgd_attack(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    targets: list[dict],
    epsilon: float,
    steps: int = PGD_STEPS,
) -> torch.Tensor:
    """PGD(Projected Gradient Descent): multi-step FGSM + L∞ epsilon-ball 투영.

    각 스텝:
      1. gradient sign step:  x_adv ← x_adv + alpha * sign(∇L)
      2. L∞ 투영:             x_adv ← clip(x_adv, x_orig±epsilon)
      3. 유효 범위 클램핑:     x_adv ← clip(x_adv, 0, 1)

    step_size(alpha) = epsilon / 4  (스텝당 최대 이동량)
    """
    if epsilon == 0:
        return img_tensor.clone()

    step_size = epsilon / 4
    x_orig    = img_tensor.detach()
    x_adv     = img_tensor.clone().detach()

    model.train()
    for _ in range(steps):
        x_input = x_adv.detach().requires_grad_(True)
        total_loss = sum(model([x_input], targets).values())
        model.zero_grad()
        total_loss.backward()

        with torch.no_grad():
            x_adv = x_adv + step_size * x_input.grad.sign()
            # L∞ epsilon-ball 투영
            x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
            # 픽셀 유효 범위 [0, 1]
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


# ══════════════════════════════════════════════════════════════════════════
# 추론 및 유틸
# ══════════════════════════════════════════════════════════════════════════
def infer(model: torch.nn.Module, img_tensor: torch.Tensor) -> tuple[list, list]:
    model.eval()
    with torch.no_grad():
        out = model([img_tensor])[0]
    keep  = out["scores"] >= SCORE_THRESH
    return out["boxes"][keep].cpu().tolist(), out["scores"][keep].cpu().tolist()


def box_centers(boxes: list) -> list[tuple[float, float]]:
    return [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]


def mean_center_error(
    ref: list[tuple[float, float]],
    adv: list[tuple[float, float]],
) -> float:
    """ref 각 중심점 → adv 최근접 중심점 거리의 평균 (nearest-neighbor)."""
    if not ref:
        return 0.0
    if not adv:
        return float("nan")   # 검출 소실
    total = 0.0
    for rx, ry in ref:
        dists = [((rx - ax) ** 2 + (ry - ay) ** 2) ** 0.5 for ax, ay in adv]
        total += min(dists)
    return total / len(ref)


# ══════════════════════════════════════════════════════════════════════════
# 공통 공격 루프
# ══════════════════════════════════════════════════════════════════════════
def run_attack_loop(
    model: torch.nn.Module,
    filenames: list[str],
    gt: dict,
    attack_fn,
    attack_name: str,
) -> dict:
    """attack_fn을 전체 이미지 × epsilon에 걸쳐 실행하고 결과 dict를 반환.

    attack_fn 시그니처: (model, img_tensor, targets, epsilon) → perturbed_tensor
    """
    results: dict           = {}
    ref_centers: dict[str, list] = {}

    for eps in EPSILONS:
        print(f"── {attack_name}  epsilon={eps:.2f} " + "─" * 30)
        for fname in filenames:
            img_tensor = TF.to_tensor(
                Image.open(DATA_DIR / fname).convert("RGB")
            ).to(DEVICE)

            targets = [{
                "boxes":  torch.tensor(gt[fname], dtype=torch.float32).to(DEVICE),
                "labels": torch.ones(len(gt[fname]), dtype=torch.int64).to(DEVICE),
            }]

            adv_img       = attack_fn(model, img_tensor, targets, eps)
            boxes, scores = infer(model, adv_img)
            centers       = box_centers(boxes)

            if eps == 0:
                ref_centers[fname] = centers
                error = 0.0
            else:
                error = mean_center_error(ref_centers[fname], centers)

            results.setdefault(fname, {})[str(eps)] = {
                "boxes":           [[round(c, 2) for c in b] for b in boxes],
                "scores":          [round(s, 4) for s in scores],
                "centers":         [[round(cx, 2), round(cy, 2)] for cx, cy in centers],
                "center_error_px": None if np.isnan(error) else round(error, 4),
                "num_detections":  len(boxes),
            }

            err_str = "검출없음" if np.isnan(error) else f"{error:.2f}px"
            print(f"  {fname}: det={len(boxes):2d}  error={err_str}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# 그래프 함수
# ══════════════════════════════════════════════════════════════════════════
def _extract_errors(results: dict, filenames: list[str]) -> dict[str, list]:
    """이미지별 epsilon 순서대로 center_error_px 추출."""
    per = {}
    for fn in filenames:
        row = []
        for eps in EPSILONS:
            val = results[fn].get(str(eps), {}).get("center_error_px")
            row.append(float(val) if val is not None else np.nan)
        per[fn] = row
    return per


def _avg_errors(per: dict[str, list], filenames: list[str]) -> list[float]:
    return [
        float(np.nanmean([per[fn][j] for fn in filenames]))
        for j in range(len(EPSILONS))
    ]


def plot_single(results: dict, filenames: list[str], out_path: Path,
                title: str = "FGSM Attack: Bounding Box Center Error vs Epsilon") -> None:
    """FGSM 단독 그래프 (기존 형식 유지)."""
    colors  = plt.cm.tab10.colors  # type: ignore[attr-defined]
    per     = _extract_errors(results, filenames)
    avg     = _avg_errors(per, filenames)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, fn in enumerate(filenames):
        ax.plot(EPSILONS, per[fn], "o--", color=colors[i],
                alpha=0.45, linewidth=1.2, markersize=5, label=fn)
    ax.plot(EPSILONS, avg, "s-", color="black",
            linewidth=2.5, markersize=9, label="Average", zorder=5)
    for eps, err in zip(EPSILONS, avg):
        ax.annotate(f"{err:.1f}", (eps, err),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Epsilon", fontsize=13)
    ax.set_ylabel("Center Error (pixels)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(EPSILONS)
    ax.set_xlim(-0.005, 0.11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"그래프 저장 → {out_path}")


def plot_comparison(
    fgsm_results: dict,
    pgd_results: dict,
    filenames: list[str],
    out_path: Path,
) -> None:
    """FGSM vs PGD 비교 그래프.

    - 개별 이미지: 반투명 점선 (FGSM 파란 계열 / PGD 빨간 계열)
    - 평균:        굵은 실선 (FGSM 파란색 / PGD 빨간색)
    """
    fgsm_per = _extract_errors(fgsm_results, filenames)
    pgd_per  = _extract_errors(pgd_results,  filenames)
    fgsm_avg = _avg_errors(fgsm_per, filenames)
    pgd_avg  = _avg_errors(pgd_per,  filenames)

    fig, ax = plt.subplots(figsize=(10, 6))

    # ── 개별 이미지 선 (얇고 반투명) ────────────────────────────────────
    for i, fn in enumerate(filenames):
        label_f = fn.replace("vehicle_", "v").replace(".png", "")
        ax.plot(EPSILONS, fgsm_per[fn], "o--",
                color="#1565C0", alpha=0.18, linewidth=1.0, markersize=4)
        ax.plot(EPSILONS, pgd_per[fn],  "s--",
                color="#B71C1C", alpha=0.18, linewidth=1.0, markersize=4)
        # 마지막 이미지에만 범례용 레이블 추가
        if i == len(filenames) - 1:
            ax.plot([], [], "o--", color="#1565C0", alpha=0.45,
                    linewidth=1.0, markersize=4, label="FGSM 개별 이미지")
            ax.plot([], [], "s--", color="#B71C1C", alpha=0.45,
                    linewidth=1.0, markersize=4, label="PGD 개별 이미지")

    # ── 평균 실선 ────────────────────────────────────────────────────────
    ax.plot(EPSILONS, fgsm_avg, "o-",
            color="#1565C0", linewidth=2.8, markersize=10,
            label=f"FGSM 평균  (steps=1)", zorder=5)
    ax.plot(EPSILONS, pgd_avg, "s-",
            color="#C62828", linewidth=2.8, markersize=10,
            label=f"PGD 평균   (steps={PGD_STEPS}, alpha=ε/4)", zorder=5)

    # ── 평균 레이블 ──────────────────────────────────────────────────────
    for eps, err in zip(EPSILONS, fgsm_avg):
        ax.annotate(f"{err:.1f}", (eps, err),
                    textcoords="offset points", xytext=(-14, 8),
                    ha="center", fontsize=9, fontweight="bold", color="#1565C0")
    for eps, err in zip(EPSILONS, pgd_avg):
        ax.annotate(f"{err:.1f}", (eps, err),
                    textcoords="offset points", xytext=(14, 8),
                    ha="center", fontsize=9, fontweight="bold", color="#C62828")

    ax.set_xlabel("Epsilon", fontsize=13)
    ax.set_ylabel("바운딩 박스 중심 오차 (픽셀)", fontsize=13)
    ax.set_title("FGSM vs PGD 공격 강도별 바운딩 박스 오차 비교",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(EPSILONS)
    ax.set_xlim(-0.008, 0.115)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.3)

    # ── 주석: PGD 우위 표시 ──────────────────────────────────────────────
    for i, (eps, fa, pa) in enumerate(zip(EPSILONS, fgsm_avg, pgd_avg)):
        if eps == 0:
            continue
        diff = pa - fa
        if diff > 1.0:
            mid_y = (fa + pa) / 2
            ax.annotate(
                f"+{diff:.1f}px",
                xy=(eps, mid_y),
                textcoords="offset points", xytext=(18, 0),
                ha="left", fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"비교 그래프 저장 → {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"Device: {DEVICE}\n")

    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    filenames = sorted(gt.keys())

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    print(f"모델 로드 완료: {MODEL_PATH}\n")

    # ── FGSM ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("FGSM 공격 실행")
    print("=" * 60)
    fgsm_results = run_attack_loop(
        model, filenames, gt,
        attack_fn=fgsm_attack,
        attack_name="FGSM",
    )
    with open(OUT_JSON_FGSM, "w", encoding="utf-8") as f:
        json.dump(fgsm_results, f, indent=2, ensure_ascii=False)
    print(f"\nFGSM 결과 저장 → {OUT_JSON_FGSM}")

    fgsm_avg = _avg_errors(_extract_errors(fgsm_results, filenames), filenames)
    print("\n[FGSM] epsilon별 평균 중심 오차")
    for eps, err in zip(EPSILONS, fgsm_avg):
        print(f"  eps={eps:.2f}  {err:6.2f}px  {'#' * int(err / 2)}")

    # ── PGD ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"PGD 공격 실행  (steps={PGD_STEPS}, alpha=epsilon/4)")
    print("=" * 60)
    pgd_results = run_attack_loop(
        model, filenames, gt,
        attack_fn=pgd_attack,
        attack_name="PGD",
    )
    with open(OUT_JSON_PGD, "w", encoding="utf-8") as f:
        json.dump(pgd_results, f, indent=2, ensure_ascii=False)
    print(f"\nPGD 결과 저장 → {OUT_JSON_PGD}")

    pgd_avg = _avg_errors(_extract_errors(pgd_results, filenames), filenames)
    print("\n[PGD] epsilon별 평균 중심 오차")
    for eps, err in zip(EPSILONS, pgd_avg):
        print(f"  eps={eps:.2f}  {err:6.2f}px  {'#' * int(err / 2)}")

    # ── 비교 요약 ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{'epsilon':>8} | {'FGSM (px)':>10} | {'PGD (px)':>10} | {'PGD/FGSM':>10}")
    print("-" * 48)
    for eps, fa, pa in zip(EPSILONS, fgsm_avg, pgd_avg):
        ratio = pa / fa if fa > 0 else float("nan")
        print(f"{eps:>8.2f} | {fa:>10.2f} | {pa:>10.2f} | {ratio:>10.2f}x")
    print("=" * 60)

    # ── 그래프 저장 ───────────────────────────────────────────────────────
    plot_single(fgsm_results, filenames, OUT_PNG_FGSM)
    plot_comparison(fgsm_results, pgd_results, filenames, OUT_PNG_COMPARE)


if __name__ == "__main__":
    main()
