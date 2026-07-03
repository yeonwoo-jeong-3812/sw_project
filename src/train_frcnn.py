"""Faster R-CNN 파인튜닝 및 바운딩 박스 추론 스크립트.

클래스: 0=배경, 1=장애물(군용 차량)
학습 후 테스트 이미지 추론 결과를 results/bbox_clean.json 에 저장한다.
"""
import json
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ── 경로 설정 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "images"
GT_PATH = ROOT / "data" / "gt_annotations.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_JSON = RESULTS_DIR / "bbox_clean.json"

# ── 하이퍼파라미터 ──────────────────────────────────────────────────────────
NUM_CLASSES = 2       # 0: 배경, 1: 장애물
NUM_EPOCHS = 3
LR = 0.005
BATCH_SIZE = 2
SCORE_THRESH = 0.3    # 추론 시 신뢰도 임계값
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 데이터셋 ───────────────────────────────────────────────────────────────
class VehicleDataset(Dataset):
    def __init__(self, img_dir: Path, gt_path: Path) -> None:
        with open(gt_path, encoding="utf-8") as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.filenames = sorted(self.annotations.keys())

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        img = Image.open(self.img_dir / fname).convert("RGB")
        tensor_img = TF.to_tensor(img)  # [C, H, W], float32 [0,1]

        boxes_list = self.annotations[fname]
        boxes = torch.tensor(boxes_list, dtype=torch.float32)
        labels = torch.ones(len(boxes_list), dtype=torch.int64)  # 모두 클래스 1

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        return tensor_img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ── 모델 구성 ──────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> torch.nn.Module:
    """사전 학습된 Faster R-CNN의 분류기 헤드를 교체한다."""
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ── 학습 루프 ──────────────────────────────────────────────────────────────
def train(model: torch.nn.Module, loader: DataLoader) -> None:
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        print(f"  Epoch [{epoch:02d}/{NUM_EPOCHS}]  loss={avg:.4f}")


# ── 추론 및 결과 저장 ─────────────────────────────────────────────────────
def infer_and_save(model: torch.nn.Module, dataset: VehicleDataset) -> dict:
    model.eval()
    results = {}

    with torch.no_grad():
        for idx in range(len(dataset)):
            fname = dataset.filenames[idx]
            img_tensor, _ = dataset[idx]
            img_tensor = img_tensor.to(DEVICE)

            outputs = model([img_tensor])[0]

            # 신뢰도 임계값 필터링
            keep = outputs["scores"] >= SCORE_THRESH
            boxes = outputs["boxes"][keep].cpu().tolist()
            scores = outputs["scores"][keep].cpu().tolist()
            labels = outputs["labels"][keep].cpu().tolist()

            results[fname] = {
                "boxes": [[round(c, 2) for c in b] for b in boxes],
                "scores": [round(s, 4) for s in scores],
                "labels": labels,
            }
            print(f"  {fname}: {len(boxes)}개 검출  scores={[round(s,3) for s in scores]}")

    with open(CLEAN_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장 완료 → {CLEAN_JSON}")
    return results


# ── 메인 ──────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"torchvision: {torchvision.__version__}")

    dataset = VehicleDataset(DATA_DIR, GT_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f"데이터셋 크기: {len(dataset)}장\n")

    model = build_model(NUM_CLASSES).to(DEVICE)

    print("── 학습 시작 ──────────────────────────────")
    train(model, loader)

    print("\n── 추론 시작 ──────────────────────────────")
    infer_and_save(model, dataset)

    # 모델 가중치 저장 (FGSM 공격 단계에서 재사용)
    model_path = RESULTS_DIR / "frcnn_finetuned.pth"
    torch.save(model.state_dict(), model_path)
    print(f"모델 가중치 저장 → {model_path}")


if __name__ == "__main__":
    main()
