"""군용 차량 시뮬레이션 샘플 이미지 생성 스크립트."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 1000, 500
random.seed(42)

# 배경 갈색 지형 색상 팔레트 (밝기 변화 포함)
BG_COLORS = [
    (139, 115, 85),
    (160, 130, 90),
    (120, 100, 70),
    (150, 120, 80),
    (145, 110, 75),
]

# 어두운 초록색 계열 (차량 색상)
VEHICLE_COLORS = [
    (34, 70, 34),
    (40, 80, 40),
    (30, 60, 30),
    (45, 85, 35),
    (38, 75, 38),
]

# 각 이미지별 바운딩 박스 정보 저장 (GT로 활용)
gt_annotations = {}


def draw_terrain_texture(draw: ImageDraw.ImageDraw, bg_color: tuple) -> None:
    """갈색 지형에 노이즈 패턴을 추가한다."""
    r, g, b = bg_color
    for _ in range(3000):
        x = random.randint(0, IMG_W - 1)
        y = random.randint(0, IMG_H - 1)
        delta = random.randint(-20, 20)
        color = (
            max(0, min(255, r + delta)),
            max(0, min(255, g + delta)),
            max(0, min(255, b + delta)),
        )
        draw.point((x, y), fill=color)


def draw_vehicle(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
) -> None:
    """어두운 초록색 군용 차량 사각형을 그린다."""
    draw.rectangle([x1, y1, x2, y2], fill=color, outline=(20, 40, 20), width=3)

    # 차량 내부 디테일 (창문/포탑 느낌)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = x2 - x1, y2 - y1
    turret_r = min(w, h) // 6
    turret_color = (color[0] - 10, color[1] - 10, color[2] - 10)
    draw.ellipse(
        [cx - turret_r, cy - turret_r, cx + turret_r, cy + turret_r],
        fill=turret_color,
    )


for img_idx in range(5):
    img = Image.new("RGB", (IMG_W, IMG_H), BG_COLORS[img_idx])
    draw = ImageDraw.Draw(img)

    draw_terrain_texture(draw, BG_COLORS[img_idx])

    # 이미지당 차량 1~3대 배치
    num_vehicles = random.randint(1, 3)
    boxes = []
    for _ in range(num_vehicles):
        w = random.randint(80, 180)
        h = random.randint(50, 100)
        x1 = random.randint(50, IMG_W - w - 50)
        y1 = random.randint(50, IMG_H - h - 50)
        x2, y2 = x1 + w, y1 + h
        draw_vehicle(draw, x1, y1, x2, y2, VEHICLE_COLORS[img_idx])
        boxes.append([x1, y1, x2, y2])

    filename = f"vehicle_{img_idx + 1:02d}.png"
    img.save(OUTPUT_DIR / filename)
    gt_annotations[filename] = boxes
    print(f"  저장: {filename}  차량 {num_vehicles}대  boxes={boxes}")

# GT 바운딩 박스를 JSON으로 저장 (학습 레이블로 활용)
gt_path = Path(__file__).parent.parent / "data" / "gt_annotations.json"
with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(gt_annotations, f, indent=2)

print(f"\n완료: 이미지 5장 → {OUTPUT_DIR}")
print(f"GT 어노테이션 → {gt_path}")
