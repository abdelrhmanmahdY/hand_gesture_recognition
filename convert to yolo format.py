"""
ASL Alphabet Dataset → YOLOv5/v8 Format Converter
===================================================
Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Sampling strategy — 200 images per class from 4 ranges:
    Range 1:    1  –  750   → 50 images (indices 0–749)
    Range 2:  751  – 1250   → 50 images (indices 750–1249)
    Range 3: 1251  – 1700   → 50 images (indices 1250–1699)
    Range 4: 2001  – 3000   → 50 images (indices 2000–2999)
    ─────────────────────────────────────────────────────
    Total per class:          200 images

Since images are already CROPPED (one sign per image),
the bounding box = the entire image → 0.5 0.5 1.0 1.0
"""

import shutil
import random
import yaml
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG — update DATASET_ROOT to match your machine
# ─────────────────────────────────────────────────────────────
DATASET_ROOT = "asl_alphabet_train/asl_alphabet_train"
OUTPUT_DIR   = "./asl_yolo"   # output folder (created automatically)
VAL_SPLIT    = 0.15           # 15% of 200 = 30 val, 170 train per class
SEED         = 42

# Sampling ranges: (start_index_inclusive, end_index_exclusive, how_many_to_pick)
# Images are sorted by name before indexing, so index 0 = first file alphabetically
RANGES = [
    (0,    750,  100),   # Range 1:    1 –  750  → pick 50
    (750,  1250, 100),   # Range 2:  751 – 1250  → pick 50
    (1250, 1700, 100),   # Range 3: 1251 – 1700  → pick 50
    (2000, 3000, 100),   # Range 4: 2001 – 3000  → pick 50
]
# ─────────────────────────────────────────────────────────────

random.seed(SEED)


def get_classes(root: Path) -> list[str]:
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    print(f"✅ Found {len(classes)} classes: {classes}\n")
    return classes


def sample_from_ranges(images: list[Path], ranges: list[tuple]) -> list[Path]:
    """
    Sort images by filename, then pick `n` random samples
    from each (start, end) slice.
    """
    images_sorted = sorted(images, key=lambda p: p.name)
    total         = len(images_sorted)
    selected      = []

    for (start, end, n) in ranges:
        # Clamp end to actual list length
        actual_end  = min(end, total)
        actual_start = min(start, actual_end)
        bucket      = images_sorted[actual_start:actual_end]

        if len(bucket) < n:
            print(f"  ⚠️  Range [{start}–{end}]: only {len(bucket)} images, taking all.")
            selected.extend(bucket)
        else:
            selected.extend(random.sample(bucket, n))

    return selected


def write_label(label_path: Path, class_id: int):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        # class_id  x_center  y_center  width  height  (normalized)
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def convert(dataset_root: str, output_dir: str, val_split: float):
    root     = Path(dataset_root)
    out      = Path(output_dir)
    classes  = get_classes(root)
    class2id = {cls: idx for idx, cls in enumerate(classes)}

    # Create folder structure
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_train = total_val = 0

    for cls_name, cls_id in class2id.items():
        cls_folder = root / cls_name
        all_images = (
            list(cls_folder.glob("*.jpg")) +
            list(cls_folder.glob("*.jpeg")) +
            list(cls_folder.glob("*.png"))
        )

        if not all_images:
            print(f"⚠️  No images in: {cls_folder}")
            continue

        # ── Sample 200 images across the 4 ranges ──
        selected = sample_from_ranges(all_images, RANGES)

        # ── Train / Val split ──
        random.shuffle(selected)
        n_val  = max(1, int(len(selected) * val_split))
        splits = {
            "val":   selected[:n_val],
            "train": selected[n_val:],
        }

        for split, split_imgs in splits.items():
            for img_path in split_imgs:
                dst_img    = out / "images" / split / f"{cls_name}_{img_path.name}"
                label_path = out / "labels" / split / f"{cls_name}_{img_path.stem}.txt"

                shutil.copy2(img_path, dst_img)
                write_label(label_path, cls_id)

        n_tr = len(splits["train"])
        n_vl = len(splits["val"])
        total_train += n_tr
        total_val   += n_vl
        print(f"  [{cls_id:02d}] {cls_name:10s}  selected={len(selected):3d}  "
              f"train={n_tr:3d}  val={n_vl:2d}")

    # ── data.yaml ───────────────────────────────────────────
    yaml_path = out / "data.yaml"
    yaml_data = {
        "path"  : str(out.resolve()),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : len(classes),
        "names" : classes,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("✅  Conversion complete!")
    print(f"   Classes        : {len(classes)}")
    print(f"   Images/class   : ~200  (50 × 4 ranges)")
    print(f"   Total train    : {total_train}")
    print(f"   Total val      : {total_val}")
    print(f"   Output dir     : {out.resolve()}")
    print(f"   YAML           : {yaml_path.resolve()}")
    print("=" * 55)
    print("\n🚀  Train YOLOv8 immediately:")
    print(f"   pip install ultralytics")
    print(f"   yolo detect train data={yaml_path.resolve()} \\\n"
          f"        model=yolov8n.pt epochs=50 imgsz=200 batch=32")


if __name__ == "__main__":
    if not Path(DATASET_ROOT).exists():
        print(f"❌ Dataset not found at:\n   {DATASET_ROOT}")
        print("\n   Update DATASET_ROOT at the top of this script.")
    else:
        convert(DATASET_ROOT, OUTPUT_DIR, VAL_SPLIT)