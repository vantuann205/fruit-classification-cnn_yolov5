#!/usr/bin/env python3
"""Generate before/after preprocessing figures and dataset distribution chart for T04."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from .preprocess import (
        CLASS_NAMES,
        AugmentConfig,
        apply_train_augmentations,
        denormalize_array,
        load_rgb_image,
        normalize_image,
        random_color_jitter,
        random_flip,
        random_resized_crop,
        random_rotate,
        resize_224,
    )
except ImportError:
    from preprocess import (
        CLASS_NAMES,
        AugmentConfig,
        apply_train_augmentations,
        denormalize_array,
        load_rgb_image,
        normalize_image,
        random_color_jitter,
        random_flip,
        random_resized_crop,
        random_rotate,
        resize_224,
    )

FONT_CANDIDATES = [
    "/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf",
    "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
BOLD_CANDIDATES = [
    "/usr/share/fonts/google-carlito-fonts/Carlito-Bold.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def load_font(size: int = 16, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = BOLD_CANDIDATES + FONT_CANDIDATES if bold else FONT_CANDIDATES
    for fp in candidates:
        if Path(fp).is_file():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


PREFERRED_SAMPLE_HASHES = {
    "Buoi": "a1d8f58a",
    "Hong": "4ed80c25",
    "dudu": "029ed772",
}


def pick_sample_images(data_dir: Path, seed: int = 7) -> dict[str, Path]:
    rng = random.Random(seed)
    chosen: dict[str, Path] = {}
    for cls in CLASS_NAMES:
        sha8 = PREFERRED_SAMPLE_HASHES.get(cls, "")
        matches = sorted((data_dir / "train" / cls).glob(f"*{sha8}.jpg")) if sha8 else []
        if not matches and sha8:
            matches = sorted(data_dir.rglob(f"*{sha8}.jpg"))
        if matches:
            chosen[cls] = matches[0]
            continue
        candidates = sorted((data_dir / "train" / cls).glob("*.jpg"))
        if not candidates:
            raise FileNotFoundError(f"No .jpg images found in {data_dir / 'train' / cls}")
        # Prefer larger/near-square real photos without pure-white text borders
        good = []
        for p in candidates[:200]:
            with Image.open(p) as im:
                w, h = im.size
                if min(w, h) >= 180 and max(w, h) / min(w, h) <= 1.45:
                    rgb = im.convert("RGB")
                    corners = [
                        rgb.getpixel((4, 4)),
                        rgb.getpixel((w - 5, 4)),
                        rgb.getpixel((4, h - 5)),
                        rgb.getpixel((w - 5, h - 5)),
                    ]
                    if sum(1 for c in corners if min(c) > 240) < 3:
                        good.append(p)
        pool = good if good else candidates
        chosen[cls] = rng.choice(pool[: min(40, len(pool))])
    return chosen


def generate_before_after_grid(samples: dict[str, Path], out_path: Path) -> None:
    """Create grid comparing Original -> Resize 224x224 -> 3 Augmented samples -> Normalized preview."""
    tile = 180
    pad = 14
    left_col = 110
    top_header = 68
    cols = ["Original\n(256px max)", "Eval Resize\n(224x224)", "Augment #1\n(Crop+Flip+Rot)", "Augment #2\n(Jitter+Rot)", "Augment #3\n(Random Seed)", "Normalized\n(Restored Check)"]

    width = left_col + len(cols) * (tile + pad) + pad
    height = top_header + len(CLASS_NAMES) * (tile + pad + 18) + pad
    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    f_title = load_font(20, bold=True)
    f_hdr = load_font(14, bold=True)
    f_cls = load_font(16, bold=True)
    f_sub = load_font(12, bold=False)

    draw.text((pad, 10), "Tiền xử lý & Augmentation (224×224) — Trước và Sau theo từng lớp", fill=(15, 23, 42), font=f_title)
    for c_idx, col_title in enumerate(cols):
        x = left_col + c_idx * (tile + pad) + tile // 2
        draw.multiline_text((x, 48), col_title, fill=(51, 65, 85), font=f_hdr, anchor="mm", align="center")

    cfg = AugmentConfig()
    cls_labels = {"Buoi": "Buoi\n(Pomelo)", "Hong": "Hong\n(Persimmon)", "dudu": "dudu\n(Papaya)"}

    for r_idx, cls in enumerate(CLASS_NAMES):
        y = top_header + r_idx * (tile + pad + 18)
        draw.multiline_text((left_col // 2, y + tile // 2), cls_labels[cls], fill=(15, 23, 42), font=f_cls, anchor="mm", align="center")

        orig = load_rgb_image(samples[cls])
        orig_box = Image.new("RGB", (tile, tile), (226, 232, 240))
        thumb = orig.copy()
        thumb.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        orig_box.paste(thumb, ((tile - thumb.width) // 2, (tile - thumb.height) // 2))

        eval_224 = resize_224(orig, 224, keep_aspect_center_crop=True).resize((tile, tile))
        aug1 = apply_train_augmentations(orig, cfg=cfg, seed=101 + r_idx).resize((tile, tile))
        aug2 = apply_train_augmentations(orig, cfg=cfg, seed=202 + r_idx).resize((tile, tile))
        aug3 = apply_train_augmentations(orig, cfg=cfg, seed=303 + r_idx).resize((tile, tile))

        norm_arr = normalize_image(apply_train_augmentations(orig, cfg=cfg, seed=101 + r_idx), model_name="mobilenet_v2")
        restored = denormalize_array(norm_arr, model_name="mobilenet_v2").resize((tile, tile))

        row_imgs = [orig_box, eval_224, aug1, aug2, aug3, restored]
        captions = [f"{orig.width}x{orig.height}", "224x224", "seed=" + str(101 + r_idx), "seed=" + str(202 + r_idx), "seed=" + str(303 + r_idx), "ImageNet mean/std"]

        for c_idx, (im, cap) in enumerate(zip(row_imgs, captions)):
            x = left_col + c_idx * (tile + pad)
            canvas.paste(im, (x, y))
            draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(203, 213, 225), width=1)
            draw.text((x + tile // 2, y + tile + 8), cap, fill=(100, 116, 139), font=f_sub, anchor="mm")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True)


def generate_step_breakdown(samples: dict[str, Path], out_path: Path) -> None:
    """Show each individual augmentation step cleanly across the 3 fruit classes."""
    tile = 172
    pad = 14
    left_col = 95
    top_header = 68
    steps = [
        "1. Original",
        "2. Resize 224x224",
        "3. Rotate (+20 deg)",
        "4. Horizontal Flip",
        "5. Brightness (+25%)",
        "6. Random Crop",
    ]

    width = left_col + len(steps) * (tile + pad) + pad
    height = top_header + len(CLASS_NAMES) * (tile + pad + 12) + pad
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_title = load_font(20, bold=True)
    f_hdr = load_font(14, bold=True)
    f_cls = load_font(16, bold=True)

    draw.text((pad, 12), "Chi tiết từng phép biến đổi trong pipeline src/preprocess.py", fill=(15, 23, 42), font=f_title)
    for c_idx, st in enumerate(steps):
        x = left_col + c_idx * (tile + pad) + tile // 2
        draw.text((x, 48), st, fill=(30, 41, 59), font=f_hdr, anchor="mm")

    for r_idx, cls in enumerate(CLASS_NAMES):
        y = top_header + r_idx * (tile + pad + 12)
        draw.text((left_col // 2, y + tile // 2), cls, fill=(15, 23, 42), font=f_cls, anchor="mm")

        orig = load_rgb_image(samples[cls])
        base_224 = resize_224(orig, 224, keep_aspect_center_crop=True)
        rot_224 = random_rotate(base_224, angle=20.0)
        flip_224 = random_flip(base_224, hflip_prob=1.0, vflip_prob=0.0, rng=random.Random(7))
        bright_224 = random_color_jitter(base_224, brightness=0.35, contrast=0.25, saturation=0.25, rng=random.Random(9 + r_idx))
        crop_224 = random_resized_crop(orig, img_size=224, scale=(0.65, 0.78), rng=random.Random(12 + r_idx))

        orig_box = Image.new("RGB", (tile, tile), (241, 245, 249))
        th = orig.copy()
        th.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        orig_box.paste(th, ((tile - th.width) // 2, (tile - th.height) // 2))

        for c_idx, im in enumerate([orig_box, base_224, rot_224, flip_224, bright_224, crop_224]):
            x = left_col + c_idx * (tile + pad)
            canvas.paste(im.resize((tile, tile)), (x, y))
            draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(203, 213, 225), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True)


def generate_distribution_chart(data_dir: Path, out_path: Path) -> None:
    """Draw a clean grouped bar chart of train/val/test counts per class."""
    manifest_path = data_dir / "split_manifest.csv"
    counts: dict[str, Counter] = {c: Counter() for c in CLASS_NAMES}
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[row["class"]][row["split"]] += 1
    else:
        for split in ("train", "val", "test"):
            for cls in CLASS_NAMES:
                counts[cls][split] = len(list((data_dir / split / cls).glob("*.jpg")))

    w, h = 880, 480
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_title = load_font(20, bold=True)
    f_sub = load_font(14, bold=False)
    f_lbl = load_font(14, bold=True)
    f_num = load_font(12, bold=True)

    total_all = sum(sum(counts[c].values()) for c in CLASS_NAMES)
    draw.text((40, 22), f"Phân bố Dataset sau làm sạch & chia 70/15/15 (Tổng: {total_all:,} ảnh)", fill=(15, 23, 42), font=f_title)
    draw.text((40, 48), "Chuẩn hóa cạnh dài 256px | 3 lớp: Buoi, Hong, dudu", fill=(100, 116, 139), font=f_sub)

    plot_l, plot_t, plot_r, plot_b = 80, 95, 830, 405
    max_val = max(counts[c][s] for c in CLASS_NAMES for s in ("train", "val", "test"))
    y_top = ((max_val // 200) + 1) * 200

    # Grid lines
    for i in range(6):
        val = round(y_top * i / 5)
        yy = plot_b - int((val / y_top) * (plot_b - plot_t))
        draw.line([(plot_l, yy), (plot_r, yy)], fill=(226, 232, 240), width=1)
        draw.text((plot_l - 10, yy), str(val), fill=(100, 116, 139), font=f_num, anchor="rm")

    draw.line([(plot_l, plot_t), (plot_l, plot_b)], fill=(148, 163, 184), width=2)
    draw.line([(plot_l, plot_b), (plot_r, plot_b)], fill=(148, 163, 184), width=2)

    colors = {"train": (37, 99, 235), "val": (16, 185, 129), "test": (245, 158, 11)}
    splits = ("train", "val", "test")
    group_w = (plot_r - plot_l) // len(CLASS_NAMES)
    bar_w = 48
    gap = 12

    for idx, cls in enumerate(CLASS_NAMES):
        cx = plot_l + idx * group_w + group_w // 2
        start_x = cx - (3 * bar_w + 2 * gap) // 2
        cls_total = sum(counts[cls].values())
        for s_idx, sp in enumerate(splits):
            val = counts[cls][sp]
            bx0 = start_x + s_idx * (bar_w + gap)
            bx1 = bx0 + bar_w
            bh = int((val / y_top) * (plot_b - plot_t))
            by0 = plot_b - bh
            draw.rectangle([bx0, by0, bx1, plot_b], fill=colors[sp])
            draw.text(((bx0 + bx1) // 2, by0 - 10), str(val), fill=(30, 41, 59), font=f_num, anchor="mm")
        draw.text((cx, plot_b + 20), f"{cls} (n={cls_total})", fill=(15, 23, 42), font=f_lbl, anchor="mm")

    # Legend
    lx = 560
    for sp, col in colors.items():
        draw.rectangle([lx, 50, lx + 16, 66], fill=col)
        draw.text((lx + 22, 58), sp, fill=(51, 65, 85), font=f_sub, anchor="lm")
        lx += 88

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("fruit_classfication/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("fruit_classfication/reports/figures"))
    args = parser.parse_args()

    samples = pick_sample_images(args.data_dir)
    generate_before_after_grid(samples, args.out_dir / "preprocess_before_after.png")
    generate_step_breakdown(samples, args.out_dir / "augmentation_steps.png")
    generate_distribution_chart(args.data_dir, args.out_dir / "dataset_distribution.png")
    print("Generated figures in:", args.out_dir)


if __name__ == "__main__":
    main()
