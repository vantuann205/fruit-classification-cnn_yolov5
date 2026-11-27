#!/usr/bin/env python3
"""Preprocessing and data augmentation utilities for fruit classification.

Supports shared 224x224 preprocessing, geometric/photometric augmentations
(rotation, flip, brightness/color jitter, random crop), and model-specific
normalization for PyTorch CNN (MobileNetV2/ResNet50), TensorFlow/Keras
MobileNetV2, and Ultralytics YOLOv5-cls.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageEnhance, ImageOps, ImageStat

CLASS_NAMES = ("Buoi", "Hong", "dudu")
DEFAULT_IMG_SIZE = 224

# Normalization specs per target model family.
# - PyTorch torchvision MobileNetV2 / ResNet50 & YOLOv5-cls expect [0, 1] scaled
#   RGB tensors normalized with ImageNet mean & std.
# - Keras MobileNetV2 (tf.keras.applications.mobilenet_v2.preprocess_input)
#   expects RGB values scaled linearly from [0, 255] to [-1, 1].
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "mobilenet_v2": {
        "framework": "pytorch",
        "input_size": (224, 224),
        "scale_01": True,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "description": "PyTorch torchvision MobileNetV2 (ImageNet normalization)",
    },
    "resnet50": {
        "framework": "pytorch",
        "input_size": (224, 224),
        "scale_01": True,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "description": "PyTorch torchvision ResNet50 (ImageNet normalization)",
    },
    "yolov5_cls": {
        "framework": "pytorch",
        "input_size": (224, 224),
        "scale_01": True,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "description": "Ultralytics YOLOv5-cls (ImageNet normalization, 224x224)",
    },
    "mobilenet_v2_keras": {
        "framework": "keras",
        "input_size": (224, 224),
        "scale_01": True,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "description": "TensorFlow/Keras MobileNetV2 (scaled to [-1, 1])",
    },
}

MODEL_ALIASES: dict[str, str] = {
    "cnn": "mobilenet_v2",
    "cnn_pytorch": "mobilenet_v2",
    "mobilenetv2": "mobilenet_v2",
    "cnn_keras": "mobilenet_v2_keras",
    "keras": "mobilenet_v2_keras",
    "yolov5": "yolov5_cls",
    "yolo": "yolov5_cls",
}


@dataclass(frozen=True)
class AugmentConfig:
    """Hyperparameters for training data augmentation."""

    img_size: int = DEFAULT_IMG_SIZE
    eval_resize_short: int = 256
    crop_scale: tuple[float, float] = (0.75, 1.0)
    crop_ratio: tuple[float, float] = (0.85, 1.15)
    rotation_degrees: float = 25.0
    hflip_prob: float = 0.5
    vflip_prob: float = 0.15
    brightness: float = 0.25
    contrast: float = 0.20
    saturation: float = 0.20
    hue: float = 0.03


def resolve_model_config(model_name: str) -> tuple[str, dict[str, Any]]:
    """Resolve a model name or alias to its canonical normalization spec."""
    key = model_name.strip().lower()
    canonical = MODEL_ALIASES.get(key, key)
    if canonical not in MODEL_CONFIGS:
        supported = sorted(set(MODEL_CONFIGS) | set(MODEL_ALIASES))
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")
    return canonical, MODEL_CONFIGS[canonical]


def load_rgb_image(source: str | Path | Image.Image) -> Image.Image:
    """Load an image path or PIL Image, apply EXIF orientation, and convert to RGB."""
    if isinstance(source, Image.Image):
        im = source.copy()
    else:
        with Image.open(source) as opened:
            opened.load()
            im = opened.copy()
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode == "P":
            im = im.convert("RGBA")
        if im.mode in ("RGBA", "LA"):
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    return im


def resize_224(
    img: Image.Image,
    img_size: int = DEFAULT_IMG_SIZE,
    keep_aspect_center_crop: bool = False,
) -> Image.Image:
    """Resize image to (img_size, img_size).

    When keep_aspect_center_crop is True, resizes the shorter edge to
    round(img_size / 0.875) (256px for 224px) then center-crops to square.
    """
    if not keep_aspect_center_crop:
        return img.resize((img_size, img_size), Image.Resampling.BILINEAR)

    w, h = img.size
    short_target = max(img_size, round(img_size / 0.875))
    scale = short_target / min(w, h)
    new_w, new_h = max(img_size, round(w * scale)), max(img_size, round(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    return resized.crop((left, top, left + img_size, top + img_size))


def random_resized_crop(
    img: Image.Image,
    img_size: int = DEFAULT_IMG_SIZE,
    scale: tuple[float, float] = (0.75, 1.0),
    ratio: tuple[float, float] = (0.85, 1.15),
    rng: random.Random | None = None,
) -> Image.Image:
    """Crop a random region matching torchvision RandomResizedCrop and resize."""
    gen = rng or random
    w, h = img.size
    area = w * h
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    for _ in range(10):
        target_area = gen.uniform(*scale) * area
        aspect = math.exp(gen.uniform(*log_ratio))
        cw = round(math.sqrt(target_area * aspect))
        ch = round(math.sqrt(target_area / aspect))
        if 0 < cw <= w and 0 < ch <= h:
            top = gen.randint(0, h - ch)
            left = gen.randint(0, w - cw)
            return img.crop((left, top, left + cw, top + ch)).resize(
                (img_size, img_size), Image.Resampling.BILINEAR
            )

    # Fallback to largest central crop matching target aspect ratio.
    in_ratio = w / h
    if in_ratio < min(ratio):
        cw = w
        ch = round(cw / min(ratio))
    elif in_ratio > max(ratio):
        ch = h
        cw = round(ch * max(ratio))
    else:
        cw, ch = w, h
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize(
        (img_size, img_size), Image.Resampling.BILINEAR
    )


def random_flip(
    img: Image.Image,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.0,
    rng: random.Random | None = None,
) -> Image.Image:
    """Apply random horizontal and/or vertical flips."""
    gen = rng or random
    out = img
    if hflip_prob > 0 and gen.random() < hflip_prob:
        out = ImageOps.mirror(out)
    if vflip_prob > 0 and gen.random() < vflip_prob:
        out = ImageOps.flip(out)
    return out


def random_rotate(
    img: Image.Image,
    max_degrees: float = 25.0,
    rng: random.Random | None = None,
    angle: float | None = None,
) -> Image.Image:
    """Rotate image by a random angle in [-max_degrees, +max_degrees]."""
    gen = rng or random
    deg = angle if angle is not None else gen.uniform(-max_degrees, max_degrees)
    mean_color = tuple(int(round(v)) for v in ImageStat.Stat(img).mean[:3])
    return img.rotate(deg, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=mean_color)


def random_color_jitter(
    img: Image.Image,
    brightness: float = 0.25,
    contrast: float = 0.20,
    saturation: float = 0.20,
    rng: random.Random | None = None,
) -> Image.Image:
    """Apply randomized brightness, contrast, and saturation adjustments."""
    gen = rng or random
    ops: list[tuple[Callable[[Image.Image], Any], float]] = [
        (ImageEnhance.Brightness, brightness),
        (ImageEnhance.Contrast, contrast),
        (ImageEnhance.Color, saturation),
    ]
    gen.shuffle(ops)
    out = img
    for enhancer_cls, delta in ops:
        if delta > 0:
            factor = gen.uniform(max(0.0, 1.0 - delta), 1.0 + delta)
            out = enhancer_cls(out).enhance(factor)
    return out


def apply_train_augmentations(
    source: str | Path | Image.Image,
    cfg: AugmentConfig = AugmentConfig(),
    seed: int | None = None,
) -> Image.Image:
    """Apply full training augmentation pipeline returning a 224x224 RGB PIL Image."""
    rng = random.Random(seed) if seed is not None else random
    img = load_rgb_image(source)
    img = random_resized_crop(img, cfg.img_size, cfg.crop_scale, cfg.crop_ratio, rng=rng)
    img = random_flip(img, cfg.hflip_prob, cfg.vflip_prob, rng=rng)
    img = random_rotate(img, cfg.rotation_degrees, rng=rng)
    img = random_color_jitter(img, cfg.brightness, cfg.contrast, cfg.saturation, rng=rng)
    return img


def normalize_image(
    img: Image.Image,
    model_name: str = "mobilenet_v2",
) -> Any:
    """Normalize a 224x224 RGB PIL Image according to the target model specification.

    Returns a float32 NumPy array of shape (H, W, 3) when NumPy is available,
    otherwise returns a nested Python float list of shape [H][W][3].
    """
    _, spec = resolve_model_config(model_name)
    target_size = spec["input_size"]
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)
    img = img.convert("RGB")

    mean = spec["mean"]
    std = spec["std"]

    try:
        import numpy as np

        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        return (arr - mean_arr) / std_arr
    except ImportError:
        w, h = img.size
        get_px = getattr(img, "get_flattened_data", img.getdata)
        raw = list(get_px())
        rows: list[list[list[float]]] = []
        for y in range(h):
            row: list[list[float]] = []
            off = y * w
            for x in range(w):
                r, g, b = raw[off + x]
                row.append([
                    (r / 255.0 - mean[0]) / std[0],
                    (g / 255.0 - mean[1]) / std[1],
                    (b / 255.0 - mean[2]) / std[2],
                ])
            rows.append(row)
        return rows


def denormalize_array(
    arr: Any,
    model_name: str = "mobilenet_v2",
) -> Image.Image:
    """Convert a normalized (H, W, 3) array/nested-list back into a displayable PIL Image."""
    _, spec = resolve_model_config(model_name)
    mean = spec["mean"]
    std = spec["std"]

    try:
        import numpy as np

        np_arr = np.asarray(arr, dtype=np.float32)
        if np_arr.ndim == 3 and np_arr.shape[0] == 3 and np_arr.shape[2] != 3:
            np_arr = np.transpose(np_arr, (1, 2, 0))
        restored = np_arr * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
        restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(restored, mode="RGB")
    except ImportError:
        h = len(arr)
        w = len(arr[0]) if h else 0
        out = Image.new("RGB", (w, h))
        pixels: list[tuple[int, int, int]] = []
        for row in arr:
            for pix in row:
                rgb = tuple(
                    max(0, min(255, round((pix[c] * std[c] + mean[c]) * 255.0)))
                    for c in range(3)
                )
                pixels.append(rgb)
        out.putdata(pixels)
        return out


def preprocess_image(
    source: str | Path | Image.Image,
    model_name: str = "mobilenet_v2",
    is_train: bool = False,
    cfg: AugmentConfig = AugmentConfig(),
    seed: int | None = None,
) -> Any:
    """End-to-end preprocessing: load -> (augment or eval resize 224x224) -> normalize."""
    if is_train:
        img = apply_train_augmentations(source, cfg=cfg, seed=seed)
    else:
        img = resize_224(load_rgb_image(source), img_size=cfg.img_size, keep_aspect_center_crop=True)
    return normalize_image(img, model_name=model_name)


def build_torchvision_transforms(
    model_name: str = "mobilenet_v2",
    is_train: bool = True,
    cfg: AugmentConfig = AugmentConfig(),
):
    """Build torchvision.transforms.Compose for Colab/PyTorch training and evaluation."""
    canonical, spec = resolve_model_config(model_name)
    if spec["framework"] != "pytorch":
        raise ValueError(
            f"build_torchvision_transforms is for PyTorch models, got '{canonical}' ({spec['framework']})."
        )
    from torchvision import transforms

    normalize = transforms.Normalize(mean=spec["mean"], std=spec["std"])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                cfg.img_size,
                scale=cfg.crop_scale,
                ratio=cfg.crop_ratio,
            ),
            transforms.RandomHorizontalFlip(p=cfg.hflip_prob),
            transforms.RandomVerticalFlip(p=cfg.vflip_prob),
            transforms.RandomRotation(degrees=cfg.rotation_degrees),
            transforms.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
                saturation=cfg.saturation,
                hue=cfg.hue,
            ),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize(cfg.eval_resize_short),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        normalize,
    ])


def build_dataloaders(
    data_dir: str | Path,
    model_name: str = "mobilenet_v2",
    batch_size: int = 32,
    num_workers: int = 2,
    cfg: AugmentConfig = AugmentConfig(),
):
    """Create PyTorch DataLoaders for train/val/test folders under data_dir."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    root = Path(data_dir)
    loaders = {}
    class_to_idx = None
    use_pin_memory = torch.cuda.is_available()

    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        tf = build_torchvision_transforms(
            model_name=model_name,
            is_train=(split == "train"),
            cfg=cfg,
        )
        ds = datasets.ImageFolder(root=str(split_dir), transform=tf)
        if class_to_idx is None:
            class_to_idx = ds.class_to_idx
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )

    return loaders, class_to_idx


def build_keras_datasets(
    data_dir: str | Path,
    model_name: str = "mobilenet_v2_keras",
    batch_size: int = 32,
    seed: int = 42,
    cfg: AugmentConfig = AugmentConfig(),
):
    """Build tf.data datasets with Keras augmentation & [-1, 1] MobileNetV2 normalization."""
    import tensorflow as tf

    canonical, spec = resolve_model_config(model_name)
    if spec["framework"] != "keras":
        raise ValueError(
            f"build_keras_datasets requires a Keras model config, got '{canonical}'."
        )

    root = Path(data_dir)
    img_shape = spec["input_size"]

    # Keras layers for train augmentation (flip, rotate, zoom/crop, brightness/contrast)
    augment_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(cfg.rotation_degrees / 360.0),
        tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.0), width_factor=(-0.2, 0.0)),
        tf.keras.layers.RandomBrightness(factor=cfg.brightness),
        tf.keras.layers.RandomContrast(factor=cfg.contrast),
    ], name="train_augmentation")

    # MobileNetV2 Keras expects [-1, 1] scaling: (x / 127.5) - 1.0
    rescale_layer = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)

    datasets_out = {}
    class_names = None
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        ds = tf.keras.utils.image_dataset_from_directory(
            str(split_dir),
            labels="inferred",
            label_mode="categorical",
            image_size=img_shape,
            batch_size=batch_size,
            shuffle=(split == "train"),
            seed=seed,
        )
        if class_names is None:
            class_names = ds.class_names
        if split == "train":
            ds = ds.map(
                lambda x, y: (rescale_layer(augment_layer(x, training=True)), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ds = ds.map(
                lambda x, y: (rescale_layer(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        datasets_out[split] = ds.prefetch(tf.data.AUTOTUNE)

    return datasets_out, class_names


def run_self_test(data_dir: Path | None = None) -> dict[str, Any]:
    """Verify PIL preprocessing, augmentations, and model-specific normalizations."""
    sample = Image.new("RGB", (320, 240), (90, 160, 70))
    cfg = AugmentConfig()

    aug1 = apply_train_augmentations(sample, cfg=cfg, seed=42)
    aug2 = apply_train_augmentations(sample, cfg=cfg, seed=42)
    assert aug1.size == (224, 224), f"Expected 224x224, got {aug1.size}"
    get_px1 = getattr(aug1, "get_flattened_data", aug1.getdata)
    get_px2 = getattr(aug2, "get_flattened_data", aug2.getdata)
    assert list(get_px1()) == list(get_px2()), "Seeded augmentation should be deterministic"

    eval_img = resize_224(sample, img_size=224, keep_aspect_center_crop=True)
    assert eval_img.size == (224, 224)

    summary: dict[str, Any] = {
        "augment_size": aug1.size,
        "deterministic_seed": True,
        "models": {},
    }

    for m_name in MODEL_CONFIGS:
        norm = preprocess_image(sample, model_name=m_name, is_train=False, cfg=cfg)
        flat = (
            norm.reshape(-1).tolist()
            if hasattr(norm, "reshape")
            else [v for row in norm for px in row for v in px]
        )
        vmin, vmax = min(flat), max(flat)
        if m_name == "mobilenet_v2_keras":
            assert -1.001 <= vmin <= vmax <= 1.001, f"Keras range invalid: {vmin}..{vmax}"
        else:
            assert -3.0 <= vmin <= vmax <= 3.0, f"ImageNet range invalid: {vmin}..{vmax}"
        restored = denormalize_array(norm, model_name=m_name)
        assert restored.size == (224, 224)
        summary["models"][m_name] = {"min": round(vmin, 4), "max": round(vmax, 4)}

    if data_dir and data_dir.is_dir():
        split_counts = {}
        for split in ("train", "val", "test"):
            sdir = data_dir / split
            if sdir.is_dir():
                split_counts[split] = {
                    c: len(list((sdir / c).glob("*.jpg")))
                    for c in CLASS_NAMES
                    if (sdir / c).is_dir()
                }
        summary["dataset_counts"] = split_counts

    for pkg in ("torch", "torchvision", "tensorflow"):
        try:
            __import__(pkg)
            summary[f"has_{pkg}"] = True
        except ImportError:
            summary[f"has_{pkg}"] = False

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fruit preprocessing and augmentation helper")
    parser.add_argument("--data-dir", type=Path, default=Path("fruit_classfication/data"))
    parser.add_argument("--self-test", action="store_true", help="Run preprocessing sanity checks")
    parser.add_argument("--print-config", action="store_true", help="Print model & augmentation configs")
    args = parser.parse_args()

    if args.print_config:
        print(json.dumps({"augment": asdict(AugmentConfig()), "models": MODEL_CONFIGS}, indent=2))
    if args.self_test or not args.print_config:
        res = run_self_test(args.data_dir)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
