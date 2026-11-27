#!/usr/bin/env python3
"""
Tiền xử lý và tăng cường dữ liệu ảnh cho bài toán phân loại trái cây.

Cấu trúc file:
  Phần 1 — Hằng số & cấu hình (class names, ImageNet mean/std, AugmentConfig)
  Phần 2 — Các hàm xử lý ảnh (viết tường minh bằng NumPy, giải thích từng bước)
  Phần 3 — Pipeline tổng hợp (preprocess_train / preprocess_eval)
  Phần 4 — DataLoader wrappers (torchvision & tf.keras, dùng khi huấn luyện trên GPU)
  Phần 5 — Self-test & CLI

Yêu cầu tối thiểu: Python 3.9+, Pillow, NumPy
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

# =============================================================================
# Phần 1 — Hằng số & cấu hình
# =============================================================================

# Tên 3 lớp trái cây trong dataset (phải khớp đúng với tên thư mục)
CLASS_NAMES = ("Buoi", "Hong", "dudu")

# Kích thước đầu vào chuẩn của CNN (MobileNetV2, ResNet50, YOLOv5-cls)
IMG_SIZE = 224

# ImageNet mean & std (theo kênh R, G, B) — dùng để chuẩn hoá đầu vào.
# Các mô hình pretrained (MobileNetV2, ResNet50, YOLOv5) đều được huấn luyện
# trên ImageNet với chuẩn hoá này, nên ta phải chuẩn hoá ảnh theo cùng cách.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class AugmentConfig:
    """
    Các siêu tham số (hyperparameter) cho bước tăng cường dữ liệu khi train.

    Giải thích lý do chọn từng giá trị:
    - crop_scale (0.75~1.0): cắt ngẫu nhiên 75%~100% diện tích gốc
      → giúp model học ở nhiều tỷ lệ zoom khác nhau
    - rotation (±25°): xoay nhẹ, trái cây trong thực tế không bao giờ hoàn toàn thẳng đứng
    - hflip_prob=0.5: lật ngang 50% — ảnh trái cây đối xứng trái/phải, label không đổi
    - brightness/contrast/saturation (±25%/±20%/±20%): mô phỏng thay đổi ánh sáng
    - hue=0.03 (rất nhỏ!): Bưởi xanh và Đu đủ xanh rất giống nhau về màu,
      nếu dịch chuyển hue nhiều thì model sẽ nhầm lẫn hai loại quả này
    """
    img_size:          int   = IMG_SIZE
    eval_resize_short: int   = 256       # resize cạnh ngắn trước khi center-crop
    crop_scale:        tuple = (0.75, 1.0)
    crop_ratio:        tuple = (0.85, 1.15)
    rotation_degrees:  float = 25.0
    hflip_prob:        float = 0.5
    brightness:        float = 0.25
    contrast:          float = 0.20
    saturation:        float = 0.20
    hue:               float = 0.03


# =============================================================================
# Phần 2 — Các hàm xử lý ảnh (tường minh, giải thích từng bước)
# =============================================================================


def load_image(source: str | Path | Image.Image) -> Image.Image:
    """
    Đọc ảnh từ đường dẫn (hoặc PIL Image sẵn có), sửa hướng EXIF, chuyển sang RGB.

    Tại sao cần bước này?
    - Nhiều ảnh chụp bằng điện thoại có metadata EXIF ghi hướng xoay (portrait/landscape).
      Nếu không sửa, ảnh sẽ bị xoay ngang dù hiển thị đúng trên máy.
    - Các ảnh PNG/RGBA có kênh alpha (trong suốt) → ghép lên nền trắng để giữ RGB.
    """
    if isinstance(source, Image.Image):
        img = source.copy()
    else:
        with Image.open(source) as f:
            f.load()
            img = f.copy()

    # Sửa hướng EXIF: đọc tag Orientation trong metadata và xoay/lật ảnh cho đúng
    img = ImageOps.exif_transpose(img)

    # Chuyển ảnh có kênh alpha (RGBA, PNG trong suốt) sang RGB bằng cách
    # ghép lên nền trắng → tránh mất thông tin màu khi normalize
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


# -----------------------------------------------------------------------------


def resize_and_center_crop(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """
    Resize ảnh sao cho cạnh ngắn = 256px, rồi cắt chính giữa thành size×size.

    Tại sao không resize thẳng về 224×224?
    - Resize thẳng làm biến dạng (méo) tỷ lệ của trái cây.
    - Cách đúng: resize cạnh ngắn → 256 (giữ tỷ lệ), rồi center-crop 224×224
      → ảnh vuông không méo, và phần trung tâm (thường chứa trái cây) được giữ lại.

    Dùng cho: ảnh validation & test (không augment, chỉ cần chuẩn hoá kích thước).
    """
    w, h = img.size
    short_side = min(w, h)

    # Tính tỷ lệ scale để cạnh ngắn = 256
    scale = 256 / short_side                        # ví dụ: 256/320 = 0.8
    new_w = max(size, round(w * scale))             # chiều rộng mới
    new_h = max(size, round(h * scale))             # chiều cao mới

    # Bước 1: resize giữ nguyên tỷ lệ (bilinear = nội suy song tuyến tính)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Bước 2: cắt vùng trung tâm size×size
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    img  = img.crop((left, top, left + size, top + size))

    return img  # PIL Image, kích thước (size, size)


# -----------------------------------------------------------------------------


def random_resized_crop(
    img: Image.Image,
    size: int = IMG_SIZE,
    scale: tuple = (0.75, 1.0),
    ratio: tuple = (0.85, 1.15),
    rng: random.Random | None = None,
) -> Image.Image:
    """
    Cắt một vùng ngẫu nhiên trên ảnh rồi resize về size×size.

    Tại sao cần bước này?
    - Giúp model học nhận diện trái cây ở nhiều vị trí và tỷ lệ khác nhau.
    - Trong thực tế, trái cây có thể chiếm 75% ảnh hoặc gần 100%, to hoặc nhỏ.

    Cách hoạt động:
    1. Chọn ngẫu nhiên diện tích vùng crop: area_crop = area_gốc × [0.75, 1.0]
    2. Chọn ngẫu nhiên tỷ lệ cạnh: aspect ∈ [0.85, 1.15]  (gần vuông)
    3. Từ area và aspect, tính cw × ch:  cw = √(area × aspect),  ch = √(area / aspect)
    4. Chọn ngẫu nhiên vị trí top-left trong ảnh gốc, cắt rồi resize → size×size.
    """
    gen = rng or random
    w, h = img.size
    area = w * h

    for _ in range(10):  # thử 10 lần để tìm crop hợp lệ
        target_area = gen.uniform(*scale) * area
        aspect      = math.exp(gen.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw = round(math.sqrt(target_area * aspect))   # crop width
        ch = round(math.sqrt(target_area / aspect))   # crop height

        if 0 < cw <= w and 0 < ch <= h:
            left = gen.randint(0, w - cw)
            top  = gen.randint(0, h - ch)
            return img.crop((left, top, left + cw, top + ch)).resize(
                (size, size), Image.Resampling.BILINEAR
            )

    # Fallback: lấy crop trung tâm lớn nhất nếu không tìm được vùng phù hợp
    r = w / h
    if r < ratio[0]:
        cw, ch = w, round(w / ratio[0])
    elif r > ratio[1]:
        ch, cw = h, round(h * ratio[1])
    else:
        cw, ch = w, h
    left = (w - cw) // 2
    top  = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


# -----------------------------------------------------------------------------


def flip_horizontal(img: Image.Image, prob: float = 0.5, rng: random.Random | None = None) -> Image.Image:
    """
    Lật ngang ảnh với xác suất `prob`.

    Tại sao cần lật ngang?
    - Trái cây đối xứng trái-phải (nhãn không thay đổi khi lật ngang).
    - Lật ngang gấp đôi số biến thể ảnh mà không cần thêm dữ liệu thực.
    - KHÔNG lật dọc quá nhiều vì trái cây trên cây/đĩa thường có hướng lên-xuống cố định.

    Phép toán (NumPy ý nghĩa): pixel[i, j] = pixel[i, W-1-j]
    → đảo thứ tự các cột từ phải sang trái.
    """
    gen = rng or random
    if gen.random() < prob:
        return ImageOps.mirror(img)   # PIL: đảo cột trái↔phải
    return img


# -----------------------------------------------------------------------------


def rotate_image(
    img: Image.Image,
    max_degrees: float = 25.0,
    rng: random.Random | None = None,
    angle: float | None = None,
) -> Image.Image:
    """
    Xoay ảnh một góc ngẫu nhiên trong [-max_degrees, +max_degrees] độ.

    Tham số `angle`: nếu truyền vào thì dùng góc cố định đó (hữu ích khi minh hoạ từng bước).
    Nếu không truyền thì lấy ngẫu nhiên trong [-max_degrees, +max_degrees].

    Tại sao cần xoay?
    - Trái cây trong ảnh thực tế không bao giờ hoàn toàn thẳng đứng.
    - Xoay giúp model tổng quát hoá tốt hơn với các góc nhìn khác nhau.

    Chi tiết kỹ thuật:
    - Dùng nội suy bilinear (BILINEAR) để ảnh sau xoay không bị "răng cưa".
    - Vùng trống sau khi xoay được lấp bằng màu trung bình của ảnh
      (thay vì màu đen) → tránh tạo ra cạnh màu đen giả tạo gây nhiễu cho model.
    - expand=False: giữ nguyên kích thước ảnh (không mở rộng khung hình).
    """
    gen = rng or random
    if angle is None:
        angle = gen.uniform(-max_degrees, max_degrees)   # góc xoay tính bằng độ

    # Tính màu trung bình của ảnh để lấp vùng trống sau xoay
    arr = np.asarray(img, dtype=np.float32)
    mean_color = tuple(int(arr[:, :, c].mean()) for c in range(3))  # (R̄, Ḡ, B̄)

    return img.rotate(angle, resample=Image.Resampling.BILINEAR,
                      expand=False, fillcolor=mean_color)


# -----------------------------------------------------------------------------


def adjust_brightness(arr: np.ndarray, factor: float) -> np.ndarray:
    """
    Điều chỉnh độ sáng bằng cách nhân toàn bộ pixel với factor.

    Công thức: pixel_mới = pixel_cũ × factor,  clip về [0, 255]

    factor > 1.0 → sáng hơn  (factor = 1.25 → sáng thêm 25%)
    factor < 1.0 → tối hơn   (factor = 0.75 → tối đi 25%)
    factor = 1.0 → giữ nguyên

    Tại sao cần?  Ảnh chụp trong nhà tối / ngoài trời sáng sẽ khác nhau rất nhiều.
    Ta mô phỏng sự thay đổi này để model không "thuộc lòng" điều kiện ánh sáng.
    """
    out = arr.astype(np.float32) * factor      # nhân từng pixel với factor
    return np.clip(out, 0, 255).astype(np.uint8)   # clip về [0, 255]


def adjust_contrast(arr: np.ndarray, factor: float) -> np.ndarray:
    """
    Điều chỉnh độ tương phản xung quanh giá trị trung bình của ảnh.

    Công thức: pixel_mới = (pixel_cũ − mean) × factor + mean,  clip [0, 255]

    factor > 1.0 → tương phản cao hơn (sáng→sáng hơn, tối→tối hơn)
    factor < 1.0 → tương phản thấp hơn (ảnh "xám xịt", ít nổi bật)
    factor = 1.0 → giữ nguyên

    Tại sao dùng mean?  Phép co giãn quanh mean đảm bảo giá trị trung bình
    của ảnh không đổi, chỉ có "khoảng cách" giữa sáng-tối thay đổi.
    """
    img_float = arr.astype(np.float32)
    mean = img_float.mean()                            # giá trị trung bình toàn ảnh
    out  = (img_float - mean) * factor + mean          # co giãn quanh mean
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_saturation(arr: np.ndarray, factor: float) -> np.ndarray:
    """
    Điều chỉnh độ bão hoà màu (saturation) qua không gian màu HSV.

    Tại sao cần đổi sang HSV?
    - Trong RGB, R/G/B đều mang thông tin cả sáng lẫn màu → khó tách riêng màu sắc.
    - Trong HSV: H = màu sắc, S = độ đậm màu, V = độ sáng → chỉnh S thuần tuý.

    Công thức:
      1. RGB → HSV (H ∈ [0,1], S ∈ [0,1], V ∈ [0,1])
      2. S_mới = S × factor,  clip về [0, 1]
      3. HSV → RGB

    factor > 1.0 → màu đậm hơn (vivid)
    factor < 1.0 → màu nhạt hơn, gần xám
    factor = 0.0 → ảnh trắng đen hoàn toàn

    Lưu ý: H (màu sắc) không thay đổi → màu Bưởi/Hồng/Đu đủ vẫn được bảo toàn.
    """
    img_float = arr.astype(np.float32) / 255.0   # scale về [0, 1]
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    # --- RGB → HSV (công thức chuẩn) ---
    Cmax = np.maximum(np.maximum(R, G), B)   # V = Cmax
    Cmin = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin                       # range = Cmax - Cmin

    # H (Hue): góc màu trên vòng tròn màu [0°, 360°] → scale về [0, 1]
    H = np.zeros_like(R)
    mask_r = (Cmax == R) & (delta > 0)
    mask_g = (Cmax == G) & (delta > 0)
    mask_b = (Cmax == B) & (delta > 0)
    H[mask_r] = ((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6
    H[mask_g] = (B[mask_g] - R[mask_g]) / delta[mask_g] + 2
    H[mask_b] = (R[mask_b] - G[mask_b]) / delta[mask_b] + 4
    H = H / 6.0  # về [0, 1]

    # S (Saturation): tỷ lệ "thuần sắc" so với tổng sáng
    # np.divide với where để tránh chia cho 0 khi pixel hoàn toàn tối (Cmax=0)
    S = np.divide(delta, Cmax, out=np.zeros_like(delta), where=Cmax > 0)

    # V (Value): độ sáng = kênh lớn nhất
    V = Cmax

    # --- Điều chỉnh độ bão hoà ---
    S_new = np.clip(S * factor, 0.0, 1.0)   # chỉ chỉnh S, H và V giữ nguyên

    # --- HSV → RGB (công thức chuẩn) ---
    C  = V * S_new                            # Chroma
    X  = C * (1 - np.abs((H * 6) % 2 - 1))  # giá trị phụ
    m  = V - C                                # offset sáng

    H6 = H * 6                               # sector [0, 6)
    R2 = np.select(
        [H6 < 1, H6 < 2, H6 < 3, H6 < 4, H6 < 5, H6 < 6],
        [C, X, 0*C, 0*C, X, C]
    )
    G2 = np.select(
        [H6 < 1, H6 < 2, H6 < 3, H6 < 4, H6 < 5, H6 < 6],
        [X, C, C, X, 0*C, 0*C]
    )
    B2 = np.select(
        [H6 < 1, H6 < 2, H6 < 3, H6 < 4, H6 < 5, H6 < 6],
        [0*C, 0*C, X, C, C, X]
    )

    result = np.stack([R2 + m, G2 + m, B2 + m], axis=2)
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def color_jitter(
    img: Image.Image,
    brightness: float = 0.25,
    contrast:   float = 0.20,
    saturation: float = 0.20,
    rng: random.Random | None = None,
) -> Image.Image:
    """
    Áp dụng ngẫu nhiên brightness, contrast, saturation theo thứ tự ngẫu nhiên.

    Tại sao áp dụng ngẫu nhiên thứ tự?
    - Brightness trước Contrast cho kết quả khác Contrast trước Brightness.
    - Xáo trộn thứ tự tạo ra đa dạng hơn các biến thể ảnh.

    Với mỗi phép biến đổi, factor được lấy ngẫu nhiên trong [1-δ, 1+δ]:
      brightness: factor ∈ [0.75, 1.25]  (±25%)
      contrast:   factor ∈ [0.80, 1.20]  (±20%)
      saturation: factor ∈ [0.80, 1.20]  (±20%)
    """
    gen = rng or random
    arr = np.asarray(img, dtype=np.uint8).copy()

    # Danh sách các phép biến đổi kèm cường độ delta
    ops = [
        (adjust_brightness, brightness),
        (adjust_contrast,   contrast),
        (adjust_saturation, saturation),
    ]
    gen.shuffle(ops)   # xáo trộn thứ tự áp dụng

    for fn, delta in ops:
        # factor ∈ [1 - delta, 1 + delta], đảm bảo không âm
        factor = gen.uniform(max(0.0, 1.0 - delta), 1.0 + delta)
        arr = fn(arr, factor)

    return Image.fromarray(arr, mode="RGB")


# -----------------------------------------------------------------------------


def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Chuẩn hoá ảnh theo ImageNet mean & std.

    Công thức (theo từng kênh RGB):
        pixel_chuẩn_hoá = (pixel / 255.0 − mean) / std

    Bước 1: pixel / 255.0  → đưa giá trị từ [0, 255] về [0.0, 1.0]
    Bước 2: (pixel − mean) → trừ giá trị trung bình của ImageNet
    Bước 3: / std          → chia độ lệch chuẩn của ImageNet

    Kết quả: float32 array shape (H, W, 3), giá trị khoảng [-2.0, +2.0]

    Tại sao dùng ImageNet mean/std?
    - MobileNetV2, ResNet50, YOLOv5 được pre-train trên ImageNet với chuẩn hoá này.
    - Nếu không chuẩn hoá đúng cách, phân phối đầu vào sẽ khác với lúc pre-train
      → model không tận dụng được trọng số đã học (transfer learning kém hiệu quả).
    """
    img_float = arr.astype(np.float32) / 255.0            # [0, 255] → [0.0, 1.0]
    normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD  # (H, W, 3) broadcast
    return normalized.astype(np.float32)


def denormalize(arr: np.ndarray) -> Image.Image:
    """
    Đảo ngược chuẩn hoá: float32 (H, W, 3) → PIL Image uint8.

    Công thức ngược lại:
        pixel_gốc = (pixel_chuẩn_hoá × std + mean) × 255

    Dùng để: hiển thị lại ảnh sau khi đã chuẩn hoá (ví dụ kiểm tra kết quả).
    """
    # Nếu array có shape (3, H, W) thì chuyển về (H, W, 3)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    restored = arr * IMAGENET_STD + IMAGENET_MEAN    # đảo chuẩn hoá
    restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(restored, mode="RGB")


# =============================================================================
# Phần 3 — Pipeline tổng hợp
# =============================================================================


def preprocess_train(
    source: str | Path | Image.Image,
    cfg: AugmentConfig = AugmentConfig(),
    seed: int | None = None,
) -> np.ndarray:
    """
    Pipeline tiền xử lý cho ảnh TRAIN (có tăng cường dữ liệu).

    Các bước theo thứ tự:
      1. load_image          → đọc ảnh, sửa EXIF, chuyển RGB
      2. random_resized_crop → cắt vùng ngẫu nhiên 75~100% → resize 224×224
      3. flip_horizontal     → lật ngang với xác suất 50%
      4. rotate_image        → xoay ±25°
      5. color_jitter        → thay đổi brightness/contrast/saturation ngẫu nhiên
      6. normalize           → chuẩn hoá ImageNet → float32 (H, W, 3)

    Trả về: np.ndarray float32, shape (224, 224, 3), giá trị ≈ [-2, +2]
    """
    rng = random.Random(seed) if seed is not None else random

    img = load_image(source)                                           # Bước 1
    img = random_resized_crop(img, cfg.img_size, cfg.crop_scale,
                              cfg.crop_ratio, rng=rng)                 # Bước 2
    img = flip_horizontal(img, prob=cfg.hflip_prob, rng=rng)          # Bước 3
    img = rotate_image(img, max_degrees=cfg.rotation_degrees, rng=rng) # Bước 4
    img = color_jitter(img, cfg.brightness, cfg.contrast,
                       cfg.saturation, rng=rng)                        # Bước 5

    arr = np.asarray(img, dtype=np.uint8)
    return normalize(arr)                                              # Bước 6


def preprocess_eval(
    source: str | Path | Image.Image,
    cfg: AugmentConfig = AugmentConfig(),
) -> np.ndarray:
    """
    Pipeline tiền xử lý cho ảnh VAL / TEST (không tăng cường).

    Các bước:
      1. load_image              → đọc ảnh, sửa EXIF, chuyển RGB
      2. resize_and_center_crop  → resize cạnh ngắn=256, center-crop 224×224
      3. normalize               → chuẩn hoá ImageNet → float32 (H, W, 3)

    Không augment vì: Val/Test dùng để đánh giá chính xác, cần ảnh chuẩn, không nhiễu.

    Trả về: np.ndarray float32, shape (224, 224, 3)
    """
    img = load_image(source)                                # Bước 1
    img = resize_and_center_crop(img, size=cfg.img_size)    # Bước 2
    arr = np.asarray(img, dtype=np.uint8)
    return normalize(arr)                                   # Bước 3


# =============================================================================
# Phần 4 — DataLoader wrappers (dùng khi huấn luyện trên GPU/Colab)
# =============================================================================
#
# Tại sao cần phần này khi đã có preprocess_train/eval ở trên?
# → preprocess_train/eval dùng NumPy thuần → chậm, chỉ xử lý 1 ảnh 1 lần.
# → Khi train trên GPU (Colab), cần xử lý batch 32+ ảnh song song trên CPU.
# → torchvision / tf.data tối ưu cho việc này: multi-thread, pipeline với GPU.
# → Hai hàm bên dưới dùng CÙNG tham số AugmentConfig → kết quả tương đương.
#


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    cfg: AugmentConfig = AugmentConfig(),
):
    """
    Tạo PyTorch DataLoader cho train/val/test (dùng torchvision để tối ưu tốc độ).

    Trả về:
      loaders      — dict {"train": DataLoader, "val": DataLoader, "test": DataLoader}
      class_to_idx — dict {"Buoi": 0, "Hong": 1, "dudu": 2}
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    mean = IMAGENET_MEAN.tolist()
    std  = IMAGENET_STD.tolist()

    # Transform cho train: augment + normalize
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=cfg.crop_scale, ratio=cfg.crop_ratio),
        transforms.RandomHorizontalFlip(p=cfg.hflip_prob),
        transforms.RandomVerticalFlip(p=0.0),          # không lật dọc
        transforms.RandomRotation(degrees=cfg.rotation_degrees),
        transforms.ColorJitter(
            brightness=cfg.brightness,
            contrast=cfg.contrast,
            saturation=cfg.saturation,
            hue=cfg.hue,                               # hue nhỏ (0.03) vì Bưởi/Đu đủ cùng màu xanh
        ),
        transforms.ToTensor(),                         # [0,255] uint8 → [0,1] float32, (H,W,3)→(3,H,W)
        transforms.Normalize(mean=mean, std=std),      # chuẩn hoá ImageNet
    ])

    # Transform cho val/test: chỉ resize + crop + normalize (không augment)
    eval_tf = transforms.Compose([
        transforms.Resize(cfg.eval_resize_short),      # cạnh ngắn → 256px
        transforms.CenterCrop(cfg.img_size),           # center crop 224×224
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    root = Path(data_dir)
    loaders = {}
    class_to_idx = None
    use_pin_memory = torch.cuda.is_available()         # pin_memory chỉ hữu ích khi có GPU

    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        tf = train_tf if split == "train" else eval_tf
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
    batch_size: int = 32,
    seed: int = 42,
    cfg: AugmentConfig = AugmentConfig(),
):
    """
    Tạo tf.data.Dataset cho train/val/test với Keras (dùng cho TensorFlow/Keras MobileNetV2).

    Khác với PyTorch:
    - Keras MobileNetV2 cần đầu vào được scale về [-1, 1]: pixel_mới = pixel/127.5 - 1
      (thay vì chuẩn hoá ImageNet của PyTorch)

    Trả về:
      datasets_out — dict {"train": tf.data.Dataset, "val": ..., "test": ...}
      class_names  — list ["Buoi", "Hong", "dudu"]
    """
    import tensorflow as tf

    root = Path(data_dir)

    # Augmentation layers cho train
    augment_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(cfg.rotation_degrees / 360.0),
        tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.0), width_factor=(-0.2, 0.0)),
        tf.keras.layers.RandomBrightness(factor=cfg.brightness),
        tf.keras.layers.RandomContrast(factor=cfg.contrast),
    ], name="train_augmentation")

    # Scale [-1, 1]: (x / 127.5) - 1.0  (yêu cầu của Keras MobileNetV2)
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
            image_size=(cfg.img_size, cfg.img_size),
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


# =============================================================================
# Phần 5 — Self-test & CLI
# =============================================================================


def run_self_test(data_dir: Path | None = None) -> dict:
    """
    Kiểm tra nhanh toàn bộ pipeline với ảnh giả (không cần dataset thực).

    Kiểm tra:
    1. preprocess_train cho ra shape (224, 224, 3), deterministic với seed cố định
    2. preprocess_eval cho ra shape (224, 224, 3)
    3. Giá trị sau normalize nằm trong khoảng hợp lệ (≈ -3.0 đến +3.0)
    4. denormalize trả về ảnh PIL đúng kích thước
    5. (nếu có data_dir) đếm số ảnh trong mỗi split
    """
    # Tạo ảnh giả màu xanh lá (giống Bưởi/Đu đủ)
    sample = Image.new("RGB", (320, 240), (90, 160, 70))
    cfg = AugmentConfig()

    # Kiểm tra 1: train pipeline, kết quả có shape đúng
    out_train = preprocess_train(sample, cfg=cfg, seed=42)
    assert out_train.shape == (224, 224, 3), f"Shape sai: {out_train.shape}"

    # Kiểm tra 2: seed cố định → kết quả giống nhau (deterministic)
    out_train2 = preprocess_train(sample, cfg=cfg, seed=42)
    assert np.allclose(out_train, out_train2), "Seed cố định nhưng kết quả khác nhau!"

    # Kiểm tra 3: eval pipeline
    out_eval = preprocess_eval(sample, cfg=cfg)
    assert out_eval.shape == (224, 224, 3), f"Eval shape sai: {out_eval.shape}"

    # Kiểm tra 4: giá trị sau normalize hợp lệ
    vmin, vmax = float(out_eval.min()), float(out_eval.max())
    assert -3.5 <= vmin <= vmax <= 3.5, f"Giá trị normalize ngoài khoảng: {vmin}..{vmax}"

    # Kiểm tra 5: denormalize trả về ảnh PIL đúng
    restored = denormalize(out_eval)
    assert restored.size == (224, 224), f"Denormalize size sai: {restored.size}"

    summary: dict = {
        "train_shape": list(out_train.shape),
        "eval_shape":  list(out_eval.shape),
        "normalized_min": round(vmin, 4),
        "normalized_max": round(vmax, 4),
        "deterministic_seed": True,
    }

    # Đếm số ảnh trong dataset (nếu có)
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

    # Kiểm tra các thư viện bổ sung có sẵn không
    for pkg in ("torch", "torchvision", "tensorflow"):
        try:
            __import__(pkg)
            summary[f"has_{pkg}"] = True
        except ImportError:
            summary[f"has_{pkg}"] = False

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý ảnh trái cây")
    parser.add_argument("--data-dir", type=Path, default=Path("fruit_classfication/data"),
                        help="Thư mục data chứa train/val/test")
    parser.add_argument("--self-test", action="store_true",
                        help="Chạy kiểm tra nhanh toàn bộ pipeline")
    parser.add_argument("--print-config", action="store_true",
                        help="In cấu hình augmentation")
    args = parser.parse_args()

    if args.print_config:
        print(json.dumps(asdict(AugmentConfig()), indent=2, ensure_ascii=False))

    if args.self_test or not args.print_config:
        result = run_self_test(args.data_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
