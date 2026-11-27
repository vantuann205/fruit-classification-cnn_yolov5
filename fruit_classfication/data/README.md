# Fruit Classification Dataset (`Buoi`, `Hong`, `dudu`)

Bộ dữ liệu phân loại 3 loại trái cây Việt Nam (**Bưởi**, **Hồng**, **Đu đủ**) đã được thu thập bổ sung, làm sạch, khử trùng lặp, chuẩn hóa kích thước và chia sẵn `train / val / test` theo tỉ lệ **70% / 15% / 15%** để cả hai nhánh mô hình **CNN (MobileNetV2 / ResNet50)** và **YOLOv5-cls** dùng chung.

## 1. Thống kê số lượng ảnh

| Lớp | Tên tiếng Việt / Khoa học | Train (70%) | Val (15%) | Test (15%) | Tổng |
|---|---|---:|---:|---:|---:|
| `Buoi` | Bưởi (*Citrus maxima*) | 1,040 | 223 | 224 | **1,487** |
| `Hong` | Quả hồng (*Diospyros kaki*) | 1,174 | 251 | 253 | **1,678** |
| `dudu` | Đu đủ (*Carica papaya*) | 1,291 | 276 | 278 | **1,845** |
| **Tổng cộng** | | **3,505** | **750** | **755** | **5,010** |

- `class_to_idx.json`: `{"Buoi": 0, "Hong": 1, "dudu": 2}`
- Dung lượng thư mục `data/` sau resize cạnh dài `256px`: nhẹ, phù hợp đẩy trực tiếp lên GitHub và clone nhanh trên Google Colab.

## 2. Nguồn thu thập dữ liệu thô (T02)

Tổng cộng **8,344** ảnh thô đã được thu thập cục bộ (`Buoi`: 2,513; `Hong`: 2,760; `dudu`: 3,071; giữ ở máy cục bộ trong `raw_data/`, không đẩy lên GitHub) từ các nguồn:

1. **Kaggle / GitHub datasets**:
   - Tập `Fruit.zip` tổng hợp từ **Fruits-262** (`pomelo`, `persimmon`, `papaya`), **Fruits-360** (`Pomelo Sweetie 1`, `Kaki 1`, `Papaya 1`, `Papaya 2`), và **Mendeley/Kaggle Papaya** (`dudu/images`).
   - Bộ ảnh **Papaya** (`sbarbonjr/Papaya` trên GitHub / `sylviobarbon/papaya` trên Kaggle).
2. **Tìm kiếm ảnh Web (Bing Images - photo filter)**:
   - Truy vấn đa dạng bằng cả tiếng Việt và tiếng Anh (`quả bưởi`, `bưởi da xanh`, `bưởi diễn`, `pomelo fruit`, `quả hồng giòn`, `quả hồng đà lạt`, `persimmon fruit`, `quả đu đủ`, `đu đủ chín`, `đu đủ xanh`, `carica papaya fruit`, ...), lọc sẵn tên miền ảnh có watermark thương mại.
3. **iNaturalist (Research-grade fruiting observations)**:
   - Ảnh thực tế ngoài tự nhiên được xác minh loài (*Citrus maxima*, *Diospyros kaki*, *Carica papaya*).

### Phân bố nguồn dữ liệu sạch cuối cùng

| Lớp | Kaggle / Fruit-262 / Fruits-360 / Mendeley | Bing Images | iNaturalist | GitHub `sbarbonjr/Papaya` | Tổng |
|---|---:|---:|---:|---:|---:|
| `Buoi` | 647 | 811 | 29 | 0 | **1,487** |
| `Hong` | 721 | 744 | 213 | 0 | **1,678** |
| `dudu` | 879 | 632 | 302 | 32 | **1,845** |

## 3. Quy trình làm sạch và chia tập (T03)

1. **Lọc chất lượng cơ bản**: kiểm tra giải mã ảnh hoàn chỉnh, chuẩn hóa xoay EXIF, chuyển không gian màu RGB; loại ảnh quá nhỏ (`min(w, h) < 96px`), tỉ lệ khung hình quá lệch (`aspect ratio > 2.5`), ảnh đơn sắc/ít biến thiên hoặc tài liệu văn bản đen trắng, và loại bỏ nguồn nhiễu nặng (`openverse`, `wikimedia`).
2. **Khử trùng lặp chính xác & gần giống (Anti-Leakage)**:
   - Khử trùng lặp chính xác theo **SHA256** (cả byte gốc và pixel RGB).
   - Khử trùng lặp cảm nhận theo **64-bit dHash** (khoảng cách Hamming $\le 6$ với ảnh tự nhiên, $\le 10$ với chuỗi khung hình xoay Fruits-360) cả trong cùng lớp lẫn xuyên lớp, đảm bảo không rò rỉ ảnh trùng giữa `train`, `val` và `test`.
3. **Lọc ảnh sai lớp & nhiễu bằng CLIP (`ViT-B-32`) + kiểm tra trực quan**:
   - Đánh giá trên toàn bộ 5,825 ảnh duy nhất và loại bỏ 815 ảnh ngoài chủ đề/sai lớp (hoa hồng, khối gỗ, poster quảng cáo/infographic nhiều chữ, chân dung/hội nghị, món ăn chế biến/thịt/gỏi, cảnh cây từ xa không rõ quả, côn trùng, quả *Asimina triloba* pawpaw lẫn trong đu đủ, chôm chôm/dưa hấu/lựu lẫn trong bưởi), giữ lại đúng **5,010** ảnh đã qua xác minh CLIP.
4. **Đặt tên thống nhất, Resize và Chia tập**:
   - Resize giữ nguyên tỉ lệ khung hình với cạnh dài nhất bằng **`256px`** (`LANCZOS`, JPEG quality 90).
   - Đặt tên thống nhất: `<lop>_<split>_<stt>_<sha256[:8]>.jpg` (ví dụ `buoi_train_0001_a1b2c3d4.jpg`).
   - Chia phân tầng theo từng lớp (`seed = 42`): **70% `train` / 15% `val` / 15% `test`**, lưu đầy đủ truy vết trong [`split_manifest.csv`](split_manifest.csv).

## 4. Cấu trúc thư mục & Cách dùng chung

```text
fruit_classfication/data/
├── README.md
├── class_to_idx.json
├── split_manifest.csv
├── train/
│   ├── Buoi/
│   ├── Hong/
│   └── dudu/
├── val/
│   ├── Buoi/
│   ├── Hong/
│   └── dudu/
└── test/
    ├── Buoi/
    ├── Hong/
    └── dudu/
```

- **CNN (PyTorch / Keras)**: dùng trực tiếp [`../src/preprocess.py`](../src/preprocess.py) (`build_dataloaders("fruit_classfication/data", model_name="mobilenet_v2")` hoặc `build_keras_datasets(...)`).
- **YOLOv5-cls**: trỏ `--data fruit_classfication/data` (lưu ý khi huấn luyện có thể copy tạm `train/` và `val/` ra `/content/fruit_trainval` để đảm bảo tách biệt hoàn toàn với `test/`, sau đó đánh giá cuối bằng `classify/val.py --data fruit_classfication/data`).
