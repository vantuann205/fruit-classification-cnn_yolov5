# Thư mục dự án

Thư mục này chứa các thành phần của dự án phân loại trái cây. Giữ đúng cấu trúc dưới đây để notebook, ứng dụng demo và tài liệu dễ tìm, dễ chia sẻ.

```text
fruit_classfication/
├── data/        Dataset đã làm sạch (train/val/test/<lớp>) và data/README.md
├── src/         Mã nguồn tiền xử lý và augmentation (preprocess.py, visualize.py)
├── notebooks/   Notebook huấn luyện CNN, YOLOv5 và đánh giá
├── models/      File trọng số hoặc mô hình đã huấn luyện
├── demo/        Mã nguồn và tài nguyên chạy ứng dụng demo
└── reports/     Báo cáo thực nghiệm, biểu đồ (figures/) và slide
```

## Nội dung từng thư mục

- `data/`: lưu bộ dữ liệu đã làm sạch và chia sẵn `train/val/test` theo tỉ lệ 70/15/15 (`Buoi`, `Hong`, `dudu`) cùng `README.md`, `split_manifest.csv` và `class_to_idx.json`. Ảnh thô giữ ở máy cục bộ (`raw_data/`, không đẩy lên GitHub).
- `src/`: lưu mã nguồn tiền xử lý và augmentation dùng chung (`preprocess.py`, `visualize.py`) hỗ trợ MobileNetV2/ResNet50 (PyTorch & Keras) và YOLOv5-cls.
- `notebooks/`: lưu notebook theo từng bước, chẳng hạn huấn luyện CNN, huấn luyện YOLOv5 và đánh giá kết quả.
- `models/`: lưu checkpoint/trọng số đã huấn luyện cùng thông tin về kiến trúc, phiên bản và cách nạp model. Các file model lớn có thể dùng Git LFS hoặc liên kết tải.
- `demo/`: lưu mã nguồn, cấu hình và hướng dẫn chạy ứng dụng minh họa dự đoán.
- `reports/`: lưu báo cáo thực nghiệm, ảnh minh họa (`reports/figures/`), slide thuyết trình và các tài liệu tổng kết dự án.

Các thư mục hiện được tạo sẵn; hãy thêm dataset, notebook, model, demo và tài liệu tương ứng vào đúng vị trí.
