# Thư mục dự án

Thư mục này chứa các thành phần của dự án phân loại trái cây. Giữ đúng cấu trúc dưới đây để notebook, ứng dụng demo và tài liệu dễ tìm, dễ chia sẻ.

```text
fruit_classfication/
├── data/        File dataset, có thể đóng gói dạng .zip
├── notebooks/   Notebook huấn luyện CNN, YOLOv5 và đánh giá
├── models/      File trọng số hoặc mô hình đã huấn luyện
├── demo/        Mã nguồn và tài nguyên chạy ứng dụng demo
└── report/      Báo cáo, slide và tài liệu liên quan
```

## Nội dung từng thư mục

- `data/`: lưu bộ dữ liệu và thông tin nguồn dữ liệu. Nếu dataset quá lớn hoặc có giới hạn phân phối, hãy cung cấp liên kết tải và hướng dẫn giải nén thay vì commit trực tiếp file.
- `notebooks/`: lưu notebook theo từng bước, chẳng hạn tiền xử lý dữ liệu, huấn luyện CNN, huấn luyện YOLOv5 và đánh giá kết quả.
- `models/`: lưu checkpoint/trọng số đã huấn luyện cùng thông tin về kiến trúc, phiên bản và cách nạp model. Các file model lớn có thể dùng Git LFS hoặc liên kết tải.
- `demo/`: lưu mã nguồn, cấu hình và hướng dẫn chạy ứng dụng minh họa dự đoán.
- `report/`: lưu báo cáo, slide thuyết trình và các tài liệu tổng kết dự án.

Các thư mục hiện được tạo sẵn; hãy thêm dataset, notebook, model, demo và tài liệu tương ứng vào đúng vị trí.
