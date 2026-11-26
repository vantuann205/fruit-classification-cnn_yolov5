# YOLOv5-cls Experiment Report

## 1. Dataset
- Dataset source: `fruit_classfication/data/Fruit.zip`
- Classes: `Buoi`, `Hong`, `dudu`
- Input size: 224x224 (preprocessed max long edge 256, aspect ratio preserved, LANCZOS)

## 2. Dataset Split
- Train: 70%
- Val: 15%
- Test: 15%
- Random Seed: 42
- Manifest: `fruit_classfication/data/split_manifest.csv`

## 3. Class Mapping
- Mapping (alphabetical):
  - `Buoi`: 0
  - `Hong`: 1
  - `dudu`: 2
- Saved at: `fruit_classfication/data/class_to_idx.json`

## 4. Model Architecture
- Backbone: YOLOv5s-cls (`yolov5s-cls.pt`)
- Official repository: `https://github.com/ultralytics/yolov5` (commit: `33ae15ef99bd04ce1d3f12155b8a85f355d73351`)

## 5. Environment
- GPU: Colab GPU (e.g. Tesla T4) - *to be recorded upon Colab execution*
- Python: 3.10+
- PyTorch / Torchvision / CUDA: Official YOLOv5 dependencies

## 6. Training Configuration
- Model: `yolov5s-cls.pt`
- Pretrained: True
- Image size: 224
- Batch size: 32
- Epochs: 30
- Optimizer: Adam
- Seed: 42
- Workers: 2
- Device: 0

## 7. Experiments
| Experiment | Epochs | Learning Rate | Batch Size | Wall Time (s) | Val Top-1 Acc |
|---|---|---|---|---|---|
| Smoke Test | 1 | 0.001 | 32 | *Pending Colab* | *Pending Colab* |
| EXP-01 (Baseline) | 30 | 0.001 | 32 | *Pending Colab* | *Pending Colab* |
| EXP-02 (Tuning) | 30 | 0.0005 | 32 | *Pending Colab* | *Pending Colab* |

## 8. Training Time
- Training time measurement pipeline implemented using `time.perf_counter()` and UTC timestamps.
- Measured separately for each experiment during Colab execution.

## 9. Validation Results
- Best experiment selection rule: Highest Top-1 Accuracy on validation set (`TRAINVAL_DIR/val`).
- *Pending execution on Colab.*

## 10. Test Results
- Evaluation target: Common test set (`DATA_DIR/test`).
- Metric: Top-1 Accuracy via `classify/val.py`.
- *Pending execution on Colab.*

## 11. Selected Checkpoint
- Target path: `fruit_classfication/models/yolov5_cls_best.pt`
- Rule: Best checkpoint from validation experiments copied to target path.

## 12. Reproducibility
- Seed: 42.
- Deterministic file naming via SHA256 prefix.
- No test leakage: `TRAINVAL_DIR` staging directory contains only `train` and `val` during training; evaluation runs separately on `DATA_DIR/test`.
