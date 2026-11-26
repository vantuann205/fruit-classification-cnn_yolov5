# YOLOv5-cls Experiment Report

## 1. Dataset
- Source: Fruit.zip (classes: Buoi, Hong, dudu)
- Input size: 224x224

## 2. Dataset Split
- Train: 70%
- Val: 15%
- Test: 15%
- Random Seed: 42
- Manifest: split_manifest.csv

## 3. Class Mapping
- Mapping: {"Buoi": 0, "Hong": 1, "dudu": 2}

## 4. Model Architecture
- Backbone: YOLOv5s-cls (yolov5s-cls.pt)
- Commit: 33ae15ef99bd04ce1d3f12155b8a85f355d73351

## 5. Environment
- GPU: Tesla T4
- Device: 0
- Workers: 2

## 6. Training Configuration
- Batch size: 32
- Image size: 224
- Optimizer: Adam

## 7. Experiments
| Experiment | Epochs | Learning Rate | Batch Size | Wall Time (s) | Val Top-1 Acc |
|---|---|---|---|---|---|
| baseline_lr001 | 30 | 0.001 | 32 | 302.82 | 0.9879 |
| exp02_lr0005 | 30 | 0.0005 | 32 | 294.42 | 0.9931 |

## 8. Training Time
- **baseline_lr001**: 302.82s (2026-11-26T17:06:03.732788+00:00 to 2026-11-26T17:11:06.557784+00:00)
- **exp02_lr0005**: 294.42s (2026-11-26T17:11:06.568992+00:00 to 2026-11-26T17:16:00.986145+00:00)

## 9. Validation Results
- Best Experiment: exp02_lr0005
- Best Val Top-1 Accuracy: 0.9931

## 10. Test Results
- Checkpoint: yolov5_cls_best.pt
- Evaluation Log:
```
[34m[1mclassify/val: [0mdata=/content/fruit-classification-cnn_yolov5/fruit_classfication/data, weights=['/content/fruit-classification-cnn_yolov5/fruit_classfication/models/yolov5_cls_best.pt'], batch_size=32, imgsz=224, device=0, workers=8, verbose=True, project=/content/yolov5/runs/val-cls, name=final_test, exist_ok=True, half=False, dnn=False
YOLOv5 🚀 v7.0-575-g33ae15ef Python-3.13.16 torch-2.14.1+cu130 CUDA:0 (Tesla T4, 14913MiB)

Fusing layers... 
Model summary: 67 layers, 4,170,531 parameters, 0 gradients, 10.4 GFLOPs

[Ktesting: 5% ╸─────────── 1/19 1.8s/it 0.5s<31.9s
[Ktesting: 31% ━━━╸──────── 6/19 12.1it/s 0.7s<1.1s
[Ktesting: 42% ━━━━━─────── 8/19 14.1it/s 0.8s<0.8s
[Ktesting: 52% ━━━━━━────── 10/19 15.5it/s 0.9s<0.6s
[Ktesting: 63% ━━━━━━━╸──── 12/19 16.7it/s 1.0s<0.4s
[Ktesting: 73% ━━━━━━━━╸─── 14/19 17.7it/s 1.1s<0.3s
[Ktesting: 89% ━━━━━━━━━━╸─ 17/19 18.2it/s 1.2s<0.1s
[Ktesting: 100% ━━━━━━━━━━━━ 19/19 13.9it/s 1.4s
                   Class      Images    top1_acc    top5_acc
                     all         581       0.981           1
                    Buoi         176       0.955           1
                    Hong         165       0.988           1
                    dudu         240       0.996           1
Speed: 0.1ms pre-process, 1.1ms inference, 0.1ms post-process per image at shape (1, 3, 224, 224)
Results saved to [1m/content/yolov5/runs/val-cls/final_test[0m
```

## 11. Selected Checkpoint
- Path: `fruit_classfication/models/yolov5_cls_best.pt`
- Size: 8461022 bytes
- Selected from: exp02_lr0005

## 12. Reproducibility
- Seed: 42
- SHA256 manifest: split_manifest.csv
- Staged train/val without test leakage