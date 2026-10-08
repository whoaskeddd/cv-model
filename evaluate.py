from ultralytics import YOLO
import numpy as np
model = YOLO("runs/classify/train-3/weights/best.pt")

results = model.val(
    data="dataset",
    split="test",
    imgsz=320,
    device=0,
    plots=True,
    workers = 0
)

print(f"accuracy: {results.top1:.4f}")

cm = results.confusion_matrix.matrix
print("\nconf matrix:")
print(cm)

names = model.names
print("\nclasses:", names)

cracked = 0

# значения из матрицы
tp = cm[cracked, cracked]
fn = cm[cracked].sum() - tp
fp = cm[:, cracked].sum() - tp

# метрики
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print("\nmetrics:")
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))


# from ultralytics import YOLO

# model = YOLO("runs/classify/train-3/weights/best.pt")

# results = model.val(
#     data="dataset",
#     split="test",
#     imgsz=320,
#     device=0,
#     workers = 0
# )

# print(results)