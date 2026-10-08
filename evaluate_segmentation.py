from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = (
    "runs/segment/runs/segmentation/"
    "yolov8n_seg-2/weights/best.pt"
)

IMAGE_DIR = Path("yolo_seg/images/test")
MASK_DIR = Path("segmentation/test/masks")


model = YOLO(MODEL_PATH)


ious = []
dices = []
precisions = []
recalls = []

true_negative = 0
false_positive_images = 0


for image_path in IMAGE_DIR.glob("*.jpg"):

    number = image_path.stem.replace("image_", "")

    mask_path = MASK_DIR / f"mask_{number}.png"

    if not mask_path.exists():
        continue


    # gt
    gt = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    gt = gt > 127


    # prediction
    results = model.predict(
        source=str(image_path),
        imgsz=224,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    pred = np.zeros_like(gt, dtype=bool)


    if result.masks is not None:

        masks = result.masks.data.cpu().numpy()

        for mask in masks:

            mask = cv2.resize(
                mask,
                (gt.shape[1], gt.shape[0])
            )

            pred |= mask > 0.5


    gt_pixels = gt.sum()
    pred_pixels = pred.sum()

    if gt_pixels == 0:

        if pred_pixels == 0:

            true_negative += 1

        else:

            false_positive_images += 1

        continue

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()


    union = np.logical_or(pred, gt).sum()

    if union > 0:
        iou = tp / union
    else:
        iou = 0.0

    denominator = 2 * tp + fp + fn

    if denominator > 0:
        dice = 2 * tp / denominator
    else:
        dice = 0.0


    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0


    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0


    ious.append(iou)
    dices.append(dice)
    precisions.append(precision)
    recalls.append(recall)


print("img:", len(list(IMAGE_DIR.glob("*.jpg"))))
print("crack", len(ious))
print("no crack:", true_negative + false_positive_images)
print("fp images:", false_positive_images)
print("IoU:", f"{np.mean(ious):.4f}")
print("Dice:", f"{np.mean(dices):.4f}")
print("Precision:", f"{np.mean(precisions):.4f}")
print("Recall:", f"{np.mean(recalls):.4f}")