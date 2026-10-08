from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


MODEL_PATH = (
    "runs/segment/runs/segmentation/"
    "yolov8n_seg-2/weights/best.pt"
)

IMAGE_DIR = Path("yolo_seg/images/test")
MASK_DIR = Path("segmentation/test/masks")

OUTPUT_DIR = Path("evaluation/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


model = YOLO(MODEL_PATH)


count = 0


for image_path in sorted(IMAGE_DIR.glob("*.jpg")):

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


    # только там где трещина

    if gt.sum() == 0:
        continue


    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


  
    gt_overlay = image.copy()
    gt_overlay[gt] = [255, 0, 0]

    gt_overlay = cv2.addWeighted(
        image,
        0.65,
        gt_overlay,
        0.35,
        0
    )



    pred_overlay = image.copy()
    pred_overlay[pred] = [0, 255, 0]

    pred_overlay = cv2.addWeighted(
        image,
        0.65,
        pred_overlay,
        0.35,
        0
    )



    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12, 3)
    )


    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")


    axes[1].imshow(gt)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")


    axes[2].imshow(pred)
    axes[2].set_title("Prediction")
    axes[2].axis("off")


    axes[3].imshow(pred_overlay)
    axes[3].set_title("Prediction Overlay")
    axes[3].axis("off")


    plt.tight_layout()


    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_comparison.png"
    )


    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    count += 1


    if count >= 10:
        break


print("images:", count)
print("saved to:", OUTPUT_DIR)