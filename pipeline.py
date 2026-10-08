from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


CLASSIFIER_PATH = "runs/classify/train-3/weights/best.pt"
SEGMENTATION_PATH = "runs/segment/runs/segmentation/yolov8n_seg-2/weights/best.pt"


# порог уверенности
CLASS_THRESHOLD = 0.5
SEG_THRESHOLD = 0.25


classifier = YOLO(CLASSIFIER_PATH)
segmenter = YOLO(SEGMENTATION_PATH)


def run_pipeline(image_path: str):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"img not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("error read")



# классификация

    cls_result = classifier.predict(
        source=image,
        imgsz=320,
        verbose=False
    )[0]

    probs = cls_result.probs

    class_id = int(probs.top1)
    confidence = float(probs.top1conf)

    class_name = classifier.names[class_id]


    print("class:", class_name)
    print("confidence:", round(confidence, 4))


    if class_name != "cracked" or confidence < CLASS_THRESHOLD:

        print()
        print("no crack")
        return {
            "crack": False,
            "classification_confidence": confidence,
            "mask": None
        }

    print()
    print("crack detected - segmentation")

   
   # сегментация

    seg_result = segmenter.predict(
        source=image,
        imgsz=224,
        conf=SEG_THRESHOLD,
        verbose=False
    )[0]

    if seg_result.masks is None:

        print("no crack")

        return {
            "crack": True,
            "classification_confidence": confidence,
            "mask": None
        }

    masks = seg_result.masks.data.cpu().numpy()

    # объединяем все найденные маски
    combined_mask = np.max(masks, axis=0)

    combined_mask = (
        combined_mask * 255
    ).astype(np.uint8)

    # возвращаем размер маски к размеру исходного изображения
    combined_mask = cv2.resize(
        combined_mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )


# визуализация

    overlay = image.copy()

    crack_pixels = combined_mask > 127

    overlay[crack_pixels] = (
        overlay[crack_pixels] * 0.5
        + np.array([0, 0, 255]) * 0.5
    )

    overlay = overlay.astype(np.uint8)

    cv2.imshow("original", image)
    cv2.imshow("crack mask", combined_mask)
    cv2.imshow("seg", overlay)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        "crack": True,
        "classification_confidence": confidence,
        "mask": combined_mask
    }


if __name__ == "__main__":
    run_pipeline("test.jpg")