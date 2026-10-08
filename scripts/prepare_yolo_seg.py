from pathlib import Path
import cv2
import shutil


SOURCE = Path("segmentation")
OUTPUT = Path("yolo_seg")


for split in ["train", "val", "test"]:

    images_dir = SOURCE / split / "images"
    masks_dir = SOURCE / split / "masks"

    output_images = OUTPUT / "images" / split
    output_labels = OUTPUT / "labels" / split

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    images = list(images_dir.glob("*.jpg"))

    print()
    print(split, ":", len(images))

    for image_path in images:



        number = image_path.stem.replace(
            "image_",
            ""
        )

        mask_path = masks_dir / f"mask_{number}.png"

        if not mask_path.exists():

            print(
                "not found",
                mask_path
            )

            continue

        # копируем
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        # загружаем маску
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        # маска в бинар
        _, mask = cv2.threshold(
            mask,
            127,
            255,
            cv2.THRESH_BINARY
        )

        # области трещин
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        label_path = (
            output_labels
            / f"{image_path.stem}.txt"
        )

        with open(label_path, "w") as file:

            for contour in contours:

                area = cv2.contourArea(contour)

                if area < 2:
                    continue

                
                epsilon = (
                    0.002
                    * cv2.arcLength(
                        contour,
                        True
                    )
                )

                polygon = cv2.approxPolyDP(
                    contour,
                    epsilon,
                    True
                )

                if len(polygon) < 3:
                    continue

                height, width = mask.shape

                points = []

                for point in polygon:

                    x, y = point[0]

                    
                    x = x / width
                    y = y / height

                    points.append(x)
                    points.append(y)

                # 0 = crack
                line = "0 " + " ".join(
                    f"{point:.6f}"
                    for point in points
                )

                file.write(line + "\n")


