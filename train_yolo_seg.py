from ultralytics import YOLO


model = YOLO("yolov8n-seg.pt")


model.train(
    data="yolo_seg/data.yaml",
    epochs=100,
    imgsz=448,
    batch=32,
    device=0,
    workers=0,
    project="runs/segmentation",
    name="yolov8n_seg",
    plots=True

)