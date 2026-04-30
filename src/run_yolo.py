from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.predict(
    source="data/raw/small_clip_720.mp4",
    save=True,
    project="outputs",
    name="yolo_test",
    conf=0.5,
    imgsz=640,
    device="cpu",
    show_labels=False,
    show_conf=False
)