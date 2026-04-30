from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.track(
    source="data/raw/small_clip_720.mp4",
    save=True,
    project="outputs",
    name="tracking_test",
    conf=0.5,
    imgsz=832,
    device="cpu",
    show_labels=True,
    show_conf=False,
    tracker="bytetrack.yaml",
    persist=True
)