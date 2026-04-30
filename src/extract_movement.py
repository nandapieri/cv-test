from ultralytics import YOLO
import numpy as np
import math
import matplotlib.pyplot as plt

model = YOLO("yolov8s.pt")

results = model.track(
    source="data/raw/small_clip_720.mp4",
    stream=True,   # ESSENCIAL agora
    conf=0.45,
    imgsz=768,
    device="cpu",
    tracker="bytetrack.yaml"
)

# armazenar posições
tracks = {}

frame_id = 0

for r in results:
    boxes = r.boxes

    if boxes is None:
        continue

    for box in boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])

        # pega centro da bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        if track_id not in tracks:
            tracks[track_id] = []

        tracks[track_id].append((frame_id, cx, cy))

    frame_id += 1

print("Número de tracks:", len(tracks))


movements = []

for track_id, points in tracks.items():
    if len(points) < 2:
        continue

    for i in range(1, len(points)):
        _, x1, y1 = points[i-1]
        _, x2, y2 = points[i]

        dx = x2 - x1
        dy = y2 - y1

        dist = math.sqrt(dx**2 + dy**2)

        movements.append(dist)

print("Número de movimentos:", len(movements))
print("Movimento médio:", sum(movements)/len(movements))

from collections import defaultdict

frame_movements = defaultdict(list)

for track_id, points in tracks.items():
    if len(points) < 2:
        continue

    for i in range(1, len(points)):
        frame_prev, x1, y1 = points[i-1]
        frame_curr, x2, y2 = points[i]

        dx = x2 - x1
        dy = y2 - y1

        dist = (dx**2 + dy**2) ** 0.5

        frame_movements[frame_curr].append(dist)

# agora agrega por frame
frame_intensity = {}

import itertools

for frame, dists in frame_movements.items():
    positions = []

    for track_id, points in tracks.items():
        for f, x, y in points:
            if f == frame:
                positions.append((x, y))

    if len(positions) < 2:
        continue

    pairwise = []
    for (x1, y1), (x2, y2) in itertools.combinations(positions, 2):
        dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        pairwise.append(dist)

    frame_intensity[frame] = np.std(pairwise)

filtered_frame_intensity = {
    frame: value
    for frame, value in frame_intensity.items()
    if value < 100
}

print(list(frame_intensity.items())[:10])

frames = sorted(frame_intensity.keys())
values = [frame_intensity[f] for f in frames]
filtered_values = [v if v < 100 else np.nan for v in values]

# suavização
window = 10
smoothed = np.convolve(
    np.nan_to_num(filtered_values), 
    np.ones(window)/window, 
    mode='same'
)

plt.figure(figsize=(12, 4))

plt.plot(frames, filtered_values, alpha=0.3, label="Filtrado")

# linha suavizada (importante)
plt.plot(frames, smoothed, label="Suavizado")

plt.title("Intensidade do jogo ao longo do tempo")
plt.xlabel("Frame")
plt.ylabel("Movimento médio (proxy)")

plt.legend()
plt.grid(True)
plt.show()

top_peaks = sorted(filtered_frame_intensity.items(), key=lambda x: x[1], reverse=True)[:5]


print("Top 5 picos:")
for frame, value in top_peaks:
    print(f"Frame {frame} → {value}")
    time_sec = frame / 30
    print(time_sec)