# ⚽ Measuring Game Intensity with Computer Vision (Exploratory)

## Overview

This project explores whether it is possible to estimate **game intensity** in football using computer vision techniques applied to broadcast video.

The idea was to move beyond traditional event-based metrics and try to capture intensity directly from **player movement patterns**.

---

## Approach

The pipeline was built using:

* YOLOv8 for player detection
* ByteTrack for tracking
* Frame-by-frame extraction of player positions
* Derivation of movement-based signals:

  * displacement between frames
  * aggregated movement per frame
  * structural variation (relative distances between players)

The goal was to construct a **proxy for intensity over time** and visualize it as a time series.

---

## Key Challenges

Working with broadcast footage introduces significant constraints:

* Camera movement (pan, zoom) affects perceived motion
* Replays and cuts introduce artificial spikes
* Perspective distortion prevents direct spatial interpretation
* Tracking instability (ID switches) breaks temporal consistency

Even after applying filtering and simple corrections, these factors strongly impact the signal.

---

## Findings

* Raw movement signals capture **visual motion**, not necessarily **game intensity**
* Camera behavior can dominate the signal, especially during replays or transitions
* Relative measures (e.g., distances between players) improve robustness but remain noisy
* Without pitch calibration or camera stabilization, interpretation is limited

---

## Conclusion

This experiment highlights a key insight:

> Measuring intensity from broadcast video is not only a modeling problem — it is fundamentally a **data problem**.

Reliable estimation would likely require:

* Camera calibration (homography)
* Stabilization or optical flow correction
* Access to tracking data or multi-camera setups

---

## Why this matters

In football analysis, "intensity" is often discussed but rarely well-defined.

This project reinforces that:

* The way we measure intensity directly shapes how we understand the game
* Different proxies (events, movement, tracking) capture different aspects of it

---

## Status

Exploratory — not intended as a production-ready solution.

---

## Next Steps (if extended)

* Camera motion compensation
* Mapping to real-world coordinates
* Combining CV signals with event data

---

## Tech Stack

* Python
* Ultralytics YOLOv8
* NumPy / Matplotlib

---

## Author

Fernanda Pieri
