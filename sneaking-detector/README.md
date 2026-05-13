# Sneaking Detector

A real-time desktop computer-vision app that **alerts you when someone is sneaking up on you** — but stays silent when people are simply walking by without paying attention to you.

---

## How it works

The app runs two detection passes on every camera frame:

| Signal | Method | Meaning |
|--------|--------|---------|
| **Approaching** | Bounding-box area growth rate over a rolling window | Person is getting closer |
| **Watching you** | Head-pose estimation (yaw / pitch via MediaPipe FaceMesh + solvePnP) | Person's face is turned toward the camera |
| **Walk-by filter** | Lateral centroid displacement ratio | Suppresses alerts when motion is primarily sideways |

An alert fires only when **all three** conditions are satisfied simultaneously for several consecutive frames:

```
THREAT = is_approaching AND facing_camera AND NOT is_walk_by
```

Someone walking past in the background with their head turned away → **no alert**.  
Someone who locks eyes on you and starts moving toward you → **alert**.

---

## Requirements

- Python 3.9+
- Webcam or USB camera

### Python packages

| Package | Purpose |
|---------|---------|
| `opencv-python` | Camera capture, HOG person detector, drawing |
| `mediapipe` | Face mesh + head-pose estimation |
| `numpy` | Array maths |
| `Pillow` | tkinter-compatible image rendering |
| `plyer` *(optional)* | Cross-platform desktop notifications |
| `ultralytics` *(optional)* | YOLOv8-nano person detector — faster & more accurate than the built-in HOG fallback |

---

## Installation

### macOS / Linux

```bash
git clone <repo>
cd sneaking-detector
bash install.sh
source .venv/bin/activate
python run.py
```

### Windows

```
install.bat
.venv\Scripts\activate.bat
python run.py
```

### Manual install

```bash
pip install -r requirements.txt
# optional: pip install ultralytics   ← enables YOLOv8
python run.py
```

---

## Usage

1. Launch the app with `python run.py`.
2. Press **▶ Start** — the camera feed appears with colour-coded bounding boxes.
3. The **status panel** changes colour:
   - **Green / SAFE** — nobody approaching and watching
   - **Orange / WATCH** — someone nearby, monitoring continues
   - **Red / ALERT** — someone sneaking up; desktop notification fires

### Controls

| Control | Description |
|---------|-------------|
| **Sensitivity** slider | Lower = triggers sooner; higher = requires faster approach |
| **Desktop notifications** | Toggle OS-level notification on/off |
| **Show debug labels** | Display yaw angle, conf score, and walk-by flag on each box |
| **Camera index** | Change input device (0 = default webcam, 1, 2 … for USB cams) |

---

## Detection tuning

Edit `sneaking_detector/config.py` to adjust thresholds:

```python
APPROACH_RATE_THRESHOLD = 0.06   # bbox area growth needed to count as "approaching"
YAW_THRESHOLD            = 35.0  # head yaw (°) within which face counts as "watching"
PITCH_THRESHOLD          = 30.0  # head pitch (°) tolerance
LATERAL_DOMINANCE_RATIO  = 0.6   # walk-by suppression: lateral/total motion ratio
MIN_FRAMES_TO_CONFIRM    = 10    # frames of consistent signal before alerting
ALERT_COOLDOWN_SECONDS   = 5     # minimum gap between notification pops
```

---

## Architecture

```
run.py
└── SneakingDetectorApp (app.py)
    ├── Capture thread   → _frame_q
    ├── Process thread   ← _frame_q → _result_q
    │   └── VisionPipeline (vision.py)
    │       ├── Person detector  (YOLOv8-nano or HOG)
    │       ├── CentroidTracker  (tracker.py)
    │       └── Head-pose estimator  (MediaPipe FaceMesh + solvePnP)
    ├── UI poll loop  ← _result_q  (tkinter after())
    └── AlertNotifier (notifier.py)
        └── OS notification dispatch (plyer / osascript / notify-send / win10toast)
```

---

## Limitations

- Works best with a **front-facing camera** (webcam on monitor top) at eye level.
- Head-pose estimation degrades at extreme angles or in low light.
- The HOG fallback detector is less accurate than YOLO — install `ultralytics` for best results.
- Performance on CPU: ~15–25 FPS with YOLO-nano; ~8–12 FPS with HOG.
