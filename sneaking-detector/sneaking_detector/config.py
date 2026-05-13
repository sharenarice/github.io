# Detection thresholds
APPROACH_RATE_THRESHOLD = 0.06   # normalized bbox area growth fraction over window
APPROACH_FRAMES_WINDOW = 8       # frame window to measure area growth
MIN_FRAMES_TO_CONFIRM = 10       # tracked frames required before any alert

YAW_THRESHOLD = 35.0             # max head yaw (°) to count as facing camera
PITCH_THRESHOLD = 30.0           # max head pitch (°)
PERSON_CONF = 0.45               # YOLO person detection confidence
FACE_CONF = 0.5                  # MediaPipe face detection confidence

# A "walk-by" is someone whose lateral centroid velocity dominates their approach;
# alert only when approach rate meaningfully exceeds lateral drift.
LATERAL_DOMINANCE_RATIO = 0.6   # lateral_speed / total_speed threshold for walk-by

# Notification
ALERT_COOLDOWN_SECONDS = 5
CONFIDENCE_THRESHOLD = 0.40      # minimum combined confidence to fire notification

# Camera
CAMERA_INDEX = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# Preview display (fits most screens without being tiny)
PREVIEW_WIDTH = 854
PREVIEW_HEIGHT = 480

# Overlay colors (BGR for OpenCV)
COLOR_SAFE = (60, 210, 60)
COLOR_ATTENTION = (40, 160, 255)
COLOR_THREAT = (30, 30, 220)
COLOR_LABEL_BG = (20, 20, 20)
