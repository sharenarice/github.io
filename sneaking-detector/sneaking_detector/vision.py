"""
Vision pipeline: person detection → tracking → head-pose attention analysis.

Detection back-ends (tried in order):
  1. YOLOv8-nano  (ultralytics)  — fast and accurate
  2. OpenCV HOG   (built-in)     — no extra install required

Attention is judged by MediaPipe FaceMesh head-pose estimation:
  * yaw  near 0° → face turned toward camera
  * pitch near 0° → face level (not looking up/down sharply)

A person counts as "sneaking up" only when BOTH conditions hold:
  is_approaching AND facing_camera AND NOT is_walk_by
"""

from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from .config import (
    PERSON_CONF,
    FACE_CONF,
    YAW_THRESHOLD,
    PITCH_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,
)
from .tracker import CentroidTracker, TrackedPerson


# ---------------------------------------------------------------------------
# 3-D face model points for solvePnP  (standard 6-point subset)
# ---------------------------------------------------------------------------
_FACE_3D = np.array(
    [
        [0.0, 0.0, 0.0],           # nose tip   (lm 4)
        [0.0, -330.0, -65.0],      # chin       (lm 152)
        [-225.0, 170.0, -135.0],   # left eye   (lm 33)
        [225.0, 170.0, -135.0],    # right eye  (lm 263)
        [-150.0, -150.0, -125.0],  # left mouth (lm 61)
        [150.0, -150.0, -125.0],   # right mouth(lm 291)
    ],
    dtype=np.float64,
)
_LM_INDICES = [4, 152, 33, 263, 61, 291]


@dataclass
class PersonAssessment:
    track_id: int
    bbox: Tuple[int, int, int, int]
    # raw signals
    is_approaching: bool
    is_walk_by: bool
    face_visible: bool
    facing_camera: bool
    yaw: float
    pitch: float
    area_growth_rate: float
    # derived
    is_threat: bool
    threat_confidence: float  # 0–1


class VisionPipeline:
    def __init__(self, frame_w: int = CAPTURE_WIDTH, frame_h: int = CAPTURE_HEIGHT):
        self.frame_w = frame_w
        self.frame_h = frame_h

        self._init_person_detector()
        self._init_face_mesh()
        self.tracker = CentroidTracker(frame_w=frame_w, frame_h=frame_h)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_person_detector(self) -> None:
        self._use_yolo = False
        try:
            from ultralytics import YOLO  # type: ignore
            self._yolo = YOLO("yolov8n.pt")
            self._use_yolo = True
        except Exception:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _init_face_mesh(self) -> None:
        self._mp_fm = mp.solutions.face_mesh
        self._face_mesh = self._mp_fm.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=FACE_CONF,
            min_tracking_confidence=0.5,
            static_image_mode=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> List[PersonAssessment]:
        """Run the full pipeline on one BGR frame."""
        h, w = frame.shape[:2]
        if (w, h) != (self.frame_w, self.frame_h):
            self.frame_w, self.frame_h = w, h
            self.tracker = CentroidTracker(frame_w=w, frame_h=h)

        person_boxes = self._detect_persons(frame)
        tracked: Dict[int, TrackedPerson] = self.tracker.update(person_boxes)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = self._face_mesh.process(rgb)

        assessments: List[PersonAssessment] = []
        for tid, person in tracked.items():
            face_lm = self._match_face_to_person(
                face_result, person.bbox, (h, w)
            )
            face_visible = face_lm is not None
            facing = False
            yaw = pitch = 0.0

            if face_visible:
                yaw, pitch, _ = self._head_pose(frame, face_lm)
                facing = abs(yaw) < YAW_THRESHOLD and abs(pitch) < PITCH_THRESHOLD

            approaching = person.is_approaching
            walk_by = person.is_walk_by

            # Core logic: threat = approaching + watching + not just walking past
            is_threat = approaching and facing and not walk_by

            confidence = 0.0
            if is_threat:
                approach_conf = min(
                    1.0, person.area_growth_rate / (0.30 + 1e-6)
                )
                attention_conf = 1.0 - (abs(yaw) / YAW_THRESHOLD)
                walk_penalty = max(0.0, 1.0 - person.lateral_ratio * 2)
                confidence = (approach_conf + attention_conf) / 2 * walk_penalty

            assessments.append(
                PersonAssessment(
                    track_id=tid,
                    bbox=person.bbox,
                    is_approaching=approaching,
                    is_walk_by=walk_by,
                    face_visible=face_visible,
                    facing_camera=facing,
                    yaw=yaw,
                    pitch=pitch,
                    area_growth_rate=person.area_growth_rate,
                    is_threat=is_threat and confidence >= CONFIDENCE_THRESHOLD,
                    threat_confidence=round(confidence, 3),
                )
            )
        return assessments

    # ------------------------------------------------------------------
    # Person detection
    # ------------------------------------------------------------------
    def _detect_persons(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        if self._use_yolo:
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def _detect_yolo(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        results = self._yolo(
            frame, classes=[0], conf=PERSON_CONF, verbose=False
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _detect_hog(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, _ = self._hog.detectMultiScale(
            gray, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        if len(rects) == 0:
            return []
        return [(x, y, x + w, y + h) for x, y, w, h in rects]

    # ------------------------------------------------------------------
    # Face-to-person matching
    # ------------------------------------------------------------------
    def _match_face_to_person(
        self,
        face_result,
        bbox: Tuple[int, int, int, int],
        shape: Tuple[int, int],
    ) -> Optional[object]:
        if not face_result.multi_face_landmarks:
            return None
        h, w = shape
        x1, y1, x2, y2 = bbox
        # Expand bbox slightly upward (heads often above shoulder bbox)
        y1_exp = max(0, y1 - int((y2 - y1) * 0.15))

        best, best_score = None, -1.0
        for lm in face_result.multi_face_landmarks:
            # Use nose tip as face representative point
            nx = lm.landmark[4].x * w
            ny = lm.landmark[4].y * h
            if x1 <= nx <= x2 and y1_exp <= ny <= y2:
                # Score by how centered the face is in the bbox
                cx = (x1 + x2) / 2
                score = 1.0 - abs(nx - cx) / ((x2 - x1) / 2 + 1e-6)
                if score > best_score:
                    best_score = score
                    best = lm
        return best

    # ------------------------------------------------------------------
    # Head-pose estimation (solvePnP)
    # ------------------------------------------------------------------
    def _head_pose(
        self, frame: np.ndarray, face_landmarks
    ) -> Tuple[float, float, float]:
        h, w = frame.shape[:2]
        face_2d = np.array(
            [
                [face_landmarks.landmark[i].x * w,
                 face_landmarks.landmark[i].y * h]
                for i in _LM_INDICES
            ],
            dtype=np.float64,
        )

        fl = float(w)
        cam_matrix = np.array(
            [[fl, 0, w / 2], [0, fl, h / 2], [0, 0, 1]], dtype=np.float64
        )
        dist = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, _ = cv2.solvePnP(
            _FACE_3D, face_2d, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        yaw   = angles[1] * 360.0
        pitch = angles[0] * 360.0
        roll  = angles[2] * 360.0
        return yaw, pitch, roll

    # ------------------------------------------------------------------
    # Overlay drawing helpers
    # ------------------------------------------------------------------
    def draw_overlays(
        self, frame: np.ndarray, assessments: List[PersonAssessment]
    ) -> np.ndarray:
        from .config import COLOR_SAFE, COLOR_ATTENTION, COLOR_THREAT, COLOR_LABEL_BG

        out = frame.copy()
        for a in assessments:
            if a.is_threat:
                color = COLOR_THREAT
                label = f"ALERT  conf:{a.threat_confidence:.0%}"
            elif a.is_approaching and a.face_visible:
                color = COLOR_ATTENTION
                label = f"WATCH  yaw:{a.yaw:+.0f}"
            elif a.is_walk_by:
                color = COLOR_SAFE
                label = "walk-by"
            else:
                color = COLOR_SAFE
                label = "ok"

            x1, y1, x2, y2 = a.bbox
            thickness = 3 if a.is_threat else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(
                out, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLOR_LABEL_BG, -1
            )
            cv2.putText(
                out, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
            )
        return out
