"""
Multi-person centroid tracker with approach and lateral-motion detection.

Each TrackedPerson stores a rolling history of bounding-box area and centroid
positions so we can distinguish "approaching and watching" from "walking by."
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .config import (
    APPROACH_RATE_THRESHOLD,
    APPROACH_FRAMES_WINDOW,
    MIN_FRAMES_TO_CONFIRM,
    LATERAL_DOMINANCE_RATIO,
)


@dataclass
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    frame_w: int
    frame_h: int

    area_history: List[float] = field(default_factory=list)
    centroid_history: List[Tuple[float, float]] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    frames_tracked: int = 0

    # ---------- derived state (refreshed each update) ----------
    area_growth_rate: float = 0.0   # positive = approaching
    lateral_ratio: float = 0.0      # 1.0 = pure lateral walk-by
    is_approaching: bool = False
    is_walk_by: bool = False

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        w, h = self.frame_w, self.frame_h
        area = ((x2 - x1) * (y2 - y1)) / (w * h)          # normalised
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h

        self.area_history.append(area)
        self.centroid_history.append((cx, cy))
        self.frames_tracked += 1
        self.last_seen = time.time()

        # Keep only the rolling window
        n = APPROACH_FRAMES_WINDOW + 2
        if len(self.area_history) > n:
            self.area_history = self.area_history[-n:]
            self.centroid_history = self.centroid_history[-n:]

        self._recompute()

    def _recompute(self) -> None:
        win = min(APPROACH_FRAMES_WINDOW, len(self.area_history))
        if win < 4:
            self.area_growth_rate = 0.0
            self.lateral_ratio = 0.0
            self.is_approaching = False
            self.is_walk_by = False
            return

        areas = self.area_history[-win:]
        base = areas[0] if areas[0] > 0 else 1e-6
        self.area_growth_rate = (areas[-1] - areas[0]) / base

        # Lateral vs. forward motion via centroid displacement
        cx_vals = [c[0] for c in self.centroid_history[-win:]]
        cy_vals = [c[1] for c in self.centroid_history[-win:]]
        lateral_disp = abs(cx_vals[-1] - cx_vals[0])
        vertical_disp = abs(cy_vals[-1] - cy_vals[0])
        total_disp = lateral_disp + vertical_disp + 1e-6
        self.lateral_ratio = lateral_disp / total_disp

        self.is_approaching = (
            self.area_growth_rate > APPROACH_RATE_THRESHOLD
            and self.frames_tracked >= MIN_FRAMES_TO_CONFIRM
        )
        # Walk-by: lateral motion dominates AND area is not growing fast
        self.is_walk_by = (
            self.lateral_ratio > LATERAL_DOMINANCE_RATIO
            and self.area_growth_rate < APPROACH_RATE_THRESHOLD * 1.5
        )

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class CentroidTracker:
    """IoU-based greedy matching tracker."""

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        max_disappeared: int = 20,
        min_iou: float = 0.20,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou

        self._next_id = 0
        self.persons: Dict[int, TrackedPerson] = {}
        self._disappeared: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def update(
        self, detections: List[Tuple[int, int, int, int]]
    ) -> Dict[int, TrackedPerson]:
        if not detections:
            for tid in list(self._disappeared):
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)
            return self.persons

        if not self.persons:
            for det in detections:
                self._register(det)
            return self.persons

        track_ids = list(self.persons.keys())
        track_boxes = [self.persons[tid].bbox for tid in track_ids]
        iou_mat = self._build_iou_matrix(detections, track_boxes)

        matched_dets: set = set()
        matched_tracks: set = set()

        # Greedy best-match loop
        while True:
            if iou_mat.size == 0:
                break
            flat_idx = np.argmax(iou_mat)
            di, ti = np.unravel_index(flat_idx, iou_mat.shape)
            if iou_mat[di, ti] < self.min_iou:
                break
            tid = track_ids[ti]
            self.persons[tid].update(detections[di])
            self._disappeared[tid] = 0
            matched_dets.add(di)
            matched_tracks.add(ti)
            iou_mat[di, :] = -1.0
            iou_mat[:, ti] = -1.0

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._register(det)

        for j, tid in enumerate(track_ids):
            if j not in matched_tracks:
                self._disappeared[tid] = self._disappeared.get(tid, 0) + 1
                if self._disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)

        return self.persons

    # ------------------------------------------------------------------
    def _register(self, bbox: Tuple) -> None:
        p = TrackedPerson(
            track_id=self._next_id,
            bbox=bbox,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
        )
        p.update(bbox)  # seed history
        self.persons[self._next_id] = p
        self._disappeared[self._next_id] = 0
        self._next_id += 1

    def _deregister(self, tid: int) -> None:
        self.persons.pop(tid, None)
        self._disappeared.pop(tid, None)

    @staticmethod
    def _iou(b1: Tuple, b2: Tuple) -> float:
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        denom = a1 + a2 - inter
        return inter / denom if denom > 0 else 0.0

    def _build_iou_matrix(
        self,
        dets: List[Tuple],
        tracks: List[Tuple],
    ) -> np.ndarray:
        mat = np.zeros((len(dets), len(tracks)), dtype=np.float32)
        for i, d in enumerate(dets):
            for j, t in enumerate(tracks):
                mat[i, j] = self._iou(d, t)
        return mat
