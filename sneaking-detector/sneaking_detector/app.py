"""
Sneaking Detector — main GUI application.

Layout
──────
┌─────────────────────────────────────────────────────┐
│  [camera preview with bounding-box overlays]        │
├──────────────────┬──────────────────────────────────┤
│  STATUS PANEL    │  CONTROLS & LOG                  │
│  (colour-coded)  │                                  │
└──────────────────┴──────────────────────────────────┘

Thread model:
  • Capture/processing runs in a daemon thread writing to a shared queue.
  • The tkinter after()-loop pulls frames from the queue to update the UI.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from .config import (
    ALERT_COOLDOWN_SECONDS,
    CAMERA_INDEX,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    CONFIDENCE_THRESHOLD,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)
from .notifier import AlertNotifier
from .vision import PersonAssessment, VisionPipeline


# ---------------------------------------------------------------------------
# Colour palette (dark theme, hex)
# ---------------------------------------------------------------------------
BG_DARK = "#0f0f1a"
BG_PANEL = "#16162a"
BG_WIDGET = "#1e1e36"
FG = "#e0e0f0"
FG_DIM = "#7070a0"

STATUS_SAFE = "#1a8c1a"
STATUS_WARN = "#b87800"
STATUS_ALERT = "#c81010"

ACCENT = "#4040c0"
BTN_START = "#1a6e1a"
BTN_STOP = "#6e1a1a"
BTN_FG = "#e0ffe0"


# ---------------------------------------------------------------------------
class SneakingDetectorApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Sneaking Detector")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        self._frame_q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=2)
        self._result_q: queue.Queue[List[PersonAssessment]] = queue.Queue(maxsize=2)

        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._pipeline: Optional[VisionPipeline] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

        self._alert_active = False
        self._frame_count = 0
        self._fps_acc: List[float] = []
        self._last_frame_t = time.time()

        # Sensitivity: 0–100, maps to scale factor for APPROACH_RATE_THRESHOLD
        self._sensitivity_var = tk.IntVar(value=50)
        self._notifications_var = tk.BooleanVar(value=True)
        self._show_debug_var = tk.BooleanVar(value=False)
        self._camera_var = tk.StringVar(value=str(CAMERA_INDEX))

        self.notifier = AlertNotifier(
            cooldown=ALERT_COOLDOWN_SECONDS,
            on_state_change=self._on_alert_state_change,
        )

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = self.root

        # ── Top: camera preview ──────────────────────────────────────
        self._preview_label = tk.Label(
            root, bg="black", width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT
        )
        self._preview_label.pack(padx=0, pady=0)
        self._set_placeholder_preview()

        # ── Bottom row ───────────────────────────────────────────────
        bottom = tk.Frame(root, bg=BG_DARK)
        bottom.pack(fill=tk.X, padx=6, pady=6)

        # Status panel (left)
        self._status_frame = tk.Frame(bottom, bg=STATUS_SAFE, width=200)
        self._status_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        self._status_frame.pack_propagate(False)

        self._status_title = tk.Label(
            self._status_frame,
            text="SAFE",
            font=("Helvetica", 22, "bold"),
            bg=STATUS_SAFE, fg="white",
        )
        self._status_title.pack(expand=True)

        self._status_sub = tk.Label(
            self._status_frame,
            text="No threats detected",
            font=("Helvetica", 9),
            bg=STATUS_SAFE, fg="#d0ffd0", wraplength=190,
        )
        self._status_sub.pack(pady=(0, 8))

        self._fps_label = tk.Label(
            self._status_frame,
            text="FPS: --",
            font=("Helvetica", 8),
            bg=STATUS_SAFE, fg="#a0e0a0",
        )
        self._fps_label.pack(pady=(0, 4))

        # Controls panel (right)
        ctrl = tk.Frame(bottom, bg=BG_PANEL, pady=6, padx=8)
        ctrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Row 1: start/stop + camera selector
        row1 = tk.Frame(ctrl, bg=BG_PANEL)
        row1.pack(fill=tk.X, pady=(0, 6))

        self._start_btn = tk.Button(
            row1, text="▶  Start", command=self._start,
            bg=BTN_START, fg=BTN_FG, font=("Helvetica", 11, "bold"),
            relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 6))

        self._stop_btn = tk.Button(
            row1, text="■  Stop", command=self._stop,
            bg=BTN_STOP, fg=BTN_FG, font=("Helvetica", 11, "bold"),
            relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
            state=tk.DISABLED,
        )
        self._stop_btn.pack(side=tk.LEFT)

        cam_lbl = tk.Label(
            row1, text="Camera index:", bg=BG_PANEL, fg=FG_DIM,
            font=("Helvetica", 9),
        )
        cam_lbl.pack(side=tk.LEFT, padx=(16, 4))
        cam_entry = tk.Entry(
            row1, textvariable=self._camera_var, width=3,
            bg=BG_WIDGET, fg=FG, insertbackground=FG,
            relief=tk.FLAT, font=("Helvetica", 9),
        )
        cam_entry.pack(side=tk.LEFT)

        # Row 2: sensitivity slider
        row2 = tk.Frame(ctrl, bg=BG_PANEL)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(
            row2, text="Sensitivity:", bg=BG_PANEL, fg=FG_DIM,
            font=("Helvetica", 9), width=10, anchor="w",
        ).pack(side=tk.LEFT)
        tk.Scale(
            row2,
            from_=10, to=90,
            orient=tk.HORIZONTAL,
            variable=self._sensitivity_var,
            bg=BG_PANEL, fg=FG, troughcolor=BG_WIDGET,
            highlightthickness=0, length=180,
            font=("Helvetica", 8),
        ).pack(side=tk.LEFT)
        tk.Label(
            row2,
            text="(lower = more sensitive)",
            bg=BG_PANEL, fg=FG_DIM, font=("Helvetica", 8),
        ).pack(side=tk.LEFT, padx=4)

        # Row 3: toggles
        row3 = tk.Frame(ctrl, bg=BG_PANEL)
        row3.pack(fill=tk.X, pady=2)
        tk.Checkbutton(
            row3, text="Desktop notifications",
            variable=self._notifications_var,
            bg=BG_PANEL, fg=FG, selectcolor=BG_WIDGET,
            activebackground=BG_PANEL, activeforeground=FG,
            font=("Helvetica", 9),
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            row3, text="Show debug labels",
            variable=self._show_debug_var,
            bg=BG_PANEL, fg=FG, selectcolor=BG_WIDGET,
            activebackground=BG_PANEL, activeforeground=FG,
            font=("Helvetica", 9),
        ).pack(side=tk.LEFT)

        # Row 4: event log
        row4 = tk.Frame(ctrl, bg=BG_PANEL)
        row4.pack(fill=tk.X, pady=(6, 0))
        tk.Label(
            row4, text="Event log:", bg=BG_PANEL, fg=FG_DIM,
            font=("Helvetica", 9),
        ).pack(anchor="w")

        log_frame = tk.Frame(row4, bg=BG_WIDGET)
        log_frame.pack(fill=tk.X)
        self._log_text = tk.Text(
            log_frame, height=4, bg=BG_WIDGET, fg=FG, font=("Courier", 8),
            relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD,
        )
        sb = tk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text.pack(fill=tk.X)

    # ------------------------------------------------------------------
    # Placeholder / preview helpers
    # ------------------------------------------------------------------
    def _set_placeholder_preview(self) -> None:
        placeholder = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Press  ▶ Start  to begin",
            (PREVIEW_WIDTH // 2 - 160, PREVIEW_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 130), 1, cv2.LINE_AA,
        )
        self._show_frame(placeholder)

    def _show_frame(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self._preview_label.configure(image=photo)
        self._preview_label.image = photo  # keep reference

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------
    def _start(self) -> None:
        try:
            cam_idx = int(self._camera_var.get())
        except ValueError:
            cam_idx = CAMERA_INDEX

        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self._log(f"ERROR: Cannot open camera {cam_idx}")
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        self._pipeline = VisionPipeline(frame_w=actual_w, frame_h=actual_h)
        self._running = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._capture_thread.start()
        self._process_thread.start()

        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._log(f"Started — camera {cam_idx} ({actual_w}x{actual_h})")
        self._poll_results()

    def _stop(self) -> None:
        self._running = False
        time.sleep(0.15)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._set_status_safe()
        self._set_placeholder_preview()
        self._log("Stopped.")

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                continue
            # Discard stale frames to stay real-time
            if not self._frame_q.full():
                try:
                    self._frame_q.put_nowait(frame)
                except queue.Full:
                    pass

    def _process_loop(self) -> None:
        while self._running:
            try:
                frame = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._pipeline is None:
                continue

            # Dynamic threshold from sensitivity slider
            sensitivity = self._sensitivity_var.get() / 100.0  # 0.1–0.9
            import sneaking_detector.config as cfg  # noqa: PLC0415
            cfg.APPROACH_RATE_THRESHOLD = 0.18 * (1.0 - sensitivity) + 0.02

            assessments = self._pipeline.process(frame)

            # Overlay
            if self._show_debug_var.get():
                frame = self._pipeline.draw_overlays(frame, assessments)
            else:
                frame = self._draw_minimal_overlays(frame, assessments)

            # Resize for preview
            preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))

            try:
                self._result_q.put_nowait((preview, assessments))
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # UI polling (runs on main thread via after())
    # ------------------------------------------------------------------
    def _poll_results(self) -> None:
        if not self._running:
            return

        try:
            preview, assessments = self._result_q.get_nowait()
        except queue.Empty:
            self.root.after(30, self._poll_results)
            return

        self._show_frame(preview)
        self._update_status(assessments)
        self._update_fps()
        self.root.after(30, self._poll_results)

    # ------------------------------------------------------------------
    # Status update
    # ------------------------------------------------------------------
    def _update_status(self, assessments: List[PersonAssessment]) -> None:
        threats = [a for a in assessments if a.is_threat]
        watchers = [
            a for a in assessments
            if not a.is_threat and a.is_approaching and a.face_visible
        ]

        if threats:
            max_conf = max(a.threat_confidence for a in threats)
            self._set_status(
                STATUS_ALERT,
                "ALERT",
                f"{len(threats)} person(s) sneaking up!  conf {max_conf:.0%}",
            )
            if self._notifications_var.get():
                self.notifier.evaluate(True, max_conf)
            if not self._alert_active:
                self._alert_active = True
                self._log(
                    f"ALERT: {len(threats)} threat(s)  conf={max_conf:.0%}"
                )
        elif watchers:
            self._set_status(
                STATUS_WARN,
                "WATCH",
                f"{len(watchers)} person(s) nearby — monitoring…",
            )
            self.notifier.evaluate(False)
            self._alert_active = False
        else:
            self._set_status(STATUS_SAFE, "SAFE", "No threats detected")
            self.notifier.evaluate(False)
            self._alert_active = False

    def _set_status(self, color: str, title: str, sub: str) -> None:
        for widget in (
            self._status_frame,
            self._status_title,
            self._status_sub,
            self._fps_label,
        ):
            widget.configure(bg=color)
        self._status_title.configure(text=title)
        self._status_sub.configure(text=sub)

    def _set_status_safe(self) -> None:
        self._set_status(STATUS_SAFE, "SAFE", "No threats detected")

    def _update_fps(self) -> None:
        now = time.time()
        self._fps_acc.append(now - self._last_frame_t)
        self._last_frame_t = now
        if len(self._fps_acc) > 20:
            self._fps_acc.pop(0)
        avg = sum(self._fps_acc) / len(self._fps_acc)
        fps = 1.0 / avg if avg > 0 else 0.0
        self._fps_label.configure(text=f"FPS: {fps:.1f}")

    # ------------------------------------------------------------------
    # Minimal overlay (no debug labels)
    # ------------------------------------------------------------------
    def _draw_minimal_overlays(
        self, frame: np.ndarray, assessments: List[PersonAssessment]
    ) -> np.ndarray:
        from .config import COLOR_SAFE, COLOR_ATTENTION, COLOR_THREAT

        out = frame.copy()
        for a in assessments:
            if a.is_threat:
                color = COLOR_THREAT
                thickness = 3
            elif a.is_approaching and a.face_visible:
                color = COLOR_ATTENTION
                thickness = 2
            else:
                color = COLOR_SAFE
                thickness = 1
            x1, y1, x2, y2 = a.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        return out

    # ------------------------------------------------------------------
    # Notification callback (arrives from notifier thread)
    # ------------------------------------------------------------------
    def _on_alert_state_change(self, is_threat: bool, confidence: float) -> None:
        pass  # visual state is already handled in _update_status

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, line)
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        self._running = False
        time.sleep(0.1)
        if self._cap:
            self._cap.release()
        self.root.destroy()

    def run(self) -> None:
        # Centre on screen
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ww = self.root.winfo_width()
        wh = self.root.winfo_height()
        self.root.geometry(f"+{(sw - ww) // 2}+{(sh - wh) // 2}")
        self.root.mainloop()
