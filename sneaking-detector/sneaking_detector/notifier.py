"""
Cross-platform desktop notification dispatcher.

Priority order:
  1. plyer  (cross-platform, pip install plyer)
  2. macOS  osascript
  3. Linux  notify-send
  4. Windows win10toast

Alerts fire at most once per ALERT_COOLDOWN_SECONDS to avoid spam.
"""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from typing import Callable, Optional

from .config import ALERT_COOLDOWN_SECONDS, CONFIDENCE_THRESHOLD


class AlertNotifier:
    def __init__(
        self,
        cooldown: float = ALERT_COOLDOWN_SECONDS,
        on_state_change: Optional[Callable[[bool, float], None]] = None,
    ):
        self._cooldown = cooldown
        self._last_fired: float = 0.0
        self._alert_active: bool = False
        self.on_state_change = on_state_change   # (is_threat, confidence)

    # ------------------------------------------------------------------
    def evaluate(self, threat_detected: bool, confidence: float = 1.0) -> None:
        """Call once per processed frame with the current threat state."""
        if threat_detected and confidence >= CONFIDENCE_THRESHOLD:
            now = time.time()
            if now - self._last_fired >= self._cooldown:
                self._last_fired = now
                self._fire(confidence)
            if not self._alert_active:
                self._alert_active = True
                if self.on_state_change:
                    self.on_state_change(True, confidence)
        else:
            if self._alert_active:
                self._alert_active = False
                if self.on_state_change:
                    self.on_state_change(False, 0.0)

    # ------------------------------------------------------------------
    def _fire(self, confidence: float) -> None:
        threading.Thread(
            target=self._send_os_notification,
            args=(confidence,),
            daemon=True,
        ).start()

    def _send_os_notification(self, confidence: float) -> None:
        title = "Sneaking Alert!"
        body = f"Someone is approaching and watching you! ({confidence:.0%})"

        # 1. plyer
        try:
            from plyer import notification  # type: ignore
            notification.notify(
                title=title,
                message=body,
                app_name="Sneaking Detector",
                timeout=5,
            )
            return
        except Exception:
            pass

        sys = platform.system()

        # 2. macOS
        if sys == "Darwin":
            try:
                script = (
                    f'display notification "{body}" '
                    f'with title "{title}" sound name "Basso"'
                )
                subprocess.run(
                    ["osascript", "-e", script],
                    check=False, capture_output=True,
                )
                return
            except Exception:
                pass

        # 3. Linux
        elif sys == "Linux":
            try:
                subprocess.run(
                    ["notify-send", "-u", "critical", "-t", "5000", title, body],
                    check=False, capture_output=True,
                )
                return
            except Exception:
                pass

        # 4. Windows (win10toast)
        elif sys == "Windows":
            try:
                from win10toast import ToastNotifier  # type: ignore
                ToastNotifier().show_toast(title, body, duration=5, threaded=True)
            except Exception:
                pass

        # All methods exhausted — the GUI visual alert is the fallback.
