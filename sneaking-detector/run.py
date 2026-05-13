#!/usr/bin/env python3
"""Entry point — launch the Sneaking Detector desktop app."""

import sys


def _check_deps() -> None:
    missing = []
    for pkg, imp in [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("Missing required packages:")
        for p in missing:
            print(f"  pip install {p}")
        print("\nRun:  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    _check_deps()
    from sneaking_detector.app import SneakingDetectorApp  # noqa: PLC0415

    app = SneakingDetectorApp()
    app.run()
