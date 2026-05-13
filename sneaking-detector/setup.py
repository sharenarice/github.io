from setuptools import setup, find_packages

setup(
    name="sneaking-detector",
    version="1.0.0",
    description="Real-time desktop app that alerts you when someone sneaks up on you",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "plyer>=2.1.0",
    ],
    extras_require={
        "yolo": ["ultralytics>=8.0.0"],
    },
    entry_points={
        "console_scripts": [
            "sneaking-detector=run:main",
        ],
    },
)
