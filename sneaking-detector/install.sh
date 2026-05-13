#!/usr/bin/env bash
# Quick-install script for Sneaking Detector
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Sneaking Detector installer ==="

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3,9), 'Python 3.9+ required'" \
    || { echo "ERROR: Python 3.9+ is required."; exit 1; }

# Create venv if not inside one
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    # shellcheck source=/dev/null
    source .venv/bin/activate
    echo "Activated: $VIRTUAL_ENV"
fi

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "=== Installation complete ==="
echo ""
echo "To run the app:"
if [ -d ".venv" ]; then
    echo "  source .venv/bin/activate"
fi
echo "  python run.py"
echo ""
echo "For faster detection, also install YOLOv8:"
echo "  pip install ultralytics"
