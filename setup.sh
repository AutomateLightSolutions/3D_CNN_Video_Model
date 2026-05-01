#!/bin/bash
set -e

# Detect the real Python on Windows (Git Bash) or Linux/Mac
if command -v python &> /dev/null && python -c "import sys; assert sys.version_info >= (3,8)" 2>/dev/null; then
    PYTHON=python
elif command -v py &> /dev/null; then
    PYTHON="py -3"
elif command -v python3 &> /dev/null && python3 -c "import sys; assert sys.version_info >= (3,8)" 2>/dev/null; then
    PYTHON=python3
else
    echo "ERROR: Python 3.8+ not found. Install from https://www.python.org/downloads/"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"

echo "Installing Python dependencies..."
$PYTHON -m pip install -r backend/requirements.txt

echo "Downloading FFmpeg binaries into backend/bin/ ..."
$PYTHON backend/download_ffmpeg.py

echo "Installing npm dependencies..."
cd frontend && npm install && cd ..

echo "Creating runtime directories..."
mkdir -p ~/highlight_system/{clips,models,exports}

# Write detected Python to a file so start.sh can reuse it
echo "$PYTHON" > .python_cmd

echo ""
echo "Setup complete. Run ./start.sh to launch."
