#!/bin/bash
set -e

# Reuse the Python detected by setup.sh, or detect again
if [ -f .python_cmd ]; then
    PYTHON=$(cat .python_cmd)
elif command -v python &> /dev/null && python -c "import sys; assert sys.version_info >= (3,8)" 2>/dev/null; then
    PYTHON=python
elif command -v py &> /dev/null; then
    PYTHON="py -3"
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo "ERROR: Python 3.8+ not found."
    exit 1
fi

echo "Using Python: $PYTHON"

echo "Starting backend..."
cd backend && $PYTHON -m uvicorn main:app --reload --port 9000 \
    --reload-exclude "*.log" \
    --reload-exclude "*.db" \
    --reload-exclude "*.pid" \
    --reload-exclude "*.pt" \
    --reload-exclude "*.csv" \
    --reload-exclude "*.json" &
BACKEND_PID=$!
cd ..

echo "Starting frontend..."
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

echo "Backend PID:  $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Backend:  http://localhost:9000"
echo "Frontend: http://localhost:5174"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
