#!/bin/bash
# Launch Jupyter Notebook for the ML Pipeline
# Usage: ./launch_jupyter.sh

set -e

echo "=============================================="
echo "ML Pipeline - Jupyter Launcher"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found."
    echo "Please run this script from the project root directory:"
    echo "  cd BA798_WEEK1_PRACTICE"
    echo "  ./launch_jupyter.sh"
    exit 1
fi

echo "[1/4] Checking dependencies..."
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.9+"
    exit 1
fi

echo "[2/4] Installing dependencies (if needed)..."
python -m pip install -q jupyter ipykernel 2>/dev/null || true

echo "[3/4] Starting Jupyter Notebook Server..."
echo ""
echo "The notebook will open in your default browser."
echo "If it doesn't, navigate to: http://localhost:8888"
echo ""
echo "To open the pipeline notebook:"
echo "  1. Click on: notebooks/"
echo "  2. Click on: 05_main_pipeline.ipynb"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=============================================="
echo ""

# Start Jupyter
jupyter notebook --notebook-dir=. &

# Wait a moment for Jupyter to start
sleep 2

echo "Jupyter is running!"
echo ""
echo "Next steps:"
echo "  1. Navigate to notebooks/"
echo "  2. Open 05_main_pipeline.ipynb"
echo "  3. Run cells with Shift+Enter or Cell > Run All"
echo ""
echo "See JUPYTER_GUIDE.md for more information"
