#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv-pi"

cd "$PROJECT_ROOT"

if [ ! -d "$VENV_DIR" ]; then
  echo "Python environment not found."
  echo "Run this first:"
  echo "  bash raspberry_pi/install_pi_launcher.sh"
  read -r -p "Press Enter to close..."
  exit 1
fi

source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/games:$PROJECT_ROOT/core"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "Starting UniversalZero..."
echo "Open this address on the Raspberry Pi:"
echo "  http://localhost:8501"
echo

python -m streamlit run ui/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --browser.gatherUsageStats=false
