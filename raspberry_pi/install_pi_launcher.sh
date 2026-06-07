#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv-pi"
DESKTOP_DIR="$HOME/Desktop"
DESKTOP_FILE="$DESKTOP_DIR/UniversalZero.desktop"

cd "$PROJECT_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is not installed."
  echo "Install it with:"
  echo "  sudo apt update && sudo apt install -y python3 python3-venv python3-pip python3-torch"
  exit 1
fi

if [ -d "$VENV_DIR" ] && ! grep -q "include-system-site-packages = true" "$VENV_DIR/pyvenv.cfg" 2>/dev/null; then
  echo "Existing Python environment cannot see apt-installed PyTorch."
  echo "Recreating Python environment with system packages enabled..."
  rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python environment..."
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Configuring pip mirror..."
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.timeout 120

echo "Upgrading pip..."
python -m pip install --upgrade pip --retries 10 --timeout 120 --no-cache-dir

echo "Installing UniversalZero dependencies..."
for attempt in 1 2 3 4 5; do
  echo "Dependency install attempt $attempt/5..."
  if python -m pip install -r requirements-pi.txt --retries 10 --timeout 120 --no-cache-dir; then
    break
  fi
  if [ "$attempt" = "5" ]; then
    echo "Dependency install failed after 5 attempts."
    exit 1
  fi
  echo "Network interrupted. Retrying in 10 seconds..."
  sleep 10
done

echo "Checking PyTorch..."
python - <<'PY'
import torch
print("PyTorch OK:", torch.__version__)
PY

chmod +x "$PROJECT_ROOT/raspberry_pi/run_universalzero_pi.sh"

mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=UniversalZero
Comment=Launch UniversalZero Streamlit UI
Exec=bash "$PROJECT_ROOT/raspberry_pi/run_universalzero_pi.sh"
Path=$PROJECT_ROOT
Terminal=true
Categories=Education;Science;
EOF

chmod +x "$DESKTOP_FILE"

echo
echo "Done."
echo "A desktop launcher was created:"
echo "  $DESKTOP_FILE"
echo
echo "Double-click UniversalZero on the desktop to start the app."
