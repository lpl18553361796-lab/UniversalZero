#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "UniversalZero Raspberry Pi lazy installer"
echo "Project root: $PROJECT_ROOT"
echo

if [ ! -d "$PROJECT_ROOT" ]; then
  echo "Project folder not found:"
  echo "  $PROJECT_ROOT"
  echo
  echo "Please check the folder name, then run this script again."
  read -r -p "Press Enter to close..."
  exit 1
fi

cd "$PROJECT_ROOT"

wait_for_apt() {
  echo "Checking apt/dpkg lock..."
  while pgrep -x apt >/dev/null 2>&1 || \
        pgrep -x apt-get >/dev/null 2>&1 || \
        pgrep -x dpkg >/dev/null 2>&1 || \
        pgrep -x unattended-upgr >/dev/null 2>&1; do
    echo "Another install/update process is running. Waiting 10 seconds..."
    sleep 10
  done
}

wait_for_apt

echo "Updating package list..."
sudo apt update

wait_for_apt

echo "Installing Python tools..."
sudo apt install -y python3 python3-venv python3-pip python3-torch

echo "Installing UniversalZero launcher..."
bash raspberry_pi/install_pi_launcher.sh

echo
echo "All done."
echo "You can now double-click UniversalZero on the Raspberry Pi desktop."
echo "Or start it manually with:"
echo "  bash \"$PROJECT_ROOT/raspberry_pi/run_universalzero_pi.sh\""
echo
read -r -p "Press Enter to close..."
