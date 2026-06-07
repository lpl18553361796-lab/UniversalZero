#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORD_SCRIPT="$SCRIPT_DIR/record_system_usage.sh"

INTERVAL_SECONDS="${1:-5}"
DURATION_SECONDS="${2:-300}"

if [ ! -f "$RECORD_SCRIPT" ]; then
  echo "Cannot find record script:"
  echo "  $RECORD_SCRIPT"
  exit 1
fi

if ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || [ "$INTERVAL_SECONDS" -lt 1 ]; then
  echo "Usage: bash monitor_system_usage.sh [interval_seconds] [duration_seconds]"
  echo "Example: bash monitor_system_usage.sh 5 300"
  exit 1
fi

if ! [[ "$DURATION_SECONDS" =~ ^[0-9]+$ ]] || [ "$DURATION_SECONDS" -lt 1 ]; then
  echo "Usage: bash monitor_system_usage.sh [interval_seconds] [duration_seconds]"
  echo "Example: bash monitor_system_usage.sh 5 300"
  exit 1
fi

echo "Monitoring system usage..."
echo "  Interval: ${INTERVAL_SECONDS}s"
echo "  Duration: ${DURATION_SECONDS}s"
echo

START_SECONDS="$(date +%s)"
END_SECONDS=$((START_SECONDS + DURATION_SECONDS))
COUNT=0

while [ "$(date +%s)" -lt "$END_SECONDS" ]; do
  COUNT=$((COUNT + 1))
  echo "Sample $COUNT"
  bash "$RECORD_SCRIPT"
  echo

  NOW_SECONDS="$(date +%s)"
  if [ "$NOW_SECONDS" -ge "$END_SECONDS" ]; then
    break
  fi

  REMAINING_SECONDS=$((END_SECONDS - NOW_SECONDS))
  if [ "$REMAINING_SECONDS" -lt "$INTERVAL_SECONDS" ]; then
    sleep "$REMAINING_SECONDS"
  else
    sleep "$INTERVAL_SECONDS"
  fi
done

echo "Monitoring done."
echo "CSV log:"
echo "  $SCRIPT_DIR/logs/system_usage.csv"
