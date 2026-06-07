#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
CSV_FILE="$LOG_DIR/system_usage.csv"
JSON_FILE="$LOG_DIR/system_usage_latest.json"

mkdir -p "$LOG_DIR"

timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
hostname_value="$(hostname)"

read -r _ user1 nice1 system1 idle1 iowait1 irq1 softirq1 steal1 _ < /proc/stat
sleep 1
read -r _ user2 nice2 system2 idle2 iowait2 irq2 softirq2 steal2 _ < /proc/stat

idle_delta=$((idle2 + iowait2 - idle1 - iowait1))
total_delta=$((user2 + nice2 + system2 + idle2 + iowait2 + irq2 + softirq2 + steal2 - user1 - nice1 - system1 - idle1 - iowait1 - irq1 - softirq1 - steal1))
cpu_percent="$(awk -v idle="$idle_delta" -v total="$total_delta" 'BEGIN { if (total > 0) printf "%.2f", (1 - idle / total) * 100; else print "0.00" }')"

read -r mem_total_kb mem_available_kb <<<"$(awk '
  /MemTotal:/ { total=$2 }
  /MemAvailable:/ { available=$2 }
  END { print total, available }
' /proc/meminfo)"

mem_used_kb=$((mem_total_kb - mem_available_kb))
mem_total_mb="$(awk -v kb="$mem_total_kb" 'BEGIN { printf "%.2f", kb / 1024 }')"
mem_used_mb="$(awk -v kb="$mem_used_kb" 'BEGIN { printf "%.2f", kb / 1024 }')"
mem_available_mb="$(awk -v kb="$mem_available_kb" 'BEGIN { printf "%.2f", kb / 1024 }')"
mem_percent="$(awk -v used="$mem_used_kb" -v total="$mem_total_kb" 'BEGIN { if (total > 0) printf "%.2f", used / total * 100; else print "0.00" }')"

temperature_c="unknown"
if command -v vcgencmd >/dev/null 2>&1; then
  temperature_c="$(vcgencmd measure_temp | tr -cd '0-9.')"
fi

if [ ! -f "$CSV_FILE" ]; then
  echo "timestamp,hostname,cpu_percent,mem_percent,mem_used_mb,mem_available_mb,mem_total_mb,temperature_c" > "$CSV_FILE"
fi

echo "$timestamp,$hostname_value,$cpu_percent,$mem_percent,$mem_used_mb,$mem_available_mb,$mem_total_mb,$temperature_c" >> "$CSV_FILE"

cat > "$JSON_FILE" <<EOF
{
  "timestamp": "$timestamp",
  "hostname": "$hostname_value",
  "cpu_percent": $cpu_percent,
  "memory": {
    "percent": $mem_percent,
    "used_mb": $mem_used_mb,
    "available_mb": $mem_available_mb,
    "total_mb": $mem_total_mb
  },
  "temperature_c": "$temperature_c",
  "csv_file": "$CSV_FILE"
}
EOF

echo "Saved system usage:"
echo "  CSV : $CSV_FILE"
echo "  JSON: $JSON_FILE"
echo
cat "$JSON_FILE"
