Raspberry Pi extra tools
========================

Copy this whole folder to the Raspberry Pi when you need the extra scripts:

  pi_tools/

Record current CPU and memory usage:

  chmod +x /home/lpl/UniversalZero-main/UniversalZero-main/pi_tools/record_system_usage.sh
  bash /home/lpl/UniversalZero-main/UniversalZero-main/pi_tools/record_system_usage.sh

Record CPU and memory usage continuously:

  chmod +x /home/lpl/UniversalZero-main/UniversalZero-main/pi_tools/monitor_system_usage.sh
  bash /home/lpl/UniversalZero-main/UniversalZero-main/pi_tools/monitor_system_usage.sh 5 300

The two numbers mean:

  5   = record once every 5 seconds
  300 = record for 300 seconds

Output files:

  pi_tools/logs/system_usage.csv
  pi_tools/logs/system_usage_latest.json

The CSV file keeps every run. The JSON file keeps only the latest snapshot.
