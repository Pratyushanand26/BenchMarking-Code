#!/usr/bin/env bash
# Usage: scripts/governor.sh performance   (before benchmarking)
#        scripts/governor.sh ondemand      (restore afterwards; Pi 5 default is 'ondemand')
set -euo pipefail
G="${1:-performance}"
echo "$G" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
echo "governor now: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
