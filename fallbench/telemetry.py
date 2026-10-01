"""Raspberry Pi telemetry. Everything degrades gracefully on non-Pi Linux.

Cheap sysfs reads are sampled at 1 Hz in a background thread. `vcgencmd` forks
a process, so it is only called before/after each clip (never during timing).
"""
import os
import platform
import shutil
import subprocess
import threading
import time

import psutil

THERMAL = "/sys/class/thermal/thermal_zone0/temp"
CPUFREQ = "/sys/devices/system/cpu/cpu0/cpufreq/"

# get_throttled bits
_NOW_MASK = 0xF            # bit0 under-voltage, 1 freq capped, 2 throttled, 3 soft temp limit
_STICKY_MASK = 0xF0000     # same four conditions, "has occurred since boot"
_BIT_NAMES = {
    0: "under-voltage", 1: "arm-freq-capped", 2: "throttled", 3: "soft-temp-limit",
    16: "under-voltage-occurred", 17: "freq-cap-occurred",
    18: "throttled-occurred", 19: "soft-temp-limit-occurred",
}


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def read_temp_c():
    v = _read(THERMAL)
    return int(v) / 1000.0 if v and v.lstrip("-").isdigit() else None


def read_freq_mhz():
    v = _read(CPUFREQ + "scaling_cur_freq")
    return int(v) / 1000.0 if v and v.isdigit() else None


def read_governor():
    return _read(CPUFREQ + "scaling_governor")


def vcgencmd_throttled():
    """Return get_throttled as int, or None if vcgencmd is unavailable."""
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=3).stdout.strip()
        return int(out.split("=")[1], 16)
    except Exception:
        return None


def describe_throttle(bits):
    if bits is None:
        return "n/a"
    names = [n for b, n in _BIT_NAMES.items() if bits & (1 << b)]
    return ",".join(names) if names else "none"


def throttle_invalid(before, after):
    """A run is invalid if any throttle/under-voltage condition is active at the
    end, or a NEW 'has occurred' bit appeared during the run."""
    if before is None or after is None:
        return False
    new_sticky = (after & _STICKY_MASK) & ~(before & _STICKY_MASK)
    return bool((after & _NOW_MASK) or new_sticky)


class Sampler(threading.Thread):
    """1 Hz sampler: SoC temperature, CPU clock, system-wide CPU %."""

    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period = period
        self._stop_evt = threading.Event()
        self.temps, self.freqs, self.cpu = [], [], []

    def run(self):
        psutil.cpu_percent(interval=None)
        while not self._stop_evt.wait(self.period):
            t, f = read_temp_c(), read_freq_mhz()
            if t is not None:
                self.temps.append(t)
            if f is not None:
                self.freqs.append(f)
            self.cpu.append(psutil.cpu_percent(interval=None))

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=3)

    def summary(self):
        def mean(x):
            return sum(x) / len(x) if x else None
        return {
            "temp_max_c": max(self.temps) if self.temps else None,
            "freq_mean_mhz": mean(self.freqs),
            "freq_min_mhz": min(self.freqs) if self.freqs else None,
            "sys_cpu_pct_mean": mean(self.cpu),
        }


def cool_down(target_c, min_s, max_s):
    """Sleep at least min_s, then until SoC temp <= target_c (or max_s)."""
    t0 = time.time()
    time.sleep(min_s)
    while time.time() - t0 < max_s:
        t = read_temp_c()
        if t is None or t <= target_c:
            return t
        time.sleep(5)
    return read_temp_c()


def system_info():
    model = _read("/proc/device-tree/model")
    os_pretty = None
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_pretty = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    info = {
        "device_model": (model or "").rstrip("\x00") or platform.node(),
        "os": os_pretty,
        "kernel": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "mem_total_mb": round(psutil.virtual_memory().total / 2**20),
        "governor": read_governor(),
        "freq_mhz_now": read_freq_mhz(),
        "temp_c_now": read_temp_c(),
        "throttled_now": describe_throttle(vcgencmd_throttled()),
    }
    for mod in ("numpy", "torch", "cv2", "mediapipe", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info
