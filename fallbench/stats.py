import numpy as np

STAT_KEYS = ("mean", "median", "p95", "p99", "max")


def ms_stats(arr_ns):
    """Summary of a list/array of nanosecond durations, in milliseconds."""
    a = np.asarray(arr_ns, dtype=np.float64) / 1e6
    if a.size == 0:
        return {k: float("nan") for k in STAT_KEYS}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }
