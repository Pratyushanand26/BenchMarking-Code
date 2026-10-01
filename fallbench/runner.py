"""Run ONE clip through ONE strategy and return (metrics_row, per_frame_trace).

Timing rules
------------
* Stage timers use time.perf_counter_ns around: pose, reconstruction,
  preprocessing, ST-GCN forward. Their sum is the *pipeline* time for a frame.
* Video decoding is timed separately and is NOT part of the pipeline time.
* CPU cost = time.process_time_ns() (user+sys, all threads, this process only)
  accumulated over the pipeline section of each frame. Immune to other processes.
* GC is disabled during the frame loop, nothing is printed or written inside it.
"""
import gc
import resource
import time
from collections import Counter

import numpy as np
import psutil

from .config import ACTION_CLASSES, FALL_CLASS
from .stats import STAT_KEYS, ms_stats
from .strategies import make_strategy
from .telemetry import (Sampler, describe_throttle, read_temp_c,
                        throttle_invalid, vcgencmd_throttled)
from .video import iter_frames, probe

_pc = time.perf_counter_ns
_cpu = time.process_time_ns


def _rss_mb():
    return psutil.Process().memory_info().rss / 2**20


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB -> MB (Linux)


def run_clip(path, clip_id, strategy_name, k, classifier, pose_factory,
             gt_fall=None, repeat=0):
    info = probe(path)
    fps, W, H = info["fps"], info["width"], info["height"]
    strat = make_strategy(strategy_name, fps, k)

    t_init = _pc()
    pose = pose_factory()
    pose_init_ms = (_pc() - t_init) / 1e6

    # ---- per-frame buffers (python lists of ints; filled inside the hot loop) ----
    pose_ns, recon_ns, prep_ns, fwd_ns = [], [], [], []
    frame_ns, decode_ns = [], []
    ran_infer = []                      # bool per frame
    buf, last_kp = [], None
    pose_calls = pose_miss = forwards = 0
    cpu_pipeline_ns = 0

    pred_counts = Counter()
    n_fall_inf, max_p_fall = 0, 0.0
    first_fall_frame = first_fall_time = None
    first_infer_idx = None
    fall_idx = ACTION_CLASSES.index(FALL_CLASS)

    thr_before = vcgencmd_throttled()
    temp_start = read_temp_c()
    rss_start = _rss_mb()
    sampler = Sampler()
    sampler.start()
    gc.collect()
    gc.disable()
    wall0 = _pc()

    try:
        for i, ts, frame, dec in iter_frames(path, fps):
            decode_ns.append(dec)
            c0, p0 = _cpu(), _pc()

            # ---- pose ----
            kp, pose_t, new_sample = None, 0, False
            if strat.pose_this_frame(i):
                a = _pc()
                kp = pose.extract(frame)
                pose_t = _pc() - a
                pose_calls += 1
                if kp is None:
                    pose_miss += 1
                else:
                    last_kp = kp
                    buf.append(kp)
                    new_sample = True
            elif strat.stale_pose_this_frame(i) and last_kp is not None:
                buf.append(last_kp)          # reuse stale pose (cost_skip)
            if len(buf) > strat.n_samples:
                del buf[0]

            # ---- classify ----
            rec_t = prep_t = fwd_t = 0
            did = False
            if len(buf) == strat.n_samples and strat.infer_this_frame(i, new_sample):
                a = _pc()
                window = strat.reconstruct(np.stack(buf, axis=0))
                rec_t = _pc() - a
                a = _pc()
                pts_t, mot_t = classifier.prepare(window, W, H)
                prep_t = _pc() - a
                a = _pc()
                probs = classifier.forward(pts_t, mot_t)
                fwd_t = _pc() - a
                did = True
                forwards += 1
                if first_infer_idx is None:
                    first_infer_idx = i
                top = int(np.argmax(probs))
                pred_counts[ACTION_CLASSES[top]] += 1
                max_p_fall = max(max_p_fall, float(probs[fall_idx]))
                if top == fall_idx:
                    n_fall_inf += 1
                    if first_fall_frame is None:
                        first_fall_frame, first_fall_time = i, ts

            p1, c1 = _pc(), _cpu()
            frame_ns.append(p1 - p0)
            cpu_pipeline_ns += c1 - c0
            pose_ns.append(pose_t)
            recon_ns.append(rec_t)
            prep_ns.append(prep_t)
            fwd_ns.append(fwd_t)
            ran_infer.append(did)
    finally:
        wall_ns = _pc() - wall0
        gc.enable()
        sampler.stop()
        pose.close()

    thr_after = vcgencmd_throttled()
    tele = sampler.summary()
    n = len(frame_ns)
    duration_s = n / fps if n else 0.0
    pipeline_s = sum(frame_ns) / 1e9

    # ---- steady state = frames from the first inference onward ----
    start = first_infer_idx if first_infer_idx is not None else n
    f_ns = np.asarray(frame_ns, dtype=np.int64)
    idx = np.arange(n)
    arrival = np.array([strat.is_arrival(i) for i in range(n)], dtype=bool)
    steady = idx >= start
    steady_arr = steady & arrival
    period_ns = strat.arrival_period_ns
    # per-frame deadline: finishing within one arrival period
    over_budget = int((f_ns[steady_arr] > period_ns).sum())

    # Live-processing lag: frames arrive in real time (arrival i at i/fps), one
    # worker processes them in order. lag = finish time - arrival time. It stays
    # small if the pipeline keeps up and grows without bound if RTF > 1.
    lag_ns = np.full(n, -1.0)
    finish = 0.0
    for i in np.nonzero(arrival)[0]:
        a = i * 1e9 / fps
        finish = max(a, finish) + f_ns[i]
        lag_ns[i] = finish - a
    steady_ns = f_ns[steady_arr]          # tail-latency stats: frames the camera delivered

    pose_arr = np.asarray(pose_ns, dtype=np.int64)
    ran_arr = np.asarray(ran_infer, dtype=bool)
    stage = {
        "pose": pose_arr[pose_arr > 0],
        "recon": np.asarray(recon_ns, dtype=np.int64)[ran_arr],
        "prep": np.asarray(prep_ns, dtype=np.int64)[ran_arr],
        "fwd": np.asarray(fwd_ns, dtype=np.int64)[ran_arr],
        "frame": steady_ns,
    }

    detected = n_fall_inf > 0
    row = {
        "strategy": strategy_name, "k": strat.k_label, "repeat": repeat,
        "clip_id": clip_id,
        "gt_fall": "" if gt_fall is None else int(gt_fall),
        "frames": n, "native_fps": round(fps, 3), "width": W, "height": H,
        "duration_s": round(duration_s, 4),
        # work counts (exact, platform independent)
        "pose_calls": pose_calls, "pose_misses": pose_miss, "forward_passes": forwards,
        "pose_calls_per_video_s": pose_calls / duration_s if duration_s else 0.0,
        "forward_per_video_s": forwards / duration_s if duration_s else 0.0,
        # time
        "decode_s": sum(decode_ns) / 1e9, "pipeline_s": pipeline_s, "wall_s": wall_ns / 1e9,
        # steady-state real-time factor (frames from the first inference onward)
        "steady_video_frames": int(steady.sum()),
        "steady_pipeline_s": float(f_ns[steady].sum() / 1e9),
        "steady_video_s": float(steady.sum() / fps),
        "rtf": float(f_ns[steady].sum() / 1e9 / (steady.sum() / fps)) if steady.any() else float("nan"),
        "rtf_whole_clip": pipeline_s / duration_s if duration_s else float("nan"),
        "pipeline_fps": n / pipeline_s if pipeline_s else float("nan"),
        "period_ms": period_ns / 1e6,
        "steady_frames": int(steady_ns.size), "over_budget_frames": over_budget,
        "over_budget_pct": 100.0 * over_budget / steady_ns.size if steady_ns.size else float("nan"),
        "amortized_ms_per_video_frame": float(f_ns[steady].mean() / 1e6) if steady.any() else float("nan"),
        "lag_ms_mean": float(lag_ns[steady_arr].mean() / 1e6) if steady_arr.any() else float("nan"),
        "lag_ms_p95": float(np.percentile(lag_ns[steady_arr], 95) / 1e6) if steady_arr.any() else float("nan"),
        "lag_ms_max": float(lag_ns[steady_arr].max() / 1e6) if steady_arr.any() else float("nan"),
        # cpu
        "cpu_s": cpu_pipeline_ns / 1e9,
        "cpu_ms_per_frame": cpu_pipeline_ns / 1e6 / n if n else float("nan"),
        "cpu_ms_per_video_s": cpu_pipeline_ns / 1e6 / duration_s if duration_s else float("nan"),
        "core_util": cpu_pipeline_ns / sum(frame_ns) if sum(frame_ns) else float("nan"),
        "sys_cpu_pct_mean": tele["sys_cpu_pct_mean"],
        # memory
        "rss_start_mb": rss_start, "rss_end_mb": _rss_mb(), "peak_rss_mb": _peak_rss_mb(),
        "pose_init_ms": pose_init_ms,
        # pi telemetry
        "temp_start_c": temp_start, "temp_max_c": tele["temp_max_c"],
        "freq_mean_mhz": tele["freq_mean_mhz"], "freq_min_mhz": tele["freq_min_mhz"],
        "throttled_before": describe_throttle(thr_before),
        "throttled_after": describe_throttle(thr_after),
        "invalid": int(throttle_invalid(thr_before, thr_after)),
        # accuracy
        "fall_detected": int(detected),
        "correct": "" if gt_fall is None else int(detected == bool(gt_fall)),
        "first_fall_frame": "" if first_fall_frame is None else first_fall_frame,
        "first_fall_time_s": "" if first_fall_time is None else round(first_fall_time, 4),
        "n_fall_inferences": n_fall_inf, "max_prob_fall": max_p_fall,
        "dominant_action": pred_counts.most_common(1)[0][0] if pred_counts else "",
    }
    for stage_name, arr in stage.items():
        s = ms_stats(arr)
        for key in STAT_KEYS:
            row[f"{stage_name}_ms_{key}"] = s[key]

    trace = {
        "frame_ns": f_ns,
        "pose_ns": pose_arr,
        "recon_ns": np.asarray(recon_ns, dtype=np.int64),
        "prep_ns": np.asarray(prep_ns, dtype=np.int64),
        "fwd_ns": np.asarray(fwd_ns, dtype=np.int64),
        "decode_ns": np.asarray(decode_ns, dtype=np.int64),
        "steady": steady,
        "arrival": arrival,
        "lag_ns": lag_ns,
    }
    return row, trace
