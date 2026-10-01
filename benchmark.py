#!/usr/bin/env python3
"""Orchestrator: runs all 5 strategies, R repeats, on every clip in videos/.

    python benchmark.py                      # full run (5 strategies x 5 repeats)
    python benchmark.py --repeats 1 --limit 3    # quick dry run

Protocol (see README):
  * each (strategy, repeat) is a fresh process -> honest peak RSS
  * strategy order is shuffled inside every repeat (thermal drift can't favour one)
  * cool-down to --cool-temp before every run
  * a run containing any throttled / under-voltage clip is discarded and re-run
  * resumable: finished runs in results/runs/ are skipped
"""
import argparse
import csv
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

from fallbench import summarize
from fallbench.config import DEFAULT_K, DEFAULT_WEIGHTS, STRATEGIES
from fallbench.telemetry import cool_down, system_info
from fallbench.video import list_videos


def run_is_invalid(csv_path):
    with open(csv_path, newline="") as f:
        return any(r.get("invalid") == "1" for r in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="videos")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--results", default="results")
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch threads, identical for every strategy")
    ap.add_argument("--cpus", default="",
                    help="taskset core list, e.g. 0-3 (empty = no pinning)")
    ap.add_argument("--cool-temp", type=float, default=55.0)
    ap.add_argument("--cool-min", type=float, default=20.0)
    ap.add_argument("--cool-max", type=float, default=600.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only first N clips (dry run)")
    a = ap.parse_args()

    if not os.path.exists(a.weights):
        sys.exit(f"missing weights: {a.weights}\n"
                 f"copy tsstg-model.pth there (see README step 4).")
    clips = list_videos(a.videos)
    if not clips:
        sys.exit(f"no video files in {a.videos}/ (see README step 5).")
    n_clips = min(len(clips), a.limit) if a.limit else len(clips)

    runs_dir = os.path.join(a.results, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    info = system_info()
    info.update(vars(a))
    info["n_clips"] = n_clips
    with open(os.path.join(a.results, "env.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(json.dumps({k: info[k] for k in
                      ("device_model", "os", "arch", "python", "governor", "cpu_count",
                       "torch", "mediapipe", "throttled_now")}, indent=2))
    if info["governor"] not in (None, "performance"):
        print(f"\n[WARN] CPU governor is '{info['governor']}', not 'performance'. "
              f"Run scripts/governor.sh performance for stable numbers.\n")

    subprocess.run([sys.executable, "-m", "fallbench.model_cost",
                    "--weights", a.weights, "--out",
                    os.path.join(a.results, "model_cost.json")], check=True)

    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = str(a.threads)
    env["PYTHONHASHSEED"] = "0"
    prefix = ["taskset", "-c", a.cpus] if a.cpus and shutil.which("taskset") else []

    rng = random.Random(a.seed)
    t_start = time.time()
    for rep in range(a.repeats):
        order = list(a.strategies)
        rng.shuffle(order)
        print(f"\n===== repeat {rep + 1}/{a.repeats}  order: {order}")
        for strat in order:
            out_csv = os.path.join(runs_dir, f"{strat}_rep{rep}.csv")
            if os.path.exists(out_csv) and not run_is_invalid(out_csv):
                print(f"-- skip {strat} rep{rep} (already done)")
                continue
            for attempt in range(a.max_retries + 1):
                t = cool_down(a.cool_temp, a.cool_min, a.cool_max)
                print(f"-- {strat} rep{rep} attempt {attempt + 1}  (start temp {t}°C)")
                cmd = prefix + [
                    sys.executable, "-m", "fallbench.run_strategy",
                    "--strategy", strat, "--k", str(a.k), "--repeat", str(rep),
                    "--videos", a.videos, "--weights", a.weights,
                    "--out", runs_dir, "--traces", os.path.join(a.results, "traces"),
                    "--threads", str(a.threads), "--limit", str(a.limit)]
                rc = subprocess.run(cmd, env=env).returncode
                if rc != 0:
                    sys.exit(f"{strat} rep{rep} crashed (exit {rc})")
                if not run_is_invalid(out_csv):
                    break
                print("   !! throttling/under-voltage detected, discarding and retrying")
            else:
                print(f"   !! {strat} rep{rep} still flagged invalid after retries; "
                      f"kept but marked invalid=1 (check cooling / power supply)")
        done = rep + 1
        el = time.time() - t_start
        print(f"   elapsed {el / 60:.1f} min, ETA ~{el / done * (a.repeats - done) / 60:.1f} min")

    frames = [open(p, newline="").read() for p in sorted(glob.glob(os.path.join(runs_dir, "*.csv")))]
    merged = os.path.join(a.results, "benchmark_runs.csv")
    with open(merged, "w", newline="") as f:
        for i, txt in enumerate(frames):
            lines = txt.splitlines(keepends=True)
            f.writelines(lines if i == 0 else lines[1:])
    print(f"\nmerged -> {merged}")
    summarize.main(a.results)


if __name__ == "__main__":
    main()
