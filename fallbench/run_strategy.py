"""Run every clip in --videos through ONE strategy, in ONE fresh process.

One process per (strategy, repeat) keeps peak-RSS honest: the process-level
high-water mark belongs to exactly one strategy.

    python -m fallbench.run_strategy --strategy cost_stride --k 3 --repeat 0
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

from .classifier import STGCNClassifier
from .config import DEFAULT_K, DEFAULT_WEIGHTS, STRATEGIES
from .labels import label_for, load_labels
from .pose import PoseExtractor
from .runner import _peak_rss_mb, _rss_mb, run_clip
from .video import list_videos


def warm_up(classifier, pose_factory, n_fwd=10, n_pose=5):
    """Hot allocator / thread pools / TFLite graph before anything is timed."""
    pts, mot = classifier.dummy_inputs()
    for _ in range(n_fwd):
        classifier.forward(pts, mot)
    p = pose_factory()
    gray = np.full((480, 640, 3), 127, dtype=np.uint8)
    for _ in range(n_pose):
        p.extract(gray)
    p.close()


def main(argv=None, pose_factory=PoseExtractor):
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--videos", default="videos")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--traces", default="results/traces")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only first N clips (dry run)")
    a = ap.parse_args(argv)

    if os.environ.get("FALLBENCH_FAKE_POSE"):      # test hook, see tests/
        from .testing import FakePose
        pose_factory = FakePose

    torch.set_num_threads(a.threads)
    clips = list_videos(a.videos)
    if a.limit:
        clips = clips[: a.limit]
    if not clips:
        sys.exit(f"no video files found in {a.videos}/")
    labels = load_labels(a.videos)

    classifier = STGCNClassifier(a.weights)
    warm_up(classifier, pose_factory)
    rss_after_load = _rss_mb()

    rows, traces = [], {k: [] for k in
                        ("frame_ns", "pose_ns", "recon_ns", "prep_ns", "fwd_ns",
                         "decode_ns", "steady", "arrival", "lag_ns", "clip_idx")}
    for ci, path in enumerate(clips):
        clip_id = os.path.splitext(os.path.basename(path))[0]
        gt, onset = label_for(clip_id, labels)
        row, tr = run_clip(path, clip_id, a.strategy, a.k, classifier,
                           pose_factory, gt_fall=gt, repeat=a.repeat)
        row["fall_onset_s"] = "" if onset is None else onset
        row["threads"] = a.threads
        row["rss_after_model_load_mb"] = rss_after_load
        rows.append(row)
        for key in tr:
            traces[key].append(tr[key])
        traces["clip_idx"].append(np.full(len(tr["frame_ns"]), ci, dtype=np.int32))
        print(f"[{a.strategy} rep{a.repeat}] {ci + 1}/{len(clips)} {clip_id}: "
              f"{row['frames']}f  rtf={row['rtf']:.2f}  fwd={row['forward_passes']}  "
              f"fall={'Y' if row['fall_detected'] else 'n'}"
              f"{'  INVALID(throttled)' if row['invalid'] else ''}", flush=True)

    for d in (a.out, a.traces):
        os.makedirs(d, exist_ok=True)
    out_csv = os.path.join(a.out, f"{a.strategy}_rep{a.repeat}.csv")
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_csv)
    np.savez_compressed(
        os.path.join(a.traces, f"{a.strategy}_rep{a.repeat}.npz"),
        **{k: np.concatenate(v) for k, v in traces.items()})
    print(f"[{a.strategy} rep{a.repeat}] done. process peak RSS {_peak_rss_mb():.0f} MB "
          f"-> {out_csv}")


if __name__ == "__main__":
    main()
