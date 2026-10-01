"""Sanity test that needs NO videos, NO real weights and NO MediaPipe.

    python tests/smoke_test.py

It builds a synthetic 90-frame / 30 fps clip, a randomly-initialised checkpoint
and a fake pose extractor, runs all 5 strategies through the real runner, and
asserts the exact work counts documented in fallbench/strategies.py. If this
passes, the strategy logic and metric plumbing are correct on this machine.
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fallbench.classifier import STGCNClassifier
from fallbench.config import ACTION_CLASSES, STRATEGIES
from fallbench.model import TwoStreamSpatialTemporalGraph
from fallbench.runner import run_clip
from fallbench.testing import FakePose

N_FRAMES, FPS, K = 90, 30, 3


def main():
    with tempfile.TemporaryDirectory() as d:
        vid = os.path.join(d, "synthetic.mp4")
        vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 120))
        for i in range(N_FRAMES):
            vw.write(np.full((120, 160, 3), i * 2 % 255, dtype=np.uint8))
        vw.release()

        w = os.path.join(d, "w.pth")
        torch.manual_seed(0)
        m = TwoStreamSpatialTemporalGraph({"strategy": "spatial"}, len(ACTION_CLASSES))
        torch.save(m.state_dict(), w)
        clf = STGCNClassifier(w)

        expected = {   # (pose_calls, forward_passes) for 90 frames @ 30 fps, k=3
            "full30": (90, 61),
            "fps_hold": (15, 11),
            "fps_interp": (15, 11),
            "cost_skip": (30, 61),
            "cost_stride": (90, 20),
        }
        ok = True
        print(f"{'strategy':12s} {'pose':>5s} {'fwd':>5s}  expected      rtf   status")
        for s in STRATEGIES:
            row, tr = run_clip(vid, "synthetic", s, K, clf, FakePose, gt_fall=False)
            got = (row["pose_calls"], row["forward_passes"])
            good = got == expected[s] and row["frames"] == N_FRAMES
            ok &= good
            print(f"{s:12s} {got[0]:5d} {got[1]:5d}  {str(expected[s]):10s} "
                  f"{row['rtf']:6.2f}   {'OK' if good else 'FAIL'}")
            assert len(tr["frame_ns"]) == N_FRAMES
        if not ok:
            sys.exit("SMOKE TEST FAILED")
        print("\nall strategies produce the expected work counts")


if __name__ == "__main__":
    main()
