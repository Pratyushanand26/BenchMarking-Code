"""The five strategies.

Every strategy answers four questions per frame, and nothing else differs:
    pose_this_frame(i)         run MediaPipe on this frame?
    stale_pose_this_frame(i)   (no pose) append the previous pose to the buffer?
    infer_this_frame(i, new)   run the ST-GCN now? (`new` = a fresh sample was buffered)
    reconstruct(samples)       turn buffered samples into the (30,13,3) model input

Work per video-second at 30 fps, k=3 (what the smoke test asserts):

    strategy      pose calls/s   ST-GCN passes/s
    full30            30              30
    fps_hold           5               5      <- simulates a real 5 fps camera
    fps_interp         5               5
    cost_skip         10              30
    cost_stride       30              10
"""
import math

import numpy as np

from .config import DEFAULT_K, LOW_FPS, WINDOW_SIZE


class Strategy:
    name = ""
    uses_k = False

    def __init__(self, native_fps, k=DEFAULT_K):
        self.native_fps = native_fps
        self.k = k
        self.n_samples = WINDOW_SIZE          # samples that must be buffered

    def pose_this_frame(self, i):
        return True

    def stale_pose_this_frame(self, i):
        return False

    def infer_this_frame(self, i, new_sample):
        return True

    def reconstruct(self, samples):
        return samples

    # A frame "arrives" when the camera would actually hand it to the pipeline.
    # Real-time deadline = time until the next arrival.
    def is_arrival(self, i):
        return True

    @property
    def arrival_period_ns(self):
        return 1e9 / self.native_fps

    @property
    def k_label(self):
        return self.k if self.uses_k else ""


class Full30(Strategy):
    name = "full30"


class _LowFps(Strategy):
    """Simulated 5 fps camera: pose only every `keep_every` native frames,
    5 real samples span the same 1 s window, ST-GCN runs once per new sample."""

    def __init__(self, native_fps, k=DEFAULT_K):
        super().__init__(native_fps, k)
        self.keep_every = max(1, round(native_fps / LOW_FPS))     # 30 fps -> 6
        self.n_samples = math.ceil(WINDOW_SIZE / self.keep_every)  # -> 5

    def pose_this_frame(self, i):
        return i % self.keep_every == 0

    def infer_this_frame(self, i, new_sample):
        return new_sample

    def is_arrival(self, i):                 # a 5 fps camera only delivers every 6th frame
        return i % self.keep_every == 0

    @property
    def arrival_period_ns(self):
        return self.keep_every * 1e9 / self.native_fps     # 200 ms at 30 fps -> 5 fps


class FpsHold(_LowFps):
    """Zero-order hold: repeat each real sample keep_every times (staircase)."""
    name = "fps_hold"

    def reconstruct(self, samples):
        idx = np.minimum(np.arange(WINDOW_SIZE) // self.keep_every, len(samples) - 1)
        return samples[idx]


class FpsInterp(_LowFps):
    """Linear interpolation between real samples (clamped after the last one)."""
    name = "fps_interp"

    def reconstruct(self, samples):
        pos = np.minimum(np.arange(WINDOW_SIZE) / self.keep_every, len(samples) - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, len(samples) - 1)
        w = (pos - lo).astype(np.float32)[:, None, None]
        return samples[lo] * (1.0 - w) + samples[hi] * w


class CostSkip(Strategy):
    """MediaPipe only every k frames; stale pose reused in between; ST-GCN every frame."""
    name = "cost_skip"
    uses_k = True

    def pose_this_frame(self, i):
        return i % self.k == 0

    def stale_pose_this_frame(self, i):
        return True


class CostStride(Strategy):
    """MediaPipe every frame; ST-GCN only every k frames."""
    name = "cost_stride"
    uses_k = True

    def infer_this_frame(self, i, new_sample):
        return i % self.k == 0


_REGISTRY = {c.name: c for c in (Full30, FpsHold, FpsInterp, CostSkip, CostStride)}


def make_strategy(name, native_fps, k=DEFAULT_K):
    if name not in _REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; choose from {list(_REGISTRY)}")
    if k < 1:
        raise ValueError("k must be >= 1")
    return _REGISTRY[name](native_fps, k)
