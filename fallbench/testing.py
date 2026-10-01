"""Test-only helpers (no MediaPipe needed)."""
import numpy as np


class FakePose:
    """Returns a random but valid 13-joint skeleton for every frame."""
    rng = np.random.default_rng(0)

    def extract(self, frame):
        kp = self.rng.uniform(0.1, 0.9, (13, 3)).astype(np.float32)
        kp[:, 0] *= frame.shape[1]
        kp[:, 1] *= frame.shape[0]
        return kp

    def close(self):
        pass
