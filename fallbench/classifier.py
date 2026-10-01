"""Pretrained ST-GCN wrapper, split into separately-timeable stages:

    prepare(window, w, h)  -> normalise + neck joint + tensors   (CPU, numpy/torch)
    forward(pts_t, mot_t)  -> class probabilities                (the ST-GCN pass)

Preprocessing is identical to the original working implementation.
"""
import numpy as np
import torch

from .config import ACTION_CLASSES, WINDOW_SIZE
from .model import (
    TwoStreamSpatialTemporalGraph,
    normalize_points_with_size,
    scale_pose,
)


class STGCNClassifier:
    def __init__(self, weights_path):
        self.device = torch.device("cpu")
        self.model = TwoStreamSpatialTemporalGraph(
            graph_args={"strategy": "spatial"}, num_class=len(ACTION_CLASSES)
        ).to(self.device)

        ckpt = torch.load(weights_path, map_location=self.device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        res = self.model.load_state_dict(state, strict=False)

        # 'A' is a fixed graph buffer rebuilt in __init__; anything else missing
        # means we'd be benchmarking random weights -> refuse.
        missing = [k for k in res.missing_keys if not k.endswith(".A") and k != "A"]
        if missing:
            raise RuntimeError(
                f"Checkpoint does not match architecture; missing keys: {missing[:5]}"
            )
        self.model.eval()

    # ---- stage 1: preprocessing -------------------------------------------
    def prepare(self, window, frame_w, frame_h):
        """window: (30, 13, 3) -> (points (1,3,30,14), motion (1,2,29,14))."""
        pts = np.asarray(window, dtype=np.float32).copy()
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], frame_w, frame_h)
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        neck = np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, axis=1)
        pts = np.concatenate((pts, neck), axis=1)                      # (30, 14, 3)

        pts_t = torch.from_numpy(np.ascontiguousarray(pts)).permute(2, 0, 1).unsqueeze(0)
        mot_t = pts_t[:, :2, 1:, :] - pts_t[:, :2, :-1, :]             # (1, 2, 29, 14)
        return pts_t, mot_t

    # ---- stage 2: the ST-GCN forward pass ---------------------------------
    def forward(self, pts_t, mot_t):
        with torch.inference_mode():
            out = self.model((pts_t, mot_t))        # sigmoid already applied
        return out.squeeze(0).numpy()

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def top_class(probs):
        return ACTION_CLASSES[int(np.argmax(probs))]

    def dummy_inputs(self):
        pts = torch.randn(1, 3, WINDOW_SIZE, 14)
        mot = torch.randn(1, 2, WINDOW_SIZE - 1, 14)
        return pts, mot
