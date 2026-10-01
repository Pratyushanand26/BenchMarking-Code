"""MediaPipe Pose -> 13-joint skeleton (the joints the TSSTG checkpoint uses)."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

from .config import MP_TO_COCO13


class PoseExtractor:
    """One instance per clip, so MediaPipe tracking state never leaks across clips."""

    def __init__(self):
        import mediapipe as mp  # lazy: tests can run without it
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This mediapipe build has no legacy `mp.solutions` API. "
                "Install the pinned version: pip install -r requirements.txt"
            )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr):
        """BGR frame -> (13, 3) float32 [x_px, y_px, visibility], or None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks.landmark
        out = np.zeros((13, 3), dtype=np.float32)
        for i, idx in enumerate(MP_TO_COCO13):
            lm = lms[idx]
            out[i] = (lm.x * w, lm.y * h, lm.visibility)
        return out

    def close(self):
        self._pose.close()
