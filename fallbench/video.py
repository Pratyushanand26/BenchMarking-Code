"""Video reading. Decode time is measured per frame so it can be reported
separately and kept out of the pipeline numbers (a deployed camera hands you
frames; decoding an mp4 from the SD card is not part of fall detection)."""
import os
import time

import cv2

from .config import VIDEO_EXTS


def list_videos(folder):
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS))
    return [os.path.join(folder, n) for n in names]


def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_frames(path, fps):
    """Yield (frame_idx, timestamp_s, frame_bgr, decode_ns)."""
    cap = cv2.VideoCapture(path)
    i = 0
    try:
        while True:
            t0 = time.perf_counter_ns()
            ok, frame = cap.read()
            dt = time.perf_counter_ns() - t0
            if not ok:
                break
            yield i, i / fps, frame, dt
            i += 1
    finally:
        cap.release()
