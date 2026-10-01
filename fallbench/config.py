"""Shared constants. Everything that defines the pipeline lives here."""

WINDOW_SIZE = 30            # frames the pretrained ST-GCN expects
LOW_FPS = 5                 # simulated camera rate for fps_hold / fps_interp
DEFAULT_K = 3               # k for cost_skip / cost_stride

STRATEGIES = ["full30", "fps_hold", "fps_interp", "cost_skip", "cost_stride"]

ACTION_CLASSES = [
    "Standing", "Walking", "Sitting", "Lying Down",
    "Stand up", "Sit down", "Fall Down",
]
FALL_CLASS = "Fall Down"

# MediaPipe Pose landmark index -> the 13 joints the TSSTG checkpoint uses
# (Nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle)
MP_TO_COCO13 = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

DEFAULT_WEIGHTS = "weights/tsstg-model.pth"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
