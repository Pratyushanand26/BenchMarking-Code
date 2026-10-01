"""Ground truth for fall/no-fall.

Priority:
  1. videos/labels.csv  with header  clip_id,fall[,fall_onset_sec]
       fall = 1/0, clip_id = file name without extension
  2. file name prefix: 'fall...' -> fall, 'adl...' -> no fall  (UR Fall style names)
  3. unknown -> accuracy columns are left blank for that clip
"""
import csv
import os


def load_labels(videos_dir):
    labels = {}
    path = os.path.join(videos_dir, "labels.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                onset = (r.get("fall_onset_sec") or "").strip()
                labels[r["clip_id"].strip()] = (
                    bool(int(r["fall"])), float(onset) if onset else None)
    return labels


def label_for(clip_id, labels):
    if clip_id in labels:
        return labels[clip_id]
    low = clip_id.lower()
    if low.startswith("fall"):
        return True, None
    if low.startswith("adl"):
        return False, None
    return None, None
