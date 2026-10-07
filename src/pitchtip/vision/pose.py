"""Per-frame pitcher pose from a broadcast clip.

The center-field camera puts the pitcher closest to the lens, so the pitcher is the
largest detected person while the shot is on the CF view. Frames on any other shot
(close-ups, replays) are dropped by a size gate and a hard-cut detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# COCO-17 keypoint indices used by YOLO pose models.
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}
HAND_PATCH = 24  # side of the grayscale hands/glove patch kept per frame

_model = None
_device = None


def get_model(weights: str = "yolo11s-pose.pt"):
    global _model, _device
    if _model is None:
        import torch
        from ultralytics import YOLO
        _model = YOLO(weights)
        _device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return _model


@dataclass
class ClipPose:
    fps: float            # effective fps of the sampled frames
    t: np.ndarray         # (T,) seconds from clip start
    kpts: np.ndarray      # (T, 17, 3) x, y, conf in pixels; NaN where pitcher missing
    boxes: np.ndarray     # (T, 4) x1, y1, x2, y2
    hands: np.ndarray     # (T, HAND_PATCH, HAND_PATCH) uint8 grayscale glove/hands crop
    frame_size: tuple[int, int]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fps=self.fps, t=self.t, kpts=self.kpts, boxes=self.boxes,
                            hands=self.hands, frame_size=np.array(self.frame_size))

    @classmethod
    def load(cls, path: Path) -> "ClipPose":
        z = np.load(path)
        return cls(float(z["fps"]), z["t"], z["kpts"], z["boxes"], z["hands"],
                   tuple(int(v) for v in z["frame_size"]))


def _hands_patch(gray: np.ndarray, kp: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Crop around the midpoint of both wrists (where the glove sits in the set)."""
    h = box[3] - box[1]
    wrists = kp[[KP["l_wrist"], KP["r_wrist"]], :2]
    cx, cy = np.nanmean(wrists, axis=0)
    if not np.isfinite(cx):
        return np.zeros((HAND_PATCH, HAND_PATCH), np.uint8)
    r = max(int(0.12 * h), 4)
    y1, y2 = int(max(cy - r, 0)), int(min(cy + r, gray.shape[0]))
    x1, x2 = int(max(cx - r, 0)), int(min(cx + r, gray.shape[1]))
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((HAND_PATCH, HAND_PATCH), np.uint8)
    return cv2.resize(crop, (HAND_PATCH, HAND_PATCH), interpolation=cv2.INTER_AREA)


def pick_pitcher(boxes: np.ndarray, frame_h: int, prev_box: np.ndarray | None) -> int | None:
    """Index of the pitcher among detected person boxes, or None if not a CF shot."""
    if len(boxes) == 0:
        return None
    heights = boxes[:, 3] - boxes[:, 1]
    # CF view: pitcher is roughly 20-75% of frame height. Larger means a close-up.
    ok = (heights > 0.18 * frame_h) & (heights < 0.75 * frame_h)
    if not ok.any():
        return None
    if prev_box is not None:
        # Prefer continuity with the previous pitcher box when one is close by.
        pc = (prev_box[:2] + prev_box[2:]) / 2
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        dist = np.linalg.norm(centers - pc, axis=1) / frame_h
        near = ok & (dist < 0.15)
        if near.any():
            return int(np.argmax(np.where(near, heights, -1)))
    return int(np.argmax(np.where(ok, heights, -1)))


def is_hard_cut(prev_gray: np.ndarray | None, gray: np.ndarray, thresh: float = 0.5) -> bool:
    if prev_gray is None:
        return False
    h1 = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
    h2 = cv2.calcHist([gray], [0], None, [32], [0, 256])
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) > thresh


def frames(source, target_fps: float = 15.0, max_seconds: float | None = 8.0):
    """Yield (t_seconds, bgr_frame) sampled at ~target_fps from a file path or stream URL."""
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / target_fps)), 1)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if max_seconds is not None and t > max_seconds:
            break
        if i % step == 0:
            yield t, frame
        i += 1
    cap.release()


def pose_frame(frame: np.ndarray, prev_box: np.ndarray | None, conf: float = 0.3):
    """Run pose on one frame; returns (kpts (17,3), box (4,)) for the pitcher or (None, None)."""
    res = get_model()(frame, verbose=False, conf=conf, device=_device)[0]
    if res.keypoints is None or len(res.boxes) == 0:
        return None, None
    boxes = res.boxes.xyxy.cpu().numpy()
    idx = pick_pitcher(boxes, frame.shape[0], prev_box)
    if idx is None:
        return None, None
    xy = res.keypoints.xy.cpu().numpy()[idx]
    kc = res.keypoints.conf
    c = kc.cpu().numpy()[idx] if kc is not None else np.ones(17)
    return np.concatenate([xy, c[:, None]], axis=1), boxes[idx]


def extract(clip: Path, target_fps: float = 15.0, max_seconds: float = 8.0) -> ClipPose:
    ts, kps, bxs, hands = [], [], [], []
    prev_gray, prev_box, size, seen_cf = None, None, (0, 0), False
    for t, frame in frames(clip, target_fps, max_seconds):
        size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        if seen_cf and is_hard_cut(prev_gray, gray):
            break  # first cut after the CF view = end of the pitch shot
        prev_gray = gray
        kp, box = pose_frame(frame, prev_box)
        ts.append(t)
        if kp is None:
            kps.append(np.full((17, 3), np.nan)); bxs.append(np.full(4, np.nan))
            hands.append(np.zeros((HAND_PATCH, HAND_PATCH), np.uint8))
            continue
        seen_cf = True
        prev_box = box
        kp = kp.copy()
        kp[kp[:, 2] < 0.3, :2] = np.nan
        kps.append(kp); bxs.append(box)
        hands.append(_hands_patch(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), kp, box))
    eff_fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else target_fps
    return ClipPose(eff_fps, np.array(ts), np.array(kps).reshape(-1, 17, 3),
                    np.array(bxs).reshape(-1, 4), np.array(hands).reshape(-1, HAND_PATCH, HAND_PATCH), size)
