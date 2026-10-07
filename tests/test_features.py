import numpy as np
import pandas as pd

from pitchtip import features
from pitchtip.labels import make_labels
from pitchtip.tips import find_tells
from pitchtip.vision import phases
from pitchtip.vision.pose import HAND_PATCH, KP, ClipPose

FPS = 15


def synthetic_clip(onset_s: float = 2.0, hands_y: float = 0.0, T: int = 60) -> ClipPose:
    """A stick figure standing still, then lifting the left knee (RHP) at onset_s."""
    t = np.arange(T) / FPS
    k = np.zeros((T, 17, 3)); k[..., 2] = 1
    base = {
        "nose": (0, -170), "l_shoulder": (-20, -140), "r_shoulder": (20, -140),
        "l_elbow": (-25, -110), "r_elbow": (25, -110), "l_wrist": (-3, -100 - hands_y),
        "r_wrist": (3, -100 - hands_y), "l_hip": (-12, 0), "r_hip": (12, 0),
        "l_knee": (-12, 50), "r_knee": (12, 50), "l_ankle": (-15, 100), "r_ankle": (15, 100),
        "l_eye": (-3, -175), "r_eye": (3, -175), "l_ear": (-6, -172), "r_ear": (6, -172),
    }
    for name, (x, y) in base.items():
        k[:, KP[name], 0] = 500 + x
        k[:, KP[name], 1] = 400 + y
    lift = t >= onset_s
    k[lift, KP["l_knee"], 1] = 400 + 50 - np.minimum((t[lift] - onset_s) * 300, 60)
    boxes = np.tile([450, 220, 550, 500], (T, 1)).astype(float)
    return ClipPose(FPS, t, k, boxes, np.zeros((T, HAND_PATCH, HAND_PATCH), np.uint8), (1280, 720))


def test_onset_detected_before_knee_rises():
    cp = synthetic_clip(onset_s=2.0)
    ph = phases.segment(cp, "R")
    assert ph is not None
    assert abs(cp.t[ph.onset] - 2.0) <= 2 / FPS
    assert ph.early_end - ph.onset <= round(0.33 * FPS) + 1


def test_no_onset_when_still():
    cp = synthetic_clip(onset_s=99)
    assert phases.segment(cp, "R") is None


def test_hands_height_feature_moves_with_hands():
    lo = synthetic_clip(hands_y=0)
    hi = synthetic_clip(hands_y=30)
    f_lo, _ = features.clip_features(lo, phases.segment(lo, "R"), "R")
    f_hi, _ = features.clip_features(hi, phases.segment(hi, "R"), "R")
    assert f_hi["set_hands_y_mean"] > f_lo["set_hands_y_mean"] + 0.05
    assert f_lo["set_still_seconds"] > 1.0


def test_find_tells_recovers_planted_tell():
    rng = np.random.default_rng(0)
    n = 300
    y = pd.Series(rng.choice(["FF", "SL"], n, p=[0.6, 0.4]))
    df = pd.DataFrame({
        "set_hands_y_mean": rng.normal(0, 1, n) + (y == "SL") * 1.5,
        "set_head_x_mean": rng.normal(0, 1, n),
    })
    t = find_tells(df, y, top=2)
    assert t.iloc[0].feature == "set_hands_y_mean"
    assert t.iloc[0].q < 1e-6


def test_labels_fb_mode():
    y = make_labels(pd.Series(["FF", "SI", "SL", "CU"]), "fb")
    assert list(y) == ["FASTBALL", "FASTBALL", "OFFSPEED", "OFFSPEED"]
