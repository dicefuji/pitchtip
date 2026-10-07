"""Turn a pitcher pose track into a fixed-length, human-readable feature vector.

Every feature is in pitcher-height units relative to the mid-hip point (or degrees),
so they transfer across camera zoom levels and are explainable as a tell.
"""
from __future__ import annotations

import numpy as np

from pitchtip.vision.phases import Phases, lead_side
from pitchtip.vision.pose import KP, ClipPose


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for (T,2) arrays."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _tilt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def per_frame(cp: ClipPose, ph: Phases, hand: str | None) -> dict[str, np.ndarray]:
    k = cp.kpts[..., :2]
    g, t = lead_side(hand), ("l" if lead_side(hand) == "r" else "r")  # glove side, throwing side
    p = lambda name: k[:, KP[name]]
    hip = (p("l_hip") + p("r_hip")) / 2
    sh = (p("l_shoulder") + p("r_shoulder")) / 2
    rel = lambda name: (p(name) - hip) / ph.scale
    hands = (rel("l_wrist") + rel("r_wrist")) / 2
    return {
        "hands_x": hands[:, 0],
        "hands_y": -hands[:, 1],
        "hands_to_chin": -(((p("l_wrist") + p("r_wrist")) / 2 - p("nose"))[:, 1]) / ph.scale,
        "wrist_gap": np.linalg.norm(p("l_wrist") - p("r_wrist"), axis=-1) / ph.scale,
        "glove_wrist_y": -rel(f"{g}_wrist")[:, 1],
        "throw_wrist_y": -rel(f"{t}_wrist")[:, 1],
        "glove_elbow_ang": _angle(p(f"{g}_shoulder"), p(f"{g}_elbow"), p(f"{g}_wrist")),
        "throw_elbow_ang": _angle(p(f"{t}_shoulder"), p(f"{t}_elbow"), p(f"{t}_wrist")),
        "glove_elbow_out": (p(f"{g}_elbow") - p(f"{g}_shoulder"))[:, 0] / ph.scale,
        "throw_elbow_out": (p(f"{t}_elbow") - p(f"{t}_shoulder"))[:, 0] / ph.scale,
        "shoulder_tilt": _tilt(p("l_shoulder"), p("r_shoulder")),
        "torso_lean": np.degrees(np.arctan2((sh - hip)[:, 0], -(sh - hip)[:, 1])),
        "head_x": rel("nose")[:, 0],
        "head_y": -rel("nose")[:, 1],
        "stance_width": np.abs(p("l_ankle") - p("r_ankle"))[:, 0] / ph.scale,
        "lead_knee_ang": _angle(p(f"{g}_hip"), p(f"{g}_knee"), p(f"{g}_ankle")),
    }


def clip_features(cp: ClipPose, ph: Phases, hand: str | None) -> tuple[dict[str, float], np.ndarray]:
    """Return (named scalar features, mean hands/glove grayscale patch over the set)."""
    pf = per_frame(cp, ph, hand)
    s, e, o = ph.set_start, ph.early_end, ph.onset
    feats: dict[str, float] = {}
    for name, v in pf.items():
        st = v[s:o]
        feats[f"set_{name}_mean"] = float(np.nanmean(st)) if np.isfinite(st).any() else np.nan
        feats[f"set_{name}_std"] = float(np.nanstd(st)) if np.isfinite(st).any() else np.nan
        ea = v[o:e]
        feats[f"early_{name}_delta"] = (float(np.nanmean(ea) - feats[f"set_{name}_mean"])
                                         if np.isfinite(ea).any() else np.nan)
    # Set duration: how long the hands have held their final set height before the lift.
    hy = pf["hands_y"]
    ref = np.nanmedian(hy[max(o - 3, 0):o])
    n = 0
    for i in range(o - 1, -1, -1):
        if np.isfinite(hy[i]) and abs(hy[i] - ref) > 0.03:
            break
        n += 1
    feats["set_still_seconds"] = n / cp.fps
    feats["onset_time"] = float(cp.t[o])
    patch = cp.hands[s:o].astype(np.float32).mean(axis=0) / 255.0
    return feats, patch
