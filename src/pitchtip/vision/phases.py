"""Delivery phase segmentation from a pitcher pose track.

We only ever look at frames up to shortly after leg-lift onset. Release happens
~0.8-1.3s after onset, so ball flight and arm action can never leak into features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitchtip.vision.pose import KP, ClipPose


@dataclass
class Phases:
    onset: int          # frame index where the leg lift starts
    set_start: int      # first frame of the set window
    early_end: int      # last frame of the early-lift window (exclusive)
    scale: float        # pitcher height in px during the set (normalizer)


def lead_side(hand: str | None) -> str:
    """Lead (stride) leg/glove side: left for RHP, right for LHP."""
    return "r" if hand == "L" else "l"


def knee_lift(cp: ClipPose, hand: str | None) -> np.ndarray:
    s = lead_side(hand)
    hip = cp.kpts[:, KP[f"{s}_hip"], 1]
    knee = cp.kpts[:, KP[f"{s}_knee"], 1]
    h = cp.boxes[:, 3] - cp.boxes[:, 1]
    return (hip - knee) / h  # rises toward 0 / positive as the knee comes up


def segment(cp: ClipPose, hand: str | None, set_seconds: float = 1.2,
            early_seconds: float = 0.33, rise: float = 0.09) -> Phases | None:
    lift = knee_lift(cp, hand)
    valid = np.isfinite(lift)
    if valid.sum() < 10:
        return None
    first = int(np.argmax(valid))
    base = np.nanmedian(lift[first:first + max(int(cp.fps), 5)])
    above = np.nan_to_num(lift - base, nan=-1) > rise
    # Onset = first time the knee is clearly up for 2 consecutive samples; back up to
    # the start of that rise so the window ends before any real movement.
    hits = np.flatnonzero(above[:-1] & above[1:])
    if len(hits) == 0:
        return None
    onset = int(hits[0])
    while onset > first and np.nan_to_num(lift[onset - 1] - base, nan=0) > rise / 3:
        onset -= 1
    set_start = max(first, onset - int(round(set_seconds * cp.fps)))
    if onset - set_start < 3:
        return None
    early_end = min(len(cp.t), onset + int(round(early_seconds * cp.fps)) + 1)
    h = cp.boxes[set_start:onset, 3] - cp.boxes[set_start:onset, 1]
    return Phases(onset, set_start, early_end, float(np.nanmedian(h)))
