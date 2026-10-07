"""Pitch-type label handling."""
from __future__ import annotations

import pandas as pd

FASTBALLS = {"FF", "SI", "FC", "FT", "FA"}


def make_labels(pt: pd.Series, mode: str = "type", min_share: float = 0.05) -> pd.Series:
    """mode='type': pitcher's real arsenal (rare pitches -> OTHER); 'fb': fastball vs offspeed."""
    if mode == "fb":
        return pt.map(lambda p: "FASTBALL" if p in FASTBALLS else "OFFSPEED")
    share = pt.value_counts(normalize=True)
    keep = set(share[share >= min_share].index)
    return pt.where(pt.isin(keep), "OTHER")
