"""Tip discovery: which single pre-release features separate one pitch from the rest."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from pitchtip.dataset import feature_columns

READABLE = {
    "hands_x": "hands horizontal position", "hands_y": "hands height",
    "hands_to_chin": "hands height vs chin", "wrist_gap": "gap between wrists",
    "glove_wrist_y": "glove-hand height", "throw_wrist_y": "throwing-hand height",
    "glove_elbow_ang": "glove-arm elbow bend", "throw_elbow_ang": "throwing-arm elbow bend",
    "glove_elbow_out": "glove elbow flare", "throw_elbow_out": "throwing elbow flare",
    "shoulder_tilt": "shoulder tilt", "torso_lean": "torso lean", "head_x": "head horizontal",
    "head_y": "head height", "stance_width": "stance width", "lead_knee_ang": "lead knee bend",
    "still_seconds": "time held in the set", "onset_time": "time before leg lift",
    "elbow_spread": "elbow spread", "hand_motion_mean": "hand/glove motion (re-grip)",
    "hand_motion_peaks": "re-grip bursts", "hand_bright_frac": "bright pixels in glove (ball showing)",
    "throw_wrist_travel": "throwing-wrist travel",
}


def describe(col: str) -> str:
    if col.startswith("set_traj_"):
        body, _, off = col[len("set_traj_"):].rpartition("_")
        base = describe(f"set_{body}_mean").replace(" in the set", "")
        n = int(off)
        when = f"{-n} frame{'s' if n != -1 else ''} before leg lift" if n < 0 else f"{n} frames into leg lift"
        return f"{base} {when} (vs his set)"
    phase, _, rest = col.partition("_")
    for key, text in READABLE.items():
        if rest.startswith(key):
            stat = rest[len(key):].strip("_")
            where = {"set": "in the set", "early": "change at leg lift"}.get(phase, phase)
            kind = {"mean": "", "std": " (jitter)", "delta": ""}.get(stat, f" ({stat})")
            return f"{text}{kind} {where}".strip()
    return col


def _bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    q = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def find_tells(df: pd.DataFrame, y: pd.Series, top: int = 10) -> pd.DataFrame:
    """Rank (feature, pitch) pairs by how well the feature alone separates that pitch."""
    rows = []
    base = y.value_counts(normalize=True)
    for col in feature_columns(df):
        x = df[col].to_numpy()
        ok = np.isfinite(x)
        if ok.sum() < 30:
            continue
        med = np.median(x[ok])
        for c in base.index:
            pos, neg = x[ok & (y == c).to_numpy()], x[ok & (y != c).to_numpy()]
            if len(pos) < 8 or len(neg) < 8:
                continue
            u, p = mannwhitneyu(pos, neg)
            auc = u / (len(pos) * len(neg))
            side = "high" if auc > 0.5 else "low"
            half = (x > med) if side == "high" else (x <= med)
            share = float((y[ok & half] == c).mean())
            rows.append({"feature": col, "pitch": c, "auc": auc, "p": p, "side": side,
                         "threshold": med, "share_when": share, "base_rate": float(base[c]),
                         "lift": share / base[c]})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["q"] = _bh(t.p.to_numpy())
    t["strength"] = (t.auc - 0.5).abs()
    t = t.sort_values("strength", ascending=False).drop_duplicates("feature").head(top)
    t["tell"] = [
        f"When {describe(r.feature)} is {r.side} → {r.pitch} {r.share_when:.0%} of the time "
        f"(vs {r.base_rate:.0%} overall, AUC {r.auc:.2f}, q={r.q:.3g})"
        for r in t.itertuples()
    ]
    return t.reset_index(drop=True)
