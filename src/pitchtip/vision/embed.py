"""DINOv2 embeddings of the glove region: grip visibility, glove shape, finger exposure.

Per pitch we embed the color glove crops from the last ~0.6s of the set and from the
early leg lift, and keep (a) the mean embedding of each window and (b) how much the
embedding changes frame to frame during the set (re-grip / fidget energy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pitchtip import config
from pitchtip.config import slug
from pitchtip.vision import phases, pose

MODEL = "facebook/dinov2-small"
SIZE = 112  # 8x8 patches of 14px
_m = None


def _model():
    global _m
    if _m is None:
        from transformers import AutoModel
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _m = (AutoModel.from_pretrained(MODEL).eval().to(dev), dev)
    return _m


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@torch.no_grad()
def embed_crops(crops: np.ndarray, batch: int = 256) -> np.ndarray:
    """(N, H, W, 3) BGR uint8 -> (N, 768) [CLS ‖ mean patch token]."""
    import cv2
    m, dev = _model()
    out = []
    for i in range(0, len(crops), batch):
        x = np.stack([cv2.resize(c[..., ::-1], (SIZE, SIZE)) for c in crops[i:i + batch]])
        x = ((x.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)
        h = m(pixel_values=torch.from_numpy(x).to(dev)).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def clip_embedding(cp: pose.ClipPose, ph: phases.Phases, set_frames: int = 9) -> np.ndarray:
    """-> (1537,) = set-window mean (768) ‖ early-window mean (768) ‖ set motion energy (1)."""
    g = cp.glove
    o = ph.onset - cp.glove_start
    e = ph.early_end - cp.glove_start
    if len(g) == 0 or o <= 1:
        return np.full(1537, np.nan, np.float32)
    s = max(0, o - set_frames)
    valid = g.reshape(len(g), -1).max(1) > 0
    emb = embed_crops(g)
    st = emb[s:o][valid[s:o]]
    ea = emb[o:e][valid[o:e]]
    if len(st) == 0:
        return np.full(1537, np.nan, np.float32)
    motion = float(np.linalg.norm(np.diff(st, axis=0), axis=1).mean()) if len(st) > 1 else 0.0
    early = ea.mean(0) if len(ea) else st.mean(0)
    return np.concatenate([st.mean(0), early, [motion]]).astype(np.float32)


def build(key: str) -> np.ndarray:
    s = slug(key)
    df = pd.read_parquet(config.FEATURES_DIR / f"{s}.parquet")
    hand = df.pitcher_hand.iloc[0]
    out = []
    for r in tqdm(df.itertuples(), total=len(df), desc=f"embed {key}"):
        cp = pose.ClipPose.load(config.POSE_DIR / slug(r.pitcher_name) / f"{r.play_id}.npz")
        ph = phases.segment(cp, hand)
        out.append(clip_embedding(cp, ph) if ph is not None else np.full(1537, np.nan, np.float32))
    arr = np.stack(out).astype(np.float32)
    np.save(config.FEATURES_DIR / f"{s}_emb.npy", arr)
    return arr


def load(key: str, per_game: bool = True, df: pd.DataFrame | None = None) -> np.ndarray:
    arr = np.load(config.FEATURES_DIR / f"{slug(key)}_emb.npy")
    arr = np.where(np.isfinite(arr), arr, np.nanmean(arr, 0))
    if per_game and df is not None:
        for _, idx in df.groupby("game_pk").indices.items():
            arr[idx] -= arr[idx].mean(0)
    return arr
