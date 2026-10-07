"""Run the tip detector on a stream or a long recorded game.

Keeps a rolling pose buffer of the CF shot; when a leg lift starts, it builds features
from the buffer and predicts ~0.4s after onset (still before release).
"""
from __future__ import annotations

import collections
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchtip import features
from pitchtip.vision import phases, pose


def resolve_source(src: str) -> str:
    """Local files pass through; web pages (YouTube, Twitch, etc.) go through yt-dlp."""
    if Path(src).exists() or not src.startswith("http") or src.endswith((".mp4", ".m3u8")):
        return src
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is required to open stream pages")
    out = subprocess.run(["yt-dlp", "-g", "-f", "best[height<=720]/best", src],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[0]


def run(src: str, hand: str | None, model, pitcher: str, target_fps: float = 15.0,
        buffer_seconds: float = 4.0, log: Path | None = None, decide=None):
    """model: a local TipModel; decide: optional callable(features_row) -> probs (e.g. Jev)."""
    url = resolve_source(src)
    cap = cv2.VideoCapture(url)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / target_fps)), 1)
    n = int(buffer_seconds * target_fps)
    buf = collections.deque(maxlen=n)
    prev_box, cooldown_until, i = None, -1.0, 0
    log_f = open(log, "a") if log else None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        t = i / fps
        kp, box = pose.pose_frame(frame, prev_box)
        if kp is None:
            buf.clear(); prev_box = None
            continue
        prev_box = box
        kp = kp.copy(); kp[kp[:, 2] < 0.3, :2] = np.nan
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buf.append((t, kp, box, pose._hands_patch(gray, kp, box)))
        if t < cooldown_until or len(buf) < n // 2:
            continue
        ts, kps, bxs, hs = map(np.array, zip(*buf))
        cp = pose.ClipPose(target_fps, ts, kps, bxs, hs, frame.shape[1::-1])
        ph = phases.segment(cp, hand)
        # Fire once the early-lift window is complete.
        if ph is None or ph.early_end >= len(ts):
            continue
        f, patch = features.clip_features(cp, ph, hand)
        row = pd.DataFrame([f])
        t0 = time.perf_counter()
        proba = (decide(row) if decide else
                 dict(zip(model.clf.classes_, model.predict_proba(row, patch.ravel()[None])[0])))
        dt = time.perf_counter() - t0
        best = max(proba, key=proba.get)
        msg = {"t": round(float(ts[ph.onset]), 2), "pitcher": pitcher, "call": best,
               "proba": {k: round(float(v), 3) for k, v in proba.items()}, "decide_ms": round(dt * 1000, 1)}
        print(f"[{msg['t']:7.2f}s] leg lift → {best:>8}  " +
              "  ".join(f"{k}:{v:.0%}" for k, v in sorted(proba.items(), key=lambda kv: -kv[1])))
        if log_f:
            log_f.write(json.dumps(msg) + "\n"); log_f.flush()
        cooldown_until = t + 6.0
        buf.clear()
    cap.release()
